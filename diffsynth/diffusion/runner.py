import os, torch, tempfile, numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _init_wandb(accelerator, args):
    """Initialize wandb if wandb_project is set and on main process. Returns wandb module or None."""
    if args is None or getattr(args, "wandb_project", None) is None or not accelerator.is_main_process:
        return None
    import wandb
    os.environ["WANDB_MODE"] = getattr(args, "wandb_mode", "online")
    experiment_name = getattr(args, "experiment_name", None)
    if experiment_name is None:
        output_path = getattr(args, "output_path", None)
        if output_path is not None:
            experiment_name = os.path.basename(os.path.normpath(output_path))
        else:
            experiment_name = "default_exp"
    log_dir = os.path.join("logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    wandb.init(
        project=args.wandb_project,
        name=experiment_name,
        dir=log_dir,
        config=vars(args),
    )
    return wandb


def _log_wandb_video(wandb, pipe, data, extra_inputs, step):
    """Generate a sample video from the current batch and log it to wandb."""
    try:
        # Build inference kwargs from training data
        kwargs = {"prompt": data.get("prompt", ""), "num_inference_steps": 20, "seed": 0}
        video_frames = data.get("video")
        if video_frames is not None and len(video_frames) > 0:
            kwargs["height"] = video_frames[0].size[1]
            kwargs["width"] = video_frames[0].size[0]
            kwargs["num_frames"] = len(video_frames)
        # Pass extra inputs (e.g. vace_video, vace_reference_image, input_image, etc.)
        for key in extra_inputs:
            if key in data:
                val = data[key]
                if key.endswith("_image") and isinstance(val, list):
                    val = val[0]
                kwargs[key] = val
        # Run inference
        pipe.scheduler.set_timesteps(kwargs.get("num_inference_steps", 20), training=False)
        frames = pipe(**kwargs, progress_bar_cmd=lambda x: x)
        # Convert PIL frames to video array (T, H, W, C) -> (T, C, H, W) for wandb
        video_array = np.stack([np.array(f) for f in frames])  # (T, H, W, C)
        video_array = video_array.transpose(0, 3, 1, 2)  # (T, C, H, W)
        wandb.log({"sample_video": wandb.Video(video_array, fps=16, format="mp4")}, step=step)
    except Exception as e:
        print(f"[wandb] Failed to generate sample video at step {step}: {e}")


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    # Wandb
    wandb = _init_wandb(accelerator, args)
    wandb_log_steps = getattr(args, "wandb_log_steps", 100) if args is not None else 100
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                # Wandb logging
                if wandb is not None and model_logger.num_steps % wandb_log_steps == 0:
                    wandb.log({"loss": loss.item(), "epoch": epoch_id}, step=model_logger.num_steps)
                    unwrapped = accelerator.unwrap_model(model)
                    pipe = unwrapped.pipe
                    extra_inputs = getattr(unwrapped, "extra_inputs", [])
                    pipe.scheduler.set_timesteps(1000, training=True)
                    with torch.no_grad():
                        _log_wandb_video(wandb, pipe, data, extra_inputs, model_logger.num_steps)
                    pipe.scheduler.set_timesteps(1000, training=True)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
    if wandb is not None:
        wandb.finish()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
