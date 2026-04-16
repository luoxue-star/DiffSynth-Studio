import os, torch, tempfile, numpy as np, random
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
    
    wandb_run_id = getattr(args, "wandb_run_id", None)
    init_kwargs = {
        "project": args.wandb_project,
        "name": experiment_name,
        "dir": log_dir,
        "config": vars(args),
    }
    if wandb_run_id is not None:
        init_kwargs["id"] = wandb_run_id
        init_kwargs["resume"] = "must"
        
    wandb.init(**init_kwargs)
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


def _run_evaluation(accelerator, model, eval_dataset, eval_save_path, step, wandb=None,
                    num_inference_steps=50, seed=0, extra_inputs=None, max_eval_samples=5):
    """Run evaluation on eval dataset and save videos (main process only)."""
    if not accelerator.is_main_process:
        return

    from ..utils.data import save_video

    unwrapped = accelerator.unwrap_model(model)
    pipe = unwrapped.pipe
    extra_inputs = extra_inputs or []

    # Switch to inference mode
    pipe.scheduler.set_timesteps(num_inference_steps, training=False)

    step_save_path = os.path.join(eval_save_path, f"step-{step}")
    os.makedirs(step_save_path, exist_ok=True)

    # Randomly sample if dataset is larger than max_eval_samples
    total = len(eval_dataset)
    if max_eval_samples is not None and total > max_eval_samples:
        sample_indices = sorted(random.sample(range(total), max_eval_samples))
    else:
        sample_indices = list(range(total))

    accelerator.print(f"[Evaluation] Running evaluation at step {step} on {len(sample_indices)}/{total} samples...")

    wandb_videos = []
    for idx in tqdm(sample_indices, desc="Evaluating"):
        data = eval_dataset[idx]
        try:
            # Build inference kwargs
            kwargs = {
                "prompt": data.get("prompt", ""),
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "tiled": True,
            }
            video_frames = data.get("video")
            if video_frames and len(video_frames) > 0:
                kwargs["height"] = video_frames[0].size[1]
                kwargs["width"] = video_frames[0].size[0]
                kwargs["num_frames"] = len(video_frames)

            # Pass extra inputs
            for key in extra_inputs:
                if key in data:
                    val = data[key]
                    if key.endswith("_image") and isinstance(val, list):
                        val = val[0]
                    kwargs[key] = val

            # Run inference
            with torch.no_grad():
                frames = pipe(**kwargs, progress_bar_cmd=lambda x: x)

            # Save video to local
            video_path = os.path.join(step_save_path, f"eval_{idx}.mp4")
            save_video(frames, video_path, fps=15, quality=5)

            # Collect for wandb
            if wandb is not None:
                video_array = np.stack([np.array(f) for f in frames]).transpose(0, 3, 1, 2)
                wandb_videos.append(wandb.Video(video_array, fps=16, format="mp4", caption=data.get("prompt", "")))

        except Exception as e:
            accelerator.print(f"[Evaluation] Failed to evaluate sample {idx}: {e}")

    # Log to wandb
    if wandb is not None and wandb_videos:
        wandb.log({"eval_videos": wandb_videos}, step=step)

    # Switch back to training mode
    pipe.scheduler.set_timesteps(1000, training=True)

    accelerator.print(f"[Evaluation] Completed. Videos saved to {step_save_path}")


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = 1000,
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

    # Evaluation config
    eval_dataset = None
    eval_steps = None
    eval_num_inference_steps = 50
    eval_seed = 0
    eval_save_path = None
    eval_extra_inputs = []
    eval_max_samples = 5
    if args is not None and getattr(args, "eval_metadata_path", None) is not None:
        from ..core import UnifiedDataset
        eval_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.eval_metadata_path,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                height=getattr(args, "height", None),
                width=getattr(args, "width", None),
                num_frames=getattr(args, "num_frames", 81),
            ),
        )
        eval_steps = getattr(args, "eval_steps", 5000)
        eval_num_inference_steps = getattr(args, "eval_num_inference_steps", 50)
        eval_seed = getattr(args, "eval_seed", 0)
        eval_save_path = getattr(args, "eval_save_path", None)
        if eval_save_path is None:
            experiment_name = getattr(args, "experiment_name", None) or "default_exp"
            eval_save_path = os.path.join("logs", experiment_name, "eval_videos")
        eval_extra_inputs = args.extra_inputs.split(",") if getattr(args, "extra_inputs", None) else []
        eval_max_samples = getattr(args, "eval_max_samples", 5) or None  # 0 means no limit
        accelerator.print(f"[Evaluation] Enabled: {len(eval_dataset)} samples (max {eval_max_samples or 'all'} per eval), every {eval_steps} steps, {eval_num_inference_steps} inference steps")

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    accelerator.register_for_checkpointing(model_logger)

    if getattr(args, "resume_from_checkpoint", None) is not None:
        accelerator.print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)

    start_epoch = model_logger.num_steps // len(dataloader) if len(dataloader) > 0 else 0
    start_step = model_logger.num_steps % len(dataloader) if len(dataloader) > 0 else 0

    for epoch_id in range(start_epoch, num_epochs):
        active_dataloader = accelerator.skip_first_batches(dataloader, start_step) if epoch_id == start_epoch and start_step > 0 else dataloader
        for data in tqdm(active_dataloader):
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
                # Evaluation
                if eval_dataset is not None and eval_steps is not None and model_logger.num_steps % eval_steps == 0:
                    accelerator.wait_for_everyone()
                    _run_evaluation(
                        accelerator, model, eval_dataset, eval_save_path,
                        model_logger.num_steps, wandb=wandb,
                        num_inference_steps=eval_num_inference_steps,
                        seed=eval_seed, extra_inputs=eval_extra_inputs,
                        max_eval_samples=eval_max_samples,
                    )
                    accelerator.wait_for_everyone()
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


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
