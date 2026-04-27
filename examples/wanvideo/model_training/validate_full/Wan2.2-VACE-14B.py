import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_vace import VaceWanModel


def init_vace_from_dit(dit):
    """Initialize VACE by copying weights from DiT backbone at corresponding layer positions."""
    vace_kwargs = {
        'vace_layers': (0, 5, 10, 15, 20, 25, 30, 35),
        'vace_in_dim': 96,
        'patch_size': (1, 2, 2),
        'has_image_input': False,
        'dim': 5120,
        'num_heads': 40,
        'ffn_dim': 13824,
        'eps': 1e-06,
    }
    vace = VaceWanModel(**vace_kwargs)
    for idx, layer_id in enumerate(vace_kwargs['vace_layers']):
        dit_block_state = dit.blocks[layer_id].state_dict()
        vace.vace_blocks[idx].load_state_dict(dit_block_state, strict=False)
    return vace


vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
)

# Initialize VACE from dit backbones, then load trained checkpoints
pipe.vace = init_vace_from_dit(pipe.dit).to(dtype=torch.bfloat16, device="cuda")
pipe.vace2 = init_vace_from_dit(pipe.dit2).to(dtype=torch.bfloat16, device="cuda")

# Load trained VACE weights (update paths to your checkpoint locations)
state_dict = load_state_dict("logs/Wan2.2-VACE-14B_full/checkpoints_high_noise/epoch-1.safetensors", torch_dtype=torch.bfloat16, device="cpu")
pipe.vace.load_state_dict(state_dict)
state_dict = load_state_dict("logs/Wan2.2-VACE-14B_full/checkpoints_low_noise/epoch-1.safetensors", torch_dtype=torch.bfloat16, device="cpu")
pipe.vace2.load_state_dict(state_dict)

video = VideoData("data/example_video_dataset/video1_softedge.mp4", height=480, width=832)
video = [video[i] for i in range(17)]
reference_image = VideoData("data/example_video_dataset/video1.mp4", height=480, width=832)[0]

video = pipe(
    prompt="from sunset to night, a small town, light, house, river",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=video, vace_reference_image=reference_image, num_frames=17,
    seed=1, tiled=True
)
save_video(video, "video_Wan2.2-VACE-14B.mp4", fps=15, quality=5)
