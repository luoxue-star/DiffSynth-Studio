CHECKPOINT_DIR="/mmu_mllm_hdd_2/jinlv/VideoEditing/checkpoints"
DATASET_BASE_PATH="/mmu_mllm_hdd_2/jinlv/VideoEditing/data/DiffSynthStudioExample"
DATASET_METADATA_PATH="/mmu_mllm_hdd_2/jinlv/VideoEditing/data/DiffSynthStudioExample/metadata_vace.csv"
DATA_FILE_KEYS="video,vace_video,vace_reference_image"
EXTRA_INPUTS="vace_video,vace_reference_image"
WANDB_PROJECT="VACE"
EXPERIMENT_NAME="Wan2.1-VACE-1.3B_lora"
WANDB_MODE="online"
WANDB_LOG_STEPS=100

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --dataset_base_path)
      DATASET_BASE_PATH="$2"
      shift 2
      ;;
    --dataset_metadata_path)
      DATASET_METADATA_PATH="$2"
      shift 2
      ;;
    --data_file_keys)
      DATA_FILE_KEYS="$2"
      shift 2
      ;;
    --extra_inputs)
      EXTRA_INPUTS="$2"
      shift 2
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --wandb_mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --wandb_log_steps)
      WANDB_LOG_STEPS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Supported arguments: --checkpoint_dir --dataset_base_path --dataset_metadata_path --data_file_keys --extra_inputs --wandb_project --experiment_name --wandb_mode --wandb_log_steps"
      exit 1
      ;;
  esac
done

accelerate launch -m examples.wanvideo.model_training.train \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "${DATA_FILE_KEYS}" \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors,Wan-AI/Wan2.1-VACE-1.3B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "logs/${EXPERIMENT_NAME}/checkpoints" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "${EXTRA_INPUTS}" \
  --use_gradient_checkpointing_offload \
  --wandb_project "${WANDB_PROJECT}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_log_steps "${WANDB_LOG_STEPS}"

# Example:
# bash examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh --data_file_keys "video,vace_video,vace_video_mask,vace_reference_image" --extra_inputs "vace_video,vace_video_mask,vace_reference_image"