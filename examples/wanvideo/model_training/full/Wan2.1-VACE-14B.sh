CHECKPOINT_DIR="/mmu_mllm_hdd_2/jinlv/VideoEditing/checkpoints"
DATASET_BASE_PATH="/mmu_mllm_hdd_2/jinlv/VideoEditing/data/DiffSynthStudioExample"
DATASET_METADATA_PATH="/mmu_mllm_hdd_2/jinlv/VideoEditing/data/DiffSynthStudioExample/metadata_vace.csv"
DATA_FILE_KEYS="video,vace_video,vace_reference_image"
EXTRA_INPUTS="vace_video,vace_reference_image"
WANDB_PROJECT="VACE"
EXPERIMENT_NAME="Wan2.1-VACE-14B_full"
WANDB_MODE="online"
WANDB_LOG_STEPS=100
WANDB_RUN_ID=""

RESUME_FROM_CHECKPOINT=""

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
    --wandb_run_id)
      WANDB_RUN_ID="$2"
      shift 2
      ;;
    --resume_from_checkpoint)
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Supported arguments: --checkpoint_dir --dataset_base_path --dataset_metadata_path --data_file_keys --extra_inputs --wandb_project --experiment_name --wandb_mode --wandb_log_steps --wandb_run_id --resume_from_checkpoint"
      exit 1
      ;;
  esac
done

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml -m examples.wanvideo.model_training.train \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "${DATA_FILE_KEYS}" \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-14B/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-14B/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-14B/Wan2.1_VAE.pth" \
  --learning_rate 5e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "logs/${EXPERIMENT_NAME}/checkpoints" \
  --trainable_models "vace" \
  --extra_inputs "${EXTRA_INPUTS}" \
  --use_gradient_checkpointing \
  --wandb_project "${WANDB_PROJECT}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_log_steps "${WANDB_LOG_STEPS}" \
  ${WANDB_RUN_ID:+--wandb_run_id "$WANDB_RUN_ID"} \
  ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"}

# The learning rate is kept consistent with the settings in the original paper
# Example:
# bash examples/wanvideo/model_training/full/Wan2.1-VACE-14B.sh --data_file_keys "video,vace_video,vace_video_mask,vace_reference_image" --extra_inputs "vace_video,vace_video_mask,vace_reference_image"