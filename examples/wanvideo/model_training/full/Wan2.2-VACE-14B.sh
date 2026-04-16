CHECKPOINT_DIR="/mmu_mllm_hdd_2/jinlv/VideoEditing/checkpoints"
DATASET_BASE_PATH="./"
DATASET_METADATA_PATH="/mmu_mllm_hdd_2/jinlv/VideoEditing/data/Custom/VACEv2/metadata_train.csv"
DATA_FILE_KEYS="video,vace_video,vace_video_mask,vace_reference_image"
EXTRA_INPUTS="vace_video,vace_video_mask,vace_reference_image"
WANDB_PROJECT="VACE"
EXPERIMENT_NAME="Wan2.2-VACE-14B_full"
WANDB_MODE="online"
WANDB_LOG_STEPS=500
WANDB_RUN_ID=""
RESUME_FROM_CHECKPOINT_HIGH=""
RESUME_FROM_CHECKPOINT_LOW=""
TIMESTEP_BOUNDARY=0.417
EVAL_METADATA_PATH="/mmu_mllm_hdd_2/jinlv/VideoEditing/data/Custom/VACEv2/metadata_val.csv"
EVAL_STEPS=1000
EVAL_NUM_INFERENCE_STEPS=""
EVAL_SEED=""
EVAL_SAVE_PATH="logs/${EXPERIMENT_NAME}/eval"

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
    --resume_from_checkpoint_high)
      RESUME_FROM_CHECKPOINT_HIGH="$2"
      shift 2
      ;;
    --resume_from_checkpoint_low)
      RESUME_FROM_CHECKPOINT_LOW="$2"
      shift 2
      ;;
    --timestep_boundary)
      TIMESTEP_BOUNDARY="$2"
      shift 2
      ;;
    --eval_metadata_path)
      EVAL_METADATA_PATH="$2"
      shift 2
      ;;
    --eval_steps)
      EVAL_STEPS="$2"
      shift 2
      ;;
    --eval_num_inference_steps)
      EVAL_NUM_INFERENCE_STEPS="$2"
      shift 2
      ;;
    --eval_seed)
      EVAL_SEED="$2"
      shift 2
      ;;
    --eval_save_path)
      EVAL_SAVE_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Supported arguments: --checkpoint_dir --dataset_base_path --dataset_metadata_path --data_file_keys --extra_inputs --wandb_project --experiment_name --wandb_mode --wandb_log_steps --wandb_run_id --resume_from_checkpoint_high --resume_from_checkpoint_low --timestep_boundary --eval_metadata_path --eval_steps --eval_num_inference_steps --eval_seed --eval_save_path"
      exit 1
      ;;
  esac
done

# === High Noise Model: VACE Training ===
# Timestep range: [boundary*1000, 1000]
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml -m examples.wanvideo.model_training.train \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "${DATA_FILE_KEYS}" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --init_vace_from_dit \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "logs/${EXPERIMENT_NAME}/checkpoints_high_noise" \
  --trainable_models "vace" \
  --extra_inputs "${EXTRA_INPUTS}" \
  --use_gradient_checkpointing \
  --max_timestep_boundary "${TIMESTEP_BOUNDARY}" \
  --min_timestep_boundary 0 \
  --wandb_project "${WANDB_PROJECT}" \
  --experiment_name "${EXPERIMENT_NAME}_high_noise" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_log_steps "${WANDB_LOG_STEPS}" \
  ${WANDB_RUN_ID:+--wandb_run_id "$WANDB_RUN_ID"} \
  ${RESUME_FROM_CHECKPOINT_HIGH:+--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT_HIGH"} \
  ${EVAL_METADATA_PATH:+--eval_metadata_path "$EVAL_METADATA_PATH"} \
  ${EVAL_STEPS:+--eval_steps "$EVAL_STEPS"} \
  ${EVAL_NUM_INFERENCE_STEPS:+--eval_num_inference_steps "$EVAL_NUM_INFERENCE_STEPS"} \
  ${EVAL_SEED:+--eval_seed "$EVAL_SEED"} \
  ${EVAL_SAVE_PATH:+--eval_save_path "$EVAL_SAVE_PATH"}

# === Low Noise Model: VACE Training ===
# Timestep range: [0, boundary*1000)
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml -m examples.wanvideo.model_training.train \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "${DATA_FILE_KEYS}" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --init_vace_from_dit \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "logs/${EXPERIMENT_NAME}/checkpoints_low_noise" \
  --trainable_models "vace" \
  --extra_inputs "${EXTRA_INPUTS}" \
  --use_gradient_checkpointing \
  --max_timestep_boundary 1 \
  --min_timestep_boundary "${TIMESTEP_BOUNDARY}" \
  --wandb_project "${WANDB_PROJECT}" \
  --experiment_name "${EXPERIMENT_NAME}_low_noise" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_log_steps "${WANDB_LOG_STEPS}" \
  ${WANDB_RUN_ID:+--wandb_run_id "$WANDB_RUN_ID"} \
  ${RESUME_FROM_CHECKPOINT_LOW:+--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT_LOW"} \
  ${EVAL_METADATA_PATH:+--eval_metadata_path "$EVAL_METADATA_PATH"} \
  ${EVAL_STEPS:+--eval_steps "$EVAL_STEPS"} \
  ${EVAL_NUM_INFERENCE_STEPS:+--eval_num_inference_steps "$EVAL_NUM_INFERENCE_STEPS"} \
  ${EVAL_SEED:+--eval_seed "$EVAL_SEED"} \
  ${EVAL_SAVE_PATH:+--eval_save_path "$EVAL_SAVE_PATH"}

# Wan2.2 uses dual-model architecture (high_noise + low_noise) with timestep boundary.
# Default boundary 0.417 corresponds to timesteps [875, 1000] for high noise and [0, 875) for low noise.
# VACE is initialized from the DiT backbone weights at corresponding layer positions.
# Example:
# bash examples/wanvideo/model_training/full/Wan2.2-VACE-14B.sh
# With custom boundary (e.g., 0.358 like VACE-Fun):
# bash examples/wanvideo/model_training/full/Wan2.2-VACE-14B.sh --timestep_boundary 0.358
