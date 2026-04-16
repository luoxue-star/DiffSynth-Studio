#!/bin/bash
# ============================================================================
# Wan2.1-VACE-14B Multi-Node Training Script
# ============================================================================
#
# 多机分布式训练启动脚本。需在每台机器上分别运行，通过 --node_rank 区分节点。
#
# ========================= 必填参数 =========================================
#   --node_rank      当前机器编号，主节点为 0，其余依次递增 (0, 1, 2, ...)
#   --master_addr    主节点 (node_rank=0) 的 IP 地址
#
# ========================= 可选参数 (多机相关) ===============================
#   --master_port         主节点通信端口 (默认: 29500)
#   --num_machines        机器总数 (默认: 2)
#   --num_gpu_per_machine 每台机器的 GPU 数量 (默认: 8)
#   --nccl_socket_ifname  指定 NCCL 使用的网卡名 (如 eth0, bond0)
#
# ========================= 可选参数 (训练相关) ===============================
#   --checkpoint_dir / --dataset_base_path / --dataset_metadata_path
#   --data_file_keys / --extra_inputs
#   --wandb_project / --experiment_name / --wandb_mode / --wandb_log_steps
#   --wandb_run_id / --resume_from_checkpoint
#
# ========================= 使用示例 =========================================
#
# 【2 台机器】 IP: 10.48.93.206, 10.48.90.207  每台 8 卡
#
#   机器 0 (10.48.93.206, 主节点):
#     bash Wan2.1-VACE-14B-multinode.sh \
#       --node_rank 0 \
#       --master_addr 10.48.93.206
#
#   机器 1 (10.48.90.207):
#     bash Wan2.1-VACE-14B-multinode.sh \
#       --node_rank 1 \
#       --master_addr 10.48.93.206
#
# 【3 台机器】 IP: 10.48.93.206, 10.48.90.207, 10.48.90.208  每台 8 卡
#
#   机器 0 (10.48.93.206, 主节点):
#     bash Wan2.1-VACE-14B-multinode.sh \
#       --node_rank 0 \
#       --master_addr 10.48.93.206 \
#       --num_machines 3
#
#   机器 1 (10.48.90.207):
#     bash Wan2.1-VACE-14B-multinode.sh \
#       --node_rank 1 \
#       --master_addr 10.48.93.206 \
#       --num_machines 3
#
#   机器 2 (10.48.90.208):
#     bash Wan2.1-VACE-14B-multinode.sh \
#       --node_rank 2 \
#       --master_addr 10.48.93.206 \
#       --num_machines 3
#
# 【N 台机器，每台 M 卡】
#   每台机器都运行:
#     bash Wan2.1-VACE-14B-multinode.sh \
#       --node_rank <0..N-1> \
#       --master_addr <主节点IP> \
#       --num_machines N \
#       --num_gpu_per_machine M
#
# ========================= 启动前检查 =======================================
#
# 1. 确保所有机器的代码路径、数据路径、checkpoint 路径一致（或共享存储）
# 2. 确保所有机器间网络互通，主节点端口可访问:
#      python3 -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('10.48.93.206', 29500)); print('OK'); s.close()"
#    或:
#      curl -s telnet://10.48.93.206:29500 --connect-timeout 3
# 3. 建议先启动 node_rank=0 主节点，再启动其他节点
# 4. 如遇 NCCL 超时，尝试指定网卡: --nccl_socket_ifname eth0
#
# ============================================================================

set -euo pipefail

# ---- Multi-node defaults ---------------------------------------------------
NODE_RANK=""
MASTER_ADDR=""
MASTER_PORT=29500
NUM_MACHINES=2
NUM_GPU_PER_MACHINE=8
NCCL_SOCKET_IFNAME_VAL=""

# ---- Training defaults (same as single-node script) ------------------------
CHECKPOINT_DIR="/mmu_mllm_hdd_2/jinlv/VideoEditing/checkpoints"
DATASET_BASE_PATH="./"
DATASET_METADATA_PATH="/mmu_mllm_hdd_2/jinlv/VideoEditing/data/Custom/VACEv2/metadata_train.csv"
DATA_FILE_KEYS="video,vace_video,vace_video_mask,vace_reference_image"
EXTRA_INPUTS="vace_video,vace_video_mask,vace_reference_image"
WANDB_PROJECT="VACE"
EXPERIMENT_NAME="Wan2.1-VACE-14B-MultiNode"
WANDB_MODE="online"
WANDB_LOG_STEPS=500
WANDB_RUN_ID=""
RESUME_FROM_CHECKPOINT=""
EVAL_METADATA_PATH="/mmu_mllm_hdd_2/jinlv/VideoEditing/data/Custom/VACEv2/metadata_val.csv"
EVAL_STEPS=""
EVAL_NUM_INFERENCE_STEPS=""
EVAL_SEED=""
EVAL_SAVE_PATH="logs/${EXPERIMENT_NAME}/eval"
export WANDB_API_KEY="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"

# ---- Parse arguments -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    # Multi-node arguments
    --node_rank)
      NODE_RANK="$2"; shift 2 ;;
    --master_addr)
      MASTER_ADDR="$2"; shift 2 ;;
    --master_port)
      MASTER_PORT="$2"; shift 2 ;;
    --num_machines)
      NUM_MACHINES="$2"; shift 2 ;;
    --num_gpu_per_machine)
      NUM_GPU_PER_MACHINE="$2"; shift 2 ;;
    --nccl_socket_ifname)
      NCCL_SOCKET_IFNAME_VAL="$2"; shift 2 ;;
    # Original training arguments
    --checkpoint_dir)
      CHECKPOINT_DIR="$2"; shift 2 ;;
    --dataset_base_path)
      DATASET_BASE_PATH="$2"; shift 2 ;;
    --dataset_metadata_path)
      DATASET_METADATA_PATH="$2"; shift 2 ;;
    --data_file_keys)
      DATA_FILE_KEYS="$2"; shift 2 ;;
    --extra_inputs)
      EXTRA_INPUTS="$2"; shift 2 ;;
    --wandb_project)
      WANDB_PROJECT="$2"; shift 2 ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"; shift 2 ;;
    --wandb_mode)
      WANDB_MODE="$2"; shift 2 ;;
    --wandb_log_steps)
      WANDB_LOG_STEPS="$2"; shift 2 ;;
    --wandb_run_id)
      WANDB_RUN_ID="$2"; shift 2 ;;
    --resume_from_checkpoint)
      RESUME_FROM_CHECKPOINT="$2"; shift 2 ;;
    --eval_metadata_path)
      EVAL_METADATA_PATH="$2"; shift 2 ;;
    --eval_steps)
      EVAL_STEPS="$2"; shift 2 ;;
    --eval_num_inference_steps)
      EVAL_NUM_INFERENCE_STEPS="$2"; shift 2 ;;
    --eval_seed)
      EVAL_SEED="$2"; shift 2 ;;
    --eval_save_path)
      EVAL_SAVE_PATH="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      echo "Multi-node args: --node_rank --master_addr --master_port --num_machines --num_gpu_per_machine --nccl_socket_ifname"
      echo "Training args:   --checkpoint_dir --dataset_base_path --dataset_metadata_path --data_file_keys --extra_inputs"
      echo "                 --wandb_project --experiment_name --wandb_mode --wandb_log_steps --wandb_run_id --resume_from_checkpoint"
      echo "Eval args:       --eval_metadata_path --eval_steps --eval_num_inference_steps --eval_seed --eval_save_path"
      exit 1 ;;
  esac
done

# ---- Validate required arguments -------------------------------------------
if [[ -z "${NODE_RANK}" ]]; then
  echo "ERROR: --node_rank is required (0 for master, 1 for worker, ...)"
  exit 1
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "ERROR: --master_addr is required (IP of node_rank=0 machine)"
  exit 1
fi

NUM_PROCESSES=$(( NUM_MACHINES * NUM_GPU_PER_MACHINE ))

# ---- NCCL environment variables --------------------------------------------
export NCCL_DEBUG=INFO
if [[ -n "${NCCL_SOCKET_IFNAME_VAL}" ]]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_VAL}"
fi

# ---- Setup log file --------------------------------------------------------
LOG_DIR="logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_node${NODE_RANK}_${TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

# ---- Print configuration ---------------------------------------------------
echo "============================================"
echo " Multi-Node Training Configuration"
echo "============================================"
echo " Start time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo " Master addr:       ${MASTER_ADDR}:${MASTER_PORT}"
echo " Num machines:      ${NUM_MACHINES}"
echo " GPUs per machine:  ${NUM_GPU_PER_MACHINE}"
echo " Total processes:   ${NUM_PROCESSES}"
echo " This node rank:    ${NODE_RANK}"
echo " Experiment:        ${EXPERIMENT_NAME}"
echo " Log file:          ${LOG_FILE}"
echo "============================================"

# ---- Generate accelerate config dynamically --------------------------------
# DeepSpeed launcher ignores accelerate CLI args (--main_process_ip, etc.)
# and requires correct values directly in the YAML config file.
TEMP_CONFIG=$(mktemp /tmp/accelerate_config_multinode_XXXXXX.yaml)
trap "rm -f ${TEMP_CONFIG}" EXIT

cat > "${TEMP_CONFIG}" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: ${NODE_RANK}
main_process_ip: ${MASTER_ADDR}
main_process_port: ${MASTER_PORT}
main_training_function: main
mixed_precision: bf16
num_machines: ${NUM_MACHINES}
num_processes: ${NUM_PROCESSES}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "Generated accelerate config: ${TEMP_CONFIG}"
cat "${TEMP_CONFIG}"
echo "--------------------------------------------"

# ---- Launch training -------------------------------------------------------
accelerate launch \
  --config_file "${TEMP_CONFIG}" \
  -m examples.wanvideo.model_training.train \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "${DATA_FILE_KEYS}" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-14B/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-14B/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-14B/Wan2.1_VAE.pth" \
  --learning_rate 5e-5 \
  --num_epochs 5 \
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
  ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"} \
  ${EVAL_METADATA_PATH:+--eval_metadata_path "$EVAL_METADATA_PATH"} \
  ${EVAL_STEPS:+--eval_steps "$EVAL_STEPS"} \
  ${EVAL_NUM_INFERENCE_STEPS:+--eval_num_inference_steps "$EVAL_NUM_INFERENCE_STEPS"} \
  ${EVAL_SEED:+--eval_seed "$EVAL_SEED"} \
  ${EVAL_SAVE_PATH:+--eval_save_path "$EVAL_SAVE_PATH"}
