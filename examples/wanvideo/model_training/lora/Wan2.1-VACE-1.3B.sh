CHECKPOINT_DIR="/mmu_mllm_hdd_2/jinlv/VideoEditing/checkpoints"

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /mmu_mllm_hdd_2/jinlv/VideoEditing/data/DiffSynthStudioExample \
  --dataset_metadata_path /mmu_mllm_hdd_2/jinlv/VideoEditing/data/DiffSynthStudioExample/metadata_vace.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors,Wan-AI/Wan2.1-VACE-1.3B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:${CHECKPOINT_DIR%/}/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "${CHECKPOINT_DIR%/}/Wan2.1-VACE-1.3B_lora" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload
