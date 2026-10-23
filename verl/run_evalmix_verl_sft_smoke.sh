#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/verl${PYTHONPATH:+:${PYTHONPATH}}"
export HYPER_PARALLEL_PLATFORM=torch
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_NPU_SOCKET_PORT_RANGE=23000-23200
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export VERL_USE_UV=0
export VERL_SFT_LOGGING_LEVEL=INFO
export USE_OPTIMIZED_MODEL=0
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1


cd /verl

TRAIN_FILES=$(cat /tmp/Qwen3.6_eval_split/train_files.hydra.txt)
VAL_FILES=$(cat /tmp/Qwen3.6_eval_split/val_files.hydra.txt)
MODEL_PATH=/data1/models/Qwen3.6-35B-A3B
CKPT_HOME=/tmp/geo3k_evalmix_smoke_ckpt
rm -rf "${CKPT_HOME}"
mkdir -p "${CKPT_HOME}"

torchrun --standalone --nnodes=1 --nproc-per-node=16 \
  -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_max_samples=64 \
    data.val_max_samples=16 \
    data.messages_key=messages \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=1 \
    data.max_token_len_per_gpu=4096 \
    data.use_dynamic_bsz=False \
    data.max_length=4096 \
    data.pad_mode=right \
    data.ignore_input_ids_mismatch=True \
    data.truncation=right \
    +data.image_patch_size=16 \
    data.num_workers=0 \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=False \
    model.use_liger=False \
    +model.override_config.attn_implementation=sdpa \
    model.use_fused_kernels=False \
    model.enable_gradient_checkpointing=True \
    engine=fsdp \
    engine.strategy=fsdp2 \
    engine.model_dtype=bfloat16 \
    engine.use_torch_compile=False \
    engine.ulysses_sequence_parallel_size=1 \
    engine.fsdp_size=16 \
    optim.lr=1e-5 \
    trainer.device=npu \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.project_name=geo3k-verl-sft \
    trainer.experiment_name=qwen36-35b-a3b-evalmix-100step \
    trainer.total_training_steps=100 \
    trainer.total_epochs=100 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.max_ckpt_to_keep=1 \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=16 \
    trainer.logger="['console']" \
    checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    "$@"
