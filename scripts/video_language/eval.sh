
CACHE_DIR="path/to/pretrained/weight"
RESUME="video_language.pt"
TRAIN_DATA="path/to/data"
# this script is for 768 total batch_size (n(16) GPUs * batch_size(24) * accum_freq(2))
cd /path/to/LanguageBind
TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=$HOST_NUM --node_rank=$INDEX --nproc_per_node $HOST_GPU_NUM --master_addr $CHIEF_IP \
    -m main  \
    --train-data ${TRAIN_DATA} \
    --train-num-samples 3020000 \
    --clip-type "vl" \
    --lock-text --lock-image --text-type "polish_mplug" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" --cache-dir ${CACHE_DIR} \
    --convert_to_lora --lora_r 64 \
    --lr 1e-4 --coef-lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 8 --force-patch-dropout 0.5 \
    --epochs 1 --batch-size 24 --accum-freq 2 --warmup 200 \
    --precision "amp" --workers 10 --video-decode-backend "imgs" \
    --save-frequency 1 --log-every-n-steps 20 --report-to "tensorboard" --resume ${RESUME} \
    --do_eval \
    --val_vl_ret_data "msrvtt" "msvd" "activity" "didemo"