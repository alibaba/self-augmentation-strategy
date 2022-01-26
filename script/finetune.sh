
set -e
export TOKENIZERS_PARALLELISM=true

task_name=$1 #
cuda_visible=$2
model_name_or_path=$3
output_dir=$4
lr_rate=$5
batch_size=$6
epoch=$7
seed=$8

output_dir_complete=${output_dir}/$task_name/bs_$batch_size/lr_$lr_rate/seed_$seed
echo ${output_dir_complete}
mkdir -p ${output_dir_complete}

CUDA_VISIBLE_DEVICES=${cuda_visible} python run_glue.py  \
    --model_name_or_path  ${model_name_or_path} \
    --task_name $task_name \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --save_steps 0 \
    --gradient_accumulation_steps 1 \
    --ignore_data_skip \
    --max_seq_length 128 \
    --per_device_train_batch_size  ${batch_size} \
    --learning_rate $lr_rate \
    --seed $seed \
    --num_train_epochs $epoch \
    --overwrite_output_dir \
    --output_dir ${output_dir_complete} \
    --warmup_steps 0.1 \
    --weight_decay 0.01 \
    --dataloader_num_workers 4 \
    --per_device_eval_batch_size 8

