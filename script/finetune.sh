
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
rdrop_weight=$9
contrast_weight=${10}
contrast_temp=${11}
sub_dim=${12}
switch_case_probability=${13}
momentum_encoder_beta=${14}


aug_type='attn_span_cutoff'
aug_ce_loss=1.0
aug_js_loss=1.0
aug_cutoff_ratio=0.1


smoothing_factor=0.0
cls_dropout=0.1
max_seq_length=128
evaluation_strategy=epoch

output_dir_complete=${output_dir}/$task_name/bs_$batch_size/lr_$lr_rate/seed_$seed
echo ${output_dir_complete}
mkdir -p ${output_dir_complete}

init=non_adv
case ${init} in
        non_adv)
        parameters=""
            ;;
        adv)
        parameters=" --do_adv \
        --adv_weight 5.0"
        cut)
        parameters=" --aug_type ${aug_type} \
        --aug_cutoff_ratio ${aug_cutoff_ratio} \
        --aug_ce_loss ${aug_ce_loss} \
        --aug_js_loss ${aug_js_loss}"
            ;;

CUDA_VISIBLE_DEVICES=${cuda_visible} python run_glue.py  \
    --model_name_or_path  ${model_name_or_path} \
    --task_name $task_name \
    --do_train \
    --do_eval \
    --do_predict \
    --rdrop_weight ${rdrop_weight} \
    --momentum_encoder_beta ${momentum_encoder_beta} \
    --switch_case_probability ${switch_case_probability} \
    --evaluation_strategy ${evaluation_strategy} \
    --contrast_weight ${contrast_weight} \
    --contrast_temp ${contrast_temp} \
    --sub_dim ${sub_dim} \
    --fp16 \
    --fp16_opt_level O2 \
    --cls_dropout ${cls_dropout} \
    --load_best_model_at_end \
    --save_steps 0 \
    --gradient_accumulation_steps 1 \
    --ignore_data_skip \
    --max_seq_length ${max_seq_length} \
    --per_device_train_batch_size  ${batch_size} \
    --learning_rate $lr_rate \
    --seed $seed \
    --label_smoothing_factor ${smoothing_factor} \
    --num_train_epochs $epoch \
    --overwrite_output_dir \
    --output_dir ${output_dir_complete} \
    --warmup_steps 0.1 \
    --weight_decay 0.01 \
    --dataloader_num_workers 4 \
    --per_device_eval_batch_size 32 $parameters

