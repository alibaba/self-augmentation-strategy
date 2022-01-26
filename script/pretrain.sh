
help="# Default parameters"
fp16=O2;                              
ds=25600000;    ds_str=25M;             help="# Set a super large value to use the whole training data (about 30M instances)"
steps=250000;   steps_str=250k;         help="# 5 epochs"
dataset='full'; dataset_eval='eval';
model_name_or_path="None"

cold_start_augumentation_method=unigram;help="# unigram"
dis_weight=50-50;dis_weight_scheduler=0;  help="# Model hyper-parameter"
augmentation_temperature=1;             help="# 0 (Always choose top 1 & dn't allow hit); Positive temperature: regular (allow hit); Negative values: Don't allow hit"
lr=0.001;batch_size=512;                help="# Training hyper-parameters;"
adam_epsilon=1e-6;                      help="# Training hyper-parameters;"
adam_beta1=0.9;adam_beta2=0.999;        help="# Training hyper-parameters;"
gradient_accumulation_steps=1;          help="# Gradient_accumulation_step"
warmup_steps=10000; weight_decay=0.01;
dataloader_num_workers=4;

cold_start_epochs=1.00;                 help="# Cold start period. Generate data augmentation from 'cold_start_epochs-1'. Useful to skip initial period when the quality of MLM output is very lo"
augmentation_copies=1;                  help="# Freeze generation capability by creating multiple version of data augmentation per instance, each used for a later epoch"

mlm_probability=0.15;whole_word_masking=0;dynamic_masking=0;
position_embedding_type=absolute; pe_abbreviation=abs;

DebugMode=1;debugExtraMetrics=1; logging_steps=500; debugMemStatsInterval=2000; debugGradOverflowInterval=3000; debugMultiTasksConflictInterval=700000;
save_steps=50000; save_total_limit=1;

model_type=sas; bert_setting=SAS_small; max_position_embeddings=128;

seed=99;     


option=SAS_dis

#SAS^c in the paper
if [ ${option} = 'SAS_const' ]; then cuda="0";
    dynamic_masking=1; whole_word_masking=1; lr=0.00100; seed=99; dis_weight=50-50; dis_weight_scheduler=0;
    save_steps=25000; save_total_limit=10;
fi


if [ ${option} = 'SAS_dis' ]; then cuda="0";
    dynamic_masking=1; whole_word_masking=1; lr=0.00100; seed=99; dis_weight=50-200;dis_weight_scheduler=2;
    save_steps=25000; save_total_limit=10;
fi

output_name=${option}_cuda${cuda}
mkdir -p ./output/${output_name}

data_path=../data_dir/

python train.py \
    -bert_setting ${bert_setting} -max_position_embeddings ${max_position_embeddings} \
    -option ${option} -cuda ${cuda} -seed ${seed}  -model_type ${model_type} \
    -fp16 ${fp16} -max_steps ${steps} -data_size ${ds}  -dataset ${dataset} -dataset_eval ${dataset_eval}\
    -lr ${lr} -warmup_steps ${warmup_steps} -weight_decay ${weight_decay} -batch_size ${batch_size} -gradient_accumulation_steps ${gradient_accumulation_steps} \
    -adam_epsilon ${adam_epsilon} -adam_beta1 ${adam_beta1} -adam_beta2 ${adam_beta2}  \
    -cold_start_augumentation_method ${cold_start_augumentation_method} -augmentation_temperature ${augmentation_temperature} \
    -dis_weight ${dis_weight} -dis_weight_scheduler ${dis_weight_scheduler} \
    -cold_start_epochs ${cold_start_epochs}  \
    -mlm_probability ${mlm_probability} \
    -debugExtraMetrics ${debugExtraMetrics} -logging_steps ${logging_steps} -debugMemStatsInterval ${debugMemStatsInterval} \
    -debugGradOverflowInterval ${debugGradOverflowInterval}  -debugMultiTasksConflictInterval ${debugMultiTasksConflictInterval} \
    -save_steps ${save_steps} -save_total_limit ${save_total_limit} \
    -whole_word_masking ${whole_word_masking} -dynamic_masking ${dynamic_masking} \
    -dataloader_num_workers ${dataloader_num_workers} \
    -position_embedding_type ${position_embedding_type} \
    -data_path ${data_path} \
    -output_dir ${output_name}
    
    