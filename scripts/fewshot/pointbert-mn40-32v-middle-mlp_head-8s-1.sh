# 2023/09/30

# NOTE 
#   1. ulip2
#   2. class pos: front -> middle
#   3. head_type: ppt-ffn


main_program=main_fewshot.py
proj_name=fewshot
exp_name=pointbert-mn40-32v-middle-mlp_head-8s-1
task=fewshot

model=ULIP_PointBERT
head_type=2         # ppt-ffn
context_len=32
class_pos=middle

dataset=modelnet40_fs   
npoints=1024
nshots=8

optim=adamw
lr=0.003
smooth=0.2

gpu=2
epochs=250
batch_size=40
print_freq=60

output_dir=outputs
log_file=run.log

if [ ! -d ${output_dir}/${proj_name}/${exp_name} ]
then
    mkdir -p ${output_dir}/${proj_name}/${exp_name}
fi

if [ ! -f ${output_dir}/${proj_name}/${exp_name}/${log_file} ]
then
    touch ${output_dir}/${proj_name}/${exp_name}/${log_file}
fi

pueue add -g ${proj_name} \
    python ${main_program} \
    --proj_name ${proj_name} \
    --exp_name ${exp_name} \
    --main_program ${main_program} \
    --task ${task} \
    --model ${model} --ulip2 \
    --head_type ${head_type} \
    --num_learnable_prompt_tokens ${context_len} \
    --class_name_position ${class_pos} \
    --dataset_name ${dataset} \
    --npoints ${npoints} \
    --nshots ${nshots} \
    --optim ${optim} \
    --lr ${lr} \
    --label_smoothing ${smooth} \
    --gpu ${gpu} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --print_freq ${print_freq} \
    --output_dir ${output_dir} \
    --wandb \
    2>&1 | tee ${output_dir}/${proj_name}/${exp_name}/${log_file}