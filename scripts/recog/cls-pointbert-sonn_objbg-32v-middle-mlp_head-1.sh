# 2023/09/30

# NOTE 
# model version: ppt-ffn


main_program=main_cls.py
proj_name=recog
exp_name=cls-pointbert-sonn_objbg-32v-middle-mlp_head-1
task=cls

model=ULIP_PointBERT
head_type=2             # ppt-ffn
context_len=32
class_pos=middle

dataset=scanobjectnn
sonn_type=obj_bg
npoints=1024

optim=adamw
lr=0.003
smooth=0.2

gpu=0
epochs=250
batch_size=120
print_freq=40

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
    --sonn_type ${sonn_type} \
    --npoints ${npoints} \
    --optim ${optim} --lr ${lr} \
    --label_smoothing ${smooth} \
    --gpu ${gpu} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --print_freq ${print_freq} \
    --output_dir ${output_dir} \
    --wandb \
    2>&1 | tee ${output_dir}/${proj_name}/${exp_name}/${log_file}