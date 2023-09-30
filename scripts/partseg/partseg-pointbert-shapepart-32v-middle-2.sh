# 2023/09/30

# NOTE
#   1. ulip2-pointbert model
#   2. class pos: middle
#   3. original lr scheduler
#   4. 6 2080Ti GPUs


nproc=6
main_program=main_partseg.py
proj_name=partseg
exp_name=partseg-pointbert-shapepart-32v-middle-2
task=partseg

model=ULIP_PointBERT_partseg
context_len=32
class_pos=middle

dataset=shapenetpart
npoints=2048    # 2048 points for each point cloud

optim=adamw
lr=0.001
smooth=0.2

epochs=250
batch_size=15
print_freq=100

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
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch \
    --nproc_per_node=${nproc} ${main_program} \
    --proj_name ${proj_name} \
    --main_program ${main_program} \
    --exp_name ${exp_name} \
    --task ${task} \
    --model ${model} --ulip2 \
    --num_learnable_prompt_tokens ${context_len} \
    --class_name_position ${class_pos} \
    --dataset_name ${dataset} \
    --npoints ${npoints} \
    --optim ${optim} --lr ${lr} \
    --label_smoothing ${smooth} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --print_freq ${print_freq} \
    --output_dir ${output_dir} \
    --wandb \
    2>&1 | tee ${output_dir}/${proj_name}/${exp_name}/${log_file}