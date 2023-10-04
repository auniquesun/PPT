# 2023/09/30

class=$1
ballradius=13
proj_name=ulip2_partseg
exp_name=partseg-pointbert-shapepart-32v-middle-2
task=partseg

gpu=0
model=ULIP_PointBERT_partseg
context_len=32
class_pos=middle

dataset=shapenetpart
npoints=2048    # 2048 points for each point cloud

batch_size=32
output_dir=outputs

python show_balls.py \
    --class_choice ${class} \
    --ballradius ${ballradius} \
    --proj_name ${proj_name} \
    --exp_name ${exp_name} \
    --task ${task} \
    --gpu ${gpu} \
    --model ${model} --ulip2 \
    --num_learnable_prompt_tokens ${context_len} \
    --class_name_position ${class_pos} \
    --dataset_name ${dataset} \
    --npoints ${npoints} \
    --batch_size ${batch_size} \
    --output_dir ${output_dir} \
