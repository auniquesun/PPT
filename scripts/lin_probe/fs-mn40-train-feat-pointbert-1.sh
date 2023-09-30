# 2023/09/30

proj_name=lin_probe
exp_name=fs-mn40-train-feat-pointbert-1
main_program=lp_feat_extractor.py
task=fewshot

model=ULIP_PointBERT
dataset=modelnet40
split=train
npoints=1024

seed=1
gpu=0
batch_size=120
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
    python lp_feat_extractor.py \
        --proj_name ${proj_name} \
        --exp_name ${exp_name} \
        --task ${task} \
        --model ${model} --ulip2 \
        --main_program ${main_program} \
        --dataset_name ${dataset} \
        --dataset_type ${split} \
        --npoints ${npoints} \
        --dataset_type ${split} \
        --seed ${seed} \
        --gpu ${gpu} \
        --batch_size ${batch_size} \
        --print_freq ${print_freq} \
        --output_dir ${output_dir} \
        2>&1 | tee ${output_dir}/${proj_name}/${exp_name}/${log_file}
