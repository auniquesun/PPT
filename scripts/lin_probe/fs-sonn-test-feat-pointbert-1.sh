# 2023/09/30

proj_name=lin_probe
exp_name=fs-sonn-test-feat-pointbert-1
main_program=lp_feat_extractor.py
task=fs_lp      # fewshot-linear_probe

model=ULIP_PointBERT
dataset=scanobjectnn
split=test
npoints=1024

seed=1
gpu=0
batch_size=120

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
        --dataset_name ${dataset} \
        --dataset_type ${split} \
        --npoints ${npoints} \
        --seed ${seed} \
        --gpu ${gpu} \
        --batch_size ${batch_size} \
        --output_dir ${output_dir} \
        2>&1 | tee ${output_dir}/${proj_name}/${exp_name}/${log_file}
