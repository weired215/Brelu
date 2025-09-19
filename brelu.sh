#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
PYTHON="/home/liaolei.pan/Software/anaconda3/envs/pytorch/bin/python3.10" # python environment path
TENSORBOARD='/home/liaolei.pan/Software/anaconda3/envs/pytorch/bin/tensorboard' # tensorboard environment path
data_path='/home/liaolei.pan/data/temp' # dataset path



DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
model=resnet18_quan
dataset=cifar10
test_batch_size=256

attack_sample_size=128 # number of data used for BFA
n_iter=15 # number of iteration to perform BFA
k_top=10 # only check k_top weights with top gradient ranking in each layer

save_path=/home/liaolei.pan/code1/Brelu/save/${DATE}/${dataset}_${model}
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}/tb_log  #tensorboard log path
boundary_path=/home/liaolei.pan/code1/Brelu/boundary

ENABLE_FIND_BOUNDARIES=false
ENABLE_FIXED=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --enable-find-boundaries) ENABLE_FIND_BOUNDARIES=true ;;
        --enable-fixed) ENABLE_FIXED=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

############### main program ########################
{
$PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ${save_path}  \
    --test_batch_size ${test_batch_size} --workers 8 --ngpu 1 \
    --reset_weight --n_iter ${n_iter} --k_top ${k_top} \
    --attack_sample_size ${attack_sample_size} \
    --boundary_path ${boundary_path} \
    $([ "$ENABLE_FIND_BOUNDARIES" = true ] && echo "--find_boundaries") \
    $([ "$ENABLE_FIXED" = true ] && echo "--fixed")
}