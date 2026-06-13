ngpus=1
accumulate_step=1
batch_size=$((ngpus * accumulate_step))

torchrun --nnodes=1 --nproc_per_node=${ngpus} --rdzv-endpoint=localhost:29520 \
    train.py \
    --output_dir=outputs/scannetpp \
    --gin_file=configs/dataset/pp.gin \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/defaultpp.gin \
    --gin_param="build_trainloader.batch_size="${batch_size} \
    --gin_param="build_trainloader.accumulate_step="${accumulate_step}

