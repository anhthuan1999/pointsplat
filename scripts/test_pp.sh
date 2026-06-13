torchrun --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:29520 \
    train.py \
    --output_dir=outputs/scannetpp \
    --gin_file=configs/dataset/pp.gin \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/defaultpp.gin \
    --gin_param="build_trainloader.batch_size=1" \
    --only_eval --eval_subdir test --compare_with_input \
    --gin_param="FeaturePredictor.resume_ckpt='outputs/scannetpp/checkpoints/model_00009999.pth'"
