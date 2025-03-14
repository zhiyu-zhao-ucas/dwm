for seed in 2; do
    python main_policy.py \
        --training_params.inference_algo=ours --cuda_id=2 --seed=$seed
done