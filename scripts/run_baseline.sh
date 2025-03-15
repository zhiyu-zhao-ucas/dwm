for seed in 1 2 3; do
    python main_policy.py \
        --training_params.inference_algo=ours --cuda_id=3 --seed=$seed
done