for seed in 1; do
    python main_policy.py \
        --training_params.inference_algo=dwm --cuda_id=0 --seed=$seed
done