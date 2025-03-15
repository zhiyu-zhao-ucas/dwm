for seed in 1 3; do
    python main_policy.py \
        --training_params.inference_algo=dwm --cuda_id=1 --seed=$seed --training_params.mute_wandb=false
done