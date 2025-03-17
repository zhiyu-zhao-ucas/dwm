for seed in 1; do
    python downstream.py \
        --training_params.inference_algo=dwm --cuda_id=2 --seed=$seed --training_params.mute_wandb=false --training_params.load_inference="data1/iwhwang/causal_rl/Chemical/dwm-${seed}/trained_models/inference_15k"
done