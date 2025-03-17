for seed in 2 3; do
    python downstream.py \
        --training_params.inference_algo=ours --cuda_id=1 --seed=$seed --training_params.mute_wandb=false --training_params.load_inference="data1/iwhwang/causal_rl/Chemical/ours-${seed}/trained_models/inference_15k"
done