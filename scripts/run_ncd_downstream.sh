for seed in 1; do
    python downstream.py \
        --training_params.inference_algo=ncd --cuda_id=2 --seed=$seed --training_params.mute_wandb=true --training_params.load_inference="data1/iwhwang/causal_rl/Chemical/ncd-${seed}/trained_models/inference_15k" --training_params.zero_shot=true
done