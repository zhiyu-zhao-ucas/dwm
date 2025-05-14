#!/bin/bash

set -ex
# Get number of available GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ $GPU_COUNT -eq 0 ]; then
    echo "No GPUs detected. Exiting."
    exit 1
fi
echo "Detected $GPU_COUNT GPUs"

# Create a single session
SESSION_NAME="dwm_wo_causal_loss_chain"
tmux new-session -d -s "$SESSION_NAME" -n "init" "echo 'Initializing session'; read"

window_index=0
gpu_index=0

for algo in dwm_wo_causal_loss; do
    for seed in 1 2 3 4 5 6 7 8; do
        # Create a unique window name
        WINDOW_NAME="${algo}_${seed}"
        window_index=$((window_index + 1))
        
        # Assign current GPU and rotate to next
        current_gpu=$gpu_index
        gpu_index=$(( (gpu_index + 1) % GPU_COUNT ))
        
        # Create a new window in the existing session
        tmux new-window -t "$SESSION_NAME:$window_index" -n "$WINDOW_NAME" \
            "source $(conda info --base)/etc/profile.d/conda.sh && conda activate fcdl && python main_policy.py \
            --training_params.inference_algo=$algo --cuda_id=$current_gpu --seed=$seed \
            --training_params.mute_wandb=false \
            --inference_params.causal_coef=0.0 \
            --training_params.zero_shot=true;
            echo \"Finished $WINDOW_NAME\"; read"
        
        echo "Started window: $WINDOW_NAME in session $SESSION_NAME on GPU $current_gpu"
    done
done

echo "All jobs started in tmux session: $SESSION_NAME"
echo "Use 'tmux attach -t $SESSION_NAME' to view the session."
echo "Once attached, use 'Ctrl+B n' to go to the next window or 'Ctrl+B NUMBER' to go to a specific window."
# Attach to the tmux session if not already in a tmux session
if [ -z "$TMUX" ]; then
    echo "Attaching to tmux session: $SESSION_NAME"
    tmux attach -t "$SESSION_NAME"
else
    echo "Already in a tmux session. Use 'tmux switch-client -t $SESSION_NAME' to switch."
fi