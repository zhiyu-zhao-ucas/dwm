set -ex

# Create a single tmux session
tmux new-session -d -s "training_session"

# Counter for window creation
window_index=0

for algo in ncd mlp gnn; do
    for seed in 1 2 3; do
        # Calculate GPU ID
        gpu=$(( (seed - 1 + (algo == "ncd" ? 0 : algo == "mlp" ? 3 : 6)) % 4 ))
        
        # Create a window name based on algorithm and seed
        window_name="${algo}_${seed}"
        
        # For the first iteration, rename the first window
        if [ $window_index -eq 0 ]; then
            tmux rename-window -t "training_session:0" "$window_name"
            tmux send-keys -t "training_session:0" "python main_policy.py --training_params.inference_algo=$algo --cuda_id=$gpu --seed=$seed; echo 'Finished $algo seed $seed on GPU $gpu'; read -p 'Press Enter to close this window'" C-m
        else
            # For subsequent iterations, create new windows
            tmux new-window -t "training_session:$window_index" -n "$window_name"
            tmux send-keys -t "training_session:$window_index" "python main_policy.py --training_params.inference_algo=$algo --cuda_id=$gpu --seed=$seed; echo 'Finished $algo seed $seed on GPU $gpu'; read -p 'Press Enter to close this window'" C-m
        fi
        
        echo "Started $algo with seed $seed on GPU $gpu in window $window_name"
        
        # Increment window index
        ((window_index++))
        
        # Small delay between commands
        sleep 1
    done
done

echo "All jobs submitted. Use 'tmux attach -t training_session' to view the session"