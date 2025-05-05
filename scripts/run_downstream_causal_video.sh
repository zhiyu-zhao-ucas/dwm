source $(conda info --base)/etc/profile.d/conda.sh && conda activate robo && python downstream_causal_video.py \
            --training_params.inference_algo=dwm --cuda_id=0 --seed=2 \
            --training_params.mute_wandb=true \
            --training_params.load_inference=\"data3/iwhwang/causal_rl/dwm_new_env_control_freq_10-1/trained_models/inference_final\" \
            --training_params.zero_shot=true;
            echo \"Finished $WINDOW_NAME\"; read