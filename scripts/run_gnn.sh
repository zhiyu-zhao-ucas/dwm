for seed in 1 2 3; do
    python main_policy.py \
        --training_params.inference_algo=gnn --cuda_id=0 --seed=$seed
done
for seed in 1 2 3; do
    python main_policy.py \
        --training_params.inference_algo=mlp --cuda_id=0 --seed=$seed
done