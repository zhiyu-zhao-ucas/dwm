import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from fcdl.utils.utils import set_seed_everywhere

# 设置显示格式
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

# 确保当前目录在path中
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

# 导入必要的模块
from fcdl.env.chemical_env import Chemical
from fcdl.model.encoder import make_encoder
from fcdl.model.inference_dwm import InferenceDWM
from fcdl.model.inference_ours_masking import InferenceOursMask
from fcdl.utils.utils import TrainingParams, get_env, update_obs_act_spec
from fcdl.utils.replay_buffer import ReplayBuffer

def process_model(name, device):
    # 设置模型路径
    reference_model_name = name.replace("dwm", "ours")
    model_path = f"data_new/iwhwang/causal_rl/Chemical/{name}/trained_models/inference_final"
    reference_model_path = f"data_new/iwhwang/causal_rl/Chemical/{reference_model_name}/trained_models/inference_final"
    params_path = f"data_new/iwhwang/causal_rl/Chemical/{name}/params"
    env_params_path = f"data_new/iwhwang/causal_rl/Chemical/{name}/params"

    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [model_path, params_path, env_params_path]):
        logger.warning(f"模型 {name} 的文件不存在，跳过")
        return None

    # 加载参数
    params_dict = torch.load(params_path)
    params = TrainingParams(training_params_fname="policy_params.json", train=False)

    # 将加载的参数字典复制到params对象
    for key, value in params_dict.items():
        setattr(params, key, value)

    # 加载环境参数
    env_params = torch.load(env_params_path)
    logger.info(f"已加载模型 {name} 的参数和环境设置")

    # 设置设备
    params.device = device

    # 创建环境
    params.env_params.num_env = 1  # 确保只使用一个环境
    env = get_env(params)

    # 更新观测和动作空间
    update_obs_act_spec(env, params)

    # 创建编码器和推理模型
    encoder = make_encoder(params)
    inference = InferenceDWM(encoder, params)
    reference_inference = InferenceOursMask(encoder, params)

    # 加载保存的模型
    inference.load(model_path, device)
    inference.eval()
    reference_inference.load(reference_model_path, device)
    reference_inference.eval()
    logger.info(f"推理模型 {name} 加载完成")

    # 创建缓冲区来收集样本
    buffer = ReplayBuffer(params)
    num_samples = 100
    current_samples = 0

    # 收集样本
    obs = env.reset()
    done = False

    while current_samples < num_samples:
        action = np.random.randint(0, 25)
        next_obs, reward, done, info = env.step(action)
        
        buffer.add(obs, action, reward, next_obs, done, info, True)
        current_samples += 1
        
        if done:
            obs = env.reset()
        else:
            obs = next_obs

    logger.info(f"收集了 {current_samples} 个样本")

    # 从缓冲区获取样本
    batch_size = 13
    obs_batch, actions_batch, next_obses_batch, info_batch = buffer.sample_inference(batch_size, "all")

    # 使用推理模型进行预测
    with torch.no_grad():
        pred_results = inference.eval_local_mask(obs_batch, actions_batch)
        reference_pred_results = reference_inference.eval_local_mask(obs_batch, actions_batch)
        pred_local_mask, pred_log_probs = pred_results
        reference_pred_local_mask, reference_pred_log_probs = reference_pred_results

        norm_list = []
        best_results = []
        high_mean_results = []  # mean > 0.5的结果
        low_mean_results = []   # mean <= 0.3的结果

        for sample_idx in range(batch_size):
            mask_shape = pred_log_probs[0][sample_idx, :, :].shape
            mask_shape = (mask_shape[0], mask_shape[1] - 1)  # remove action dimension
            action = actions_batch[sample_idx, 0, 0] // 5
            ground_truth_mask = info_batch['lcms'][sample_idx, 0, :, :]
            
            if torch.sum(ground_truth_mask[action, : action + 1]) > 3 or torch.sum(ground_truth_mask[-1, :]) > 7:
                ground_truth_mask[action, : action + 1] = 1.0
            else:
                # ground_truth_mask[action, 0] = 1.0
                ground_truth_mask[action, action] = 1.0
            
            ground_truth_mask[torch.arange(mask_shape[0]), torch.arange(mask_shape[1])] = 1.0
            
            pred_log_probs_mask = pred_log_probs[0][sample_idx, :, :-1]
            norm = torch.abs(pred_log_probs_mask.exp() - ground_truth_mask[:, :-1]).sum()
            # torch.abs((pred_log_probs_mask.exp() > 0.5).float() / ((ground_truth_mask[:, :-1] > 0.5).float() + 1e-6) - 1).sum()
            # + torch.abs((ground_truth_mask[:, :-1] > 0.5).float() / ((pred_log_probs_mask.exp() > 0.5).float() + 1e-6) - 1).sum()
            norm_list.append(norm)

            # 计算ground truth的mean值
            gt_mean = ground_truth_mask[:, :-1].mean().item()

            # 保存当前样本的结果
            result = {
                'model_name': name,
                'sample_idx': sample_idx,
                'norm': norm.item(),
                'gt_mean': gt_mean,
                'obs': obs_batch,
                'action': action.item(),
                'pred_log_probs': pred_log_probs_mask,
                'ground_truth_mask': ground_truth_mask[:, :-1],
                'reference_pred_log_probs': reference_pred_log_probs[0][sample_idx, :, :-1]
            }

            # 根据mean值分类
            if gt_mean > 0.5:
                high_mean_results.append(result)
            elif gt_mean <= 0.3:
                low_mean_results.append(result)

            best_results.append(result)

        # 找到每个类别中norm最小的3个样本
        high_mean_results.sort(key=lambda x: x['norm'])
        low_mean_results.sort(key=lambda x: x['norm'])
        best_high_mean = high_mean_results[:3] if high_mean_results else []
        best_low_mean = low_mean_results[:3] if low_mean_results else []

        # 保存结果
        results_dir = f"dwm_local_mask_analysis_results/{name}"
        os.makedirs(results_dir, exist_ok=True)

        # 保存high mean的结果
        for idx, result in enumerate(best_high_mean):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # DWM预测结果
            sns.heatmap(result['pred_log_probs'].exp().cpu().numpy(), ax=axes[0], annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                        yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
            axes[0].set_title("DWM Prediction")
            
            # Reference模型预测结果
            sns.heatmap(result['reference_pred_log_probs'].exp().cpu().numpy(), ax=axes[1], annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                        yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
            axes[1].set_title("Reference Prediction")
            
            # Ground Truth
            sns.heatmap(result['ground_truth_mask'].cpu().numpy(), ax=axes[2], annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                        yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
            axes[2].set_title("Ground Truth")
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/best_high_mean_comparison_{idx+1}_all.png")
            plt.close()

        # 保存high mean的数据
        if best_high_mean:
            data_dict = {
                'model_name': [r['model_name'] for r in best_high_mean],
                'sample_idx': [r['sample_idx'] for r in best_high_mean],
                'norm': [r['norm'] for r in best_high_mean],
                'gt_mean': [r['gt_mean'] for r in best_high_mean],
                'action': [r['action'] for r in best_high_mean]
            }
            df = pd.DataFrame(data_dict)
            df.to_csv(f"{results_dir}/best_high_mean_data.csv", index=False)

        # 保存low mean的结果
        for idx, result in enumerate(best_low_mean):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # DWM预测结果
            sns.heatmap(result['pred_log_probs'].exp().cpu().numpy(), ax=axes[0], annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                        yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
            axes[0].set_title("DWM Prediction")
            
            # Reference模型预测结果
            sns.heatmap(result['reference_pred_log_probs'].exp().cpu().numpy(), ax=axes[1], annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                        yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
            axes[1].set_title("Reference Prediction")
            
            # Ground Truth
            sns.heatmap(result['ground_truth_mask'].cpu().numpy(), ax=axes[2], annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                        yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
            axes[2].set_title("Ground Truth")
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/best_low_mean_comparison_{idx+1}_all.png")
            plt.close()

        # 保存low mean的数据
        if best_low_mean:
            data_dict = {
                'model_name': [r['model_name'] for r in best_low_mean],
                'sample_idx': [r['sample_idx'] for r in best_low_mean],
                'norm': [r['norm'] for r in best_low_mean],
                'gt_mean': [r['gt_mean'] for r in best_low_mean],
                'action': [r['action'] for r in best_low_mean]
            }
            df = pd.DataFrame(data_dict)
            df.to_csv(f"{results_dir}/best_low_mean_data.csv", index=False)
        
        return best_high_mean, best_low_mean

def main():
    set_seed_everywhere(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_high_mean_results = []
    all_low_mean_results = []
    
    # 遍历所有模型版本
    for i in range(1, 9):
        name = f"dwm-{i}"
        logger.info(f"处理模型 {name}")
        result = process_model(name, device)
        if result is not None:
            high_mean, low_mean = result
            all_high_mean_results.extend(high_mean)
            all_low_mean_results.extend(low_mean)
    
    # 找到所有模型中每个类别norm最小的3个结果
    if all_high_mean_results:
        all_high_mean_results.sort(key=lambda x: x['norm'])
        best_high_mean = all_high_mean_results[:3]
        logger.info(f"High mean类别中最好的3个结果:")
        for idx, result in enumerate(best_high_mean):
            logger.info(f"Rank {idx+1}: 模型 {result['model_name']}, norm值: {result['norm']}, mean值: {result['gt_mean']}")
        
        # 保存high mean的最佳结果
        data_dict = {
            'model_name': [r['model_name'] for r in best_high_mean],
            'sample_idx': [r['sample_idx'] for r in best_high_mean],
            'norm': [r['norm'] for r in best_high_mean],
            'gt_mean': [r['gt_mean'] for r in best_high_mean],
            'action': [r['action'] for r in best_high_mean]
        }
        df = pd.DataFrame(data_dict)
        df.to_csv("dwm_local_mask_analysis_results/best_high_mean_overall.csv", index=False)
    
    if all_low_mean_results:
        all_low_mean_results.sort(key=lambda x: x['norm'])
        best_low_mean = all_low_mean_results[:3]
        logger.info(f"Low mean类别中最好的3个结果:")
        for idx, result in enumerate(best_low_mean):
            logger.info(f"Rank {idx+1}: 模型 {result['model_name']}, norm值: {result['norm']}, mean值: {result['gt_mean']}")
        
        # 保存low mean的最佳结果
        data_dict = {
            'model_name': [r['model_name'] for r in best_low_mean],
            'sample_idx': [r['sample_idx'] for r in best_low_mean],
            'norm': [r['norm'] for r in best_low_mean],
            'gt_mean': [r['gt_mean'] for r in best_low_mean],
            'action': [r['action'] for r in best_low_mean]
        }
        df = pd.DataFrame(data_dict)
        df.to_csv("dwm_local_mask_analysis_results/best_low_mean_overall.csv", index=False)
    
    if not all_high_mean_results and not all_low_mean_results:
        logger.warning("没有找到任何有效的结果")

if __name__ == "__main__":
    main()