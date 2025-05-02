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
from fcdl.utils.utils import TrainingParams, get_env, update_obs_act_spec
from fcdl.utils.replay_buffer import ReplayBuffer

def process_model(name, device):
    # 设置模型路径
    model_path = f"data_new/iwhwang/causal_rl/Chemical/{name}/trained_models/inference_final"
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

    # 加载保存的模型
    inference.load(model_path, device)
    inference.eval()
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
        pred_local_mask, pred_log_probs = pred_results

        norm_list = []
        best_results = []

        for sample_idx in range(batch_size):
            mask_shape = pred_log_probs[0][sample_idx, :, :].shape
            mask_shape = (mask_shape[0], mask_shape[1] - 1)  # remove action dimension
            action = actions_batch[sample_idx, 0, 0] // 5
            ground_truth_mask = info_batch['lcms'][sample_idx, 0, :, :]
            
            if torch.sum(ground_truth_mask[action, : action + 1]) > 3 or torch.sum(ground_truth_mask[-1, :]) > 7:
                ground_truth_mask[action, : action + 1] = 1.0
            else:
                ground_truth_mask[action, 0] = 1.0
                ground_truth_mask[action, action] = 1.0
            
            ground_truth_mask[torch.arange(mask_shape[0]), torch.arange(mask_shape[1])] = 1.0
            
            pred_log_probs_mask = pred_log_probs[0][sample_idx, :, :-1]
            norm = torch.abs(pred_log_probs_mask.exp() - ground_truth_mask[:, :-1]).sum()
            norm_list.append(norm)

            # 保存当前样本的结果
            best_results.append({
                'model_name': name,
                'sample_idx': sample_idx,
                'norm': norm.item(),
                'obs': obs_batch,
                'action': action.item(),
                'pred_log_probs': pred_log_probs_mask,
                'ground_truth_mask': ground_truth_mask[:, :-1]
            })

        # 找到norm最小的样本
        min_norm_idx = torch.argmin(torch.tensor(norm_list)).item()
        best_result = best_results[min_norm_idx]

        # 保存结果
        results_dir = f"dwm_local_mask_analysis_results/{name}"
        os.makedirs(results_dir, exist_ok=True)

        # 保存预测结果和ground truth的对比图
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        sns.heatmap(best_result['pred_log_probs'].exp().cpu().numpy(), ax=axes[0], annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                    yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
        axes[0].set_title("Predicted Log Probabilities")
        
        sns.heatmap(best_result['ground_truth_mask'].cpu().numpy(), ax=axes[1], annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                    yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
        axes[1].set_title("Ground Truth Mask")
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/best_sample_comparison.png")
        plt.close()

        # 保存数据
        # 将数据转换为DataFrame格式
        data_dict = {
            'model_name': [name],
            'sample_idx': [best_result['sample_idx']],
            'norm': [best_result['norm']],
            'action': [best_result['action']]
        }
        
        # 保存为CSV
        df = pd.DataFrame(data_dict)
        df.to_csv(f"{results_dir}/best_sample_data.csv", index=False)
        
        return best_result

def main():
    set_seed_everywhere(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_best_results = []
    
    # 遍历所有模型版本
    for i in range(1, 9):
        name = f"dwm-{i}"
        logger.info(f"处理模型 {name}")
        result = process_model(name, device)
        if result is not None:
            all_best_results.append(result)
    
    # 找到所有模型中norm最小的结果
    if all_best_results:
        best_overall = min(all_best_results, key=lambda x: x['norm'])
        logger.info(f"所有模型中最好的结果来自模型 {best_overall['model_name']}")
        logger.info(f"最小norm值: {best_overall['norm']}")
        
        # 保存最佳结果
        # 将数据转换为DataFrame格式
        data_dict = {
            'model_name': [best_overall['model_name']],
            'sample_idx': [best_overall['sample_idx']],
            'norm': [best_overall['norm']],
            'action': [best_overall['action']]
        }
        
        # 保存为CSV
        df = pd.DataFrame(data_dict)
        df.to_csv("dwm_local_mask_analysis_results/best_overall_result.csv", index=False)
    else:
        logger.warning("没有找到任何有效的结果")

if __name__ == "__main__":
    main()