import os
import sys
import numpy as np
import torch
import pandas as pd
from loguru import logger
from fcdl.utils.utils import set_seed_everywhere
from collections import defaultdict

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

def calculate_shd(pred_mask, ground_truth_mask, threshold=0.3):
    """
    计算结构性汉明距离 (Structural Hamming Distance)
    SHD: 两个二值化掩码之间不同位置的数量
    
    Args:
        pred_mask: 预测的概率掩码
        ground_truth_mask: 真实掩码
        threshold: 将概率转换为二值掩码的阈值
    
    Returns:
        shd: 结构性汉明距离
    """
    # 二值化预测掩码
    # pred_binary = (pred_mask > threshold).float()
    # # 二值化真实掩码 (如果尚未二值化)
    # gt_binary = (ground_truth_mask > threshold).float()
    
    # # 计算不同位置的数量
    # shd = torch.abs(pred_binary - gt_binary).sum().item()
    shd = torch.norm(pred_mask - ground_truth_mask, p=1).item()
    return shd

def process_model(name, device):
    # 设置模型路径
    model_path = f"{name}/trained_models/inference_final"
    params_path = f"{name}/params"
    env_params_path = f"{name}/params"

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
    num_samples = 1000
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
    batch_size = 1000
    obs_batch, actions_batch, next_obses_batch, info_batch = buffer.sample_inference(batch_size, "all")

    # 使用推理模型进行预测
    with torch.no_grad():
        pred_results = inference.eval_local_mask(obs_batch, actions_batch)
        pred_local_mask, pred_log_probs = pred_results

        # 统计指标
        shd_list = []
        l1_list = []
        results_dir = f"dwm_mask_metrics/{name}"
        os.makedirs(results_dir, exist_ok=True)

        for sample_idx in range(batch_size):
            mask_shape = pred_log_probs[0][sample_idx, :, :].shape
            mask_shape = (mask_shape[0], mask_shape[1] - 1)  # remove action dimension
            action = actions_batch[sample_idx, 0, 0] // 5
            ground_truth_mask = info_batch['lcms'][sample_idx, 0, :, :]
            
            if torch.sum(ground_truth_mask[action, : action + 1]) > 3 or torch.sum(ground_truth_mask[-1, :]) > 7:
                ground_truth_mask[action, : action + 1] = 1.0
            else:
                ground_truth_mask[action, action] = 1.0
            
            ground_truth_mask[torch.arange(mask_shape[0]), torch.arange(mask_shape[1])] = 1.0
            
            pred_log_probs_mask = pred_log_probs[0][sample_idx, :, :-1]
            pred_probs = pred_log_probs_mask.exp()
            
            # 计算L1距离
            l1_norm = torch.abs(pred_probs - ground_truth_mask[:, :-1]).sum().item()
            l1_list.append(l1_norm)
            
            # 计算结构性汉明距离
            shd = calculate_shd(pred_probs, ground_truth_mask[:, :-1])
            shd_list.append(shd)

        # 计算平均值
        avg_shd = sum(shd_list) / len(shd_list)
        avg_l1 = sum(l1_list) / len(l1_list)
        
        # 记录结果
        metrics = {
            'model_name': name,
            'avg_shd': avg_shd,
            'avg_l1': avg_l1,
            'num_samples': batch_size
        }
        
        logger.info(f"模型 {name} 结果:")
        logger.info(f"  平均SHD: {avg_shd:.4f}")
        logger.info(f"  平均L1距离: {avg_l1:.4f}")
        
        # 保存详细结果
        df_metrics = pd.DataFrame({
            'sample_idx': list(range(batch_size)),
            'shd': shd_list,
            'l1_norm': l1_list
        })
        
        os.makedirs(results_dir, exist_ok=True)
        df_metrics.to_csv(f"{results_dir}/metrics_details.csv", index=False)
        
        # 保存汇总结果
        with open(f"{results_dir}/summary.txt", "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Average SHD: {avg_shd:.4f}\n")
            f.write(f"Average L1 Distance: {avg_l1:.4f}\n")
            f.write(f"Number of samples: {batch_size}\n")
        
        return metrics

def main():
    set_seed_everywhere(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_metrics = []
    
    # 处理所有模型 (1-16)
    for model_num in [1, 2, 4, 8, 16]:
        name = f"dwm_fork_code_size/dwm_fork_code_size_{model_num}"
        logger.info(f"处理模型 {name}")
        metrics = process_model(name, device)
        if metrics is not None:
            all_metrics.append(metrics)
    
    # 保存所有模型的结果
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        os.makedirs("dwm_mask_metrics", exist_ok=True)
        df.to_csv("dwm_mask_metrics/all_models_summary.csv", index=False)
        logger.info("已保存所有模型的结果汇总")
    else:
        logger.warning("没有找到任何有效的结果")

if __name__ == "__main__":
    main() 