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

def main():
    set_seed_everywhere(0)
    # 设置模型路径
    name = f"dwm-2"  # 根据实际模型名称修改
    model_path = f"data3/iwhwang/causal_rl/Chemical/{name}/trained_models/inference_final"
    params_path = f"data3/iwhwang/causal_rl/Chemical/{name}/params"
    env_params_path = f"data3/iwhwang/causal_rl/Chemical/{name}/params"

    # 检查文件是否存在
    assert os.path.exists(model_path), f"模型文件不存在: {model_path}"
    assert os.path.exists(params_path), f"参数文件不存在: {params_path}"
    assert os.path.exists(env_params_path), f"环境参数文件不存在: {env_params_path}"

    # 加载参数
    params_dict = torch.load(params_path)
    params = TrainingParams(training_params_fname="policy_params.json", train=False)

    # 将加载的参数字典复制到params对象
    for key, value in params_dict.items():
        setattr(params, key, value)

    # 加载环境参数
    env_params = torch.load(env_params_path)
    print(f"已加载参数和环境设置")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.device = device
    print(f"使用设备: {device}")

    # 创建环境
    params.env_params.num_env = 1  # 确保只使用一个环境
    env = get_env(params)
    print(f"环境创建完成: {params.env_params.env_name}")

    # 更新观测和动作空间
    update_obs_act_spec(env, params)

    # 创建编码器和推理模型
    encoder = make_encoder(params)
    inference = InferenceDWM(encoder, params)

    # 加载保存的模型
    inference.load(model_path, device)
    inference.eval()
    print(f"推理模型加载完成")

    # 创建缓冲区来收集样本
    buffer = ReplayBuffer(params)
    num_samples = 100
    current_samples = 0

    # 收集样本
    obs = env.reset()
    done = False

    while current_samples < num_samples:
        # 随机选择动作 - 对于Causal环境，动作空间是离散的，范围是[0, 3]
        action = np.random.randint(0, 25)
        next_obs, reward, done, info = env.step(action)
        
        # 添加到缓冲区
        buffer.add(obs, action, reward, next_obs, done, info, True)
        current_samples += 1
        
        # 如果回合结束，重置环境
        if done:
            obs = env.reset()
        else:
            obs = next_obs

    print(f"收集了 {current_samples} 个样本")

    # 从缓冲区获取样本
    batch_size = 13
    obs_batch, actions_batch, next_obses_batch, info_batch = buffer.sample_inference(batch_size, "all")
    logger.info(f"info_batch: {info_batch['lcms'].shape}")

    # 使用推理模型进行预测
    with torch.no_grad():
        # 获取模型预测的local mask
        pred_results = inference.eval_local_mask(obs_batch, actions_batch)
        # logger.info(f"obs_batch: {obs_batch.shape}")
        logger.info(f"actions_batch: {actions_batch.shape}")
        
        # 提取预测的local mask和log probabilities
        pred_local_mask, pred_log_probs = pred_results

        # 打印预测结果的形状
        print(f"预测的local mask形状: {pred_local_mask.shape}")
        print(f"预测的log probabilities形状: {pred_log_probs.shape}")
        print(f"预测的单个log probabilities形状: {pred_log_probs[0, 0, :, :].shape}")

        # save the norm of (local mask - ground truth mask)
        norm_list = []
        # for sample_idx in range(batch_size):
        #     norm = torch.abs(pred_local_mask[0, 0, sample_idx, :, :] - info_batch['lcms'][sample_idx, 0, :, :]).sum()
        #     norm_list.append(norm)
        # norm_list = torch.tensor(norm_list)
        # logger.info(f"norm_list: {norm_list}")
        # get the top 3 min norm
        # 可视化样本的local mask
        for sample_idx in range(batch_size):
            plt.figure(figsize=(10, 8))
            
            # 获取矩阵的维度
            mask_shape = pred_log_probs[0][sample_idx, :, :].shape
            mask_shape = (mask_shape[0], mask_shape[1] - 1)  # remove action dimension
            action = actions_batch[sample_idx, 0, 0] // 5
            ground_truth_mask = info_batch['lcms'][sample_idx, 0, :, :]
            if torch.sum(ground_truth_mask[action, : action + 1]) > 3 or torch.sum(ground_truth_mask[-1, :]) > 7:
                ground_truth_mask[action, : action + 1] = 1.0
            else:
                ground_truth_mask[action, 0] = 1.0
                ground_truth_mask[action, action] = 1.0
            # let the diagonal of ground truth mask be 1
            ground_truth_mask[torch.arange(mask_shape[0]), torch.arange(mask_shape[1])] = 1.0
            
            pred_log_probs_mask = pred_log_probs[0][sample_idx, :, :-1]
            norm = torch.abs(pred_log_probs_mask.exp() - ground_truth_mask[:, :-1]).sum()
            norm_list.append(norm)
            logger.info(f"预测的log probabilities形状: {mask_shape}")
            
            # 创建热力图
            sns.heatmap(pred_log_probs_mask.exp().cpu().numpy(), 
                        annot=True, 
                        fmt=".2f", 
                        cmap="YlGnBu",
                        xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                        yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
            
            plt.title("Local Mask")
            plt.tight_layout()
            plt.savefig(f"dwm_local_mask_sample_{sample_idx}.png")
            plt.close()

            # 创建热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(ground_truth_mask[:, :-1].cpu().numpy(), 
                        annot=True, 
                        fmt=".2f", 
                        cmap="YlGnBu",
                        xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                        yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
            plt.title("Local Mask")
            plt.tight_layout()
            plt.savefig(f"dwm_ground_truth_mask_sample_{sample_idx}.png")
            plt.close()
            # 将以上图拼在一起
            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            sns.heatmap(pred_log_probs[0][sample_idx, :, :-1].exp().cpu().numpy(), ax=axes[0], annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                        yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
            axes[0].set_title("Predicted Log Probabilities")
            
            sns.heatmap(ground_truth_mask[:, :-1].cpu().numpy(), ax=axes[1], annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=[f"Obj {i}" for i in range(mask_shape[1])],
                        yticklabels=[f"Obj {i}" for i in range(mask_shape[0])])
            axes[1].set_title("Ground Truth Mask")
            
            plt.tight_layout()
            plt.savefig(f"dwm_local_mask_analysis_results/dwm_local_mask_analysis_sample_{sample_idx}.png")
            plt.close()
            
            # 保存预测结果
            results_dir = f"dwm_local_mask_analysis_results"
            os.makedirs(results_dir, exist_ok=True)
            # 将obs_batch和actions_batch保存为同一个csv
            # 将字典形式的obs_batch转换为tensor
            obs_dict = torch.cat([v.cpu() for k, v in obs_batch.items()], axis=1)
            obs_df = pd.DataFrame(obs_dict)
            actions_df = pd.DataFrame(actions_batch[:, 0, :].view(-1).cpu().numpy())
            # 将obs_batch和actions_batch保存为同一个csv
            data = pd.concat([obs_df, actions_df], axis=1)
            data.to_csv(f"{results_dir}/obs_and_actions_batch.csv", index=True)
            results_dir = f"{results_dir}/dwm_local_mask_analysis_results"

        print(f"已保存local mask预测结果到 {results_dir} 目录")
        top_3_min_norm = torch.topk(torch.tensor(norm_list), 3, dim=0, largest=False)
        logger.info(f"top_3_min_norm: {top_3_min_norm}")


if __name__ == "__main__":
    main() 