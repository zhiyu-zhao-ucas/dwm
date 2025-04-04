import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import traceback


class IdentityEncoder(nn.Module):
    # extract 1D obs and concatenate them
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.keys = [key for key in params.obs_keys if params.obs_spec[key].ndim == 1]
        self.feature_dim = np.sum([len(params.obs_spec[key]) for key in self.keys])
        # logger.info(f"feature_dim: {self.feature_dim}")
        # logger.info(f"keys: {self.keys}")
        # logger.info(f"obs_spec: {params.obs_spec}")

        self.continuous_state = params.continuous_state
        self.feature_inner_dim = params.feature_inner_dim
        if "Physical" in self.params.env_params.env_name:
            feature_inner_dim = []
            feature_inner_dim.extend([self.params.env_params.physical_env_params.width * self.params.env_params.physical_env_params.height] * self.params.env_params.physical_env_params.num_objects)  # For x and y coordinates
            self.feature_inner_dim = feature_inner_dim

        self.chemical_train = True
        self.chemical_match_type_train = list(range(len(self.keys)))
        self.to(params.device)
    
    def get_clean_obs(self, obs, detach=False):
        if self.continuous_state:
            obs = torch.cat([obs[k] for k in self.keys], dim=-1)
        else:
            obs = [obs_k_i
                   for k in self.keys
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim)]
        return obs
    
    def forward(self, obs, detach=False, info=None):
        if self.continuous_state:
            obs = torch.cat([obs[k] for k in self.keys], dim=-1)
            return obs
        else:
            if "Physical" in self.params.env_params.env_name:
                # logger.info(f"obs: {obs}")
                keys = [key for key in obs.keys()]
                # 遍历所有批次元素，由于每个元素是长度为2的列表，所以我们计算第一个元素乘以宽度加上第二个元素，得到一个新的obs，其中每个元素都是一个长度为1的列表
                if len(obs[keys[0]].shape) == 2:
                    obs = {k: (obs[k][:, 0] * self.params.env_params.physical_env_params.width + obs[k][:, 1]).view(-1, 1)
                            for k in keys}
                elif len(obs[keys[0]].shape) == 3:
                    obs = {k: (obs[k][:, :, 0] * self.params.env_params.physical_env_params.width + obs[k][:, :, 1]).unsqueeze(-1)
                            for k in keys}
                    
            # logger.info(f"obs: {obs}")
            obs = [obs_k_i
                   for k in self.keys
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            # logger.info(f"obs: {len(obs)}, obs: {obs}")
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim)]
            # if "Physical" in self.params.env_params.env_name:
            #     # for Physical env, we need to concatenate the even and odd features
            #     obs_even = obs[::2]
            #     obs_odd = obs[1::2]
            #     obs = [torch.cat([obs_even_item, obs_odd_item], dim=-1) for obs_even_item, obs_odd_item in zip(obs_even, obs_odd)]

            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "chemical_train", True):
                assert self.params.env_params.env_name in ["Chemical", "Physical"]
                if self.params.env_params.env_name == "Chemical":
                    assert self.params.env_params.chemical_env_params.continuous_pos
                else:
                    assert self.params.env_params.physical_env_params.continuous_pos
                test_scale = self.chemical_test_scale
                test_level = self.chemical_test_level
                if "Chemical" in self.params.env_params.env_name:
                    if test_level == 0:
                        noise_variable_list = [1, 2] # number of noisy nodes = 2
                    elif test_level == 1:
                        noise_variable_list = [1, 2, 3, 4] # number of noisy nodes = 4
                    elif test_level == 2:
                        noise_variable_list = [1, 2, 3, 4, 5, 6] # number of noisy nodes = 6
                elif "Physical" in self.params.env_params.env_name:
                    if test_level == 0:
                        noise_variable_list = [1]
                    elif test_level == 1:
                        noise_variable_list = [1, 2]
                    elif test_level == 2:
                        noise_variable_list = [1, 2, 3]
                self.chemical_match_type_test = list(set(self.chemical_match_type_train) - set(noise_variable_list))
                
                if test_scale == 0:
                    return obs
                else:
                    for i in noise_variable_list:
                        # logger.info(f"i: {i}, noise_variable_list: {noise_variable_list}, obs[i]: {obs[i]}")
                        obs[i] = obs[i] * torch.randn_like(obs[i]) * test_scale
                    return obs
            else: return obs

    # def forward(self, obs, detach=False, info=None):
    #     if self.continuous_state:
    #         obs = torch.cat([obs[k] for k in self.keys], dim=-1)
    #         return obs
    #     else:
    #         # logger.info(f"obs: {obs}")
    #         obs = [obs_k_i
    #                for k in self.keys
    #                for obs_k_i in torch.unbind(obs[k], dim=-1)]
    #         # logger.info(f"obs: {len(obs)}, obs: {obs}")
    #         if "Physical" in self.params.env_params.env_name:
    #             # for Physical env, we need to concatenate the even and odd features
    #             # logger.debug(f"obs: {len(obs)}, obs[0]: {obs[0]}")
    #             # 检查是否处理的是三维张量（带批次维度）
    #             if len(obs[0].shape) > 1 and obs[0].shape[0] > 1:  # 批处理模式
    #                 batch_size = obs[0].shape[0]
    #                 logger.info(f"batch_size: {batch_size}")
    #                 device = obs[0].device
                    
    #                 # 单独处理每个批次元素
    #                 results = []
    #                 for b in range(batch_size):
    #                     batch_even = torch.vstack([t[b] for t in obs[::2]]).cpu()
    #                     batch_odd = torch.vstack([t[b] for t in obs[1::2]]).cpu()
                        
    #                     # 按原方式组合坐标
    #                     batch_new = batch_even * self.params.env_params.physical_env_params.width + batch_odd
    #                     results.append(batch_new)
                    
    #                 # 重新组合为批次
    #                 obs = torch.stack(results).to(device)
    #             else:
    #                 # 原来的二维张量处理代码
    #                 obs_even = torch.vstack(obs[::2]).cpu()
    #                 obs_odd = torch.vstack(obs[1::2]).cpu()
    #                 new_obs = torch.tensor(obs_even) * self.params.env_params.physical_env_params.width + torch.tensor(obs_odd)
    #                 obs = new_obs.to(obs[0].device)
    #             # logger.info(f"obs: {obs}")
    #             # obs = [torch.cat([obs_even_item, obs_odd_item], dim=-1) for obs_even_item, obs_odd_item in zip(obs_even, obs_odd)]
    #         obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
    #                for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim)]
    #         # if "Physical" in self.params.env_params.env_name:
    #         #     # for Physical env, we need to concatenate the even and odd features
    #         #     obs_even = obs[::2]
    #         #     obs_odd = obs[1::2]
    #         #     obs = [torch.cat([obs_even_item, obs_odd_item], dim=-1) for obs_even_item, obs_odd_item in zip(obs_even, obs_odd)]

    #         # overwrite some observations for out-of-distribution evaluation
    #         if not getattr(self, "chemical_train", True):
    #             assert self.params.env_params.env_name in ["Chemical", "Physical"]
    #             if self.params.env_params.env_name == "Chemical":
    #                 assert self.params.env_params.chemical_env_params.continuous_pos
    #             else:
    #                 assert self.params.env_params.physical_env_params.continuous_pos
    #             test_scale = self.chemical_test_scale
    #             test_level = self.chemical_test_level
    #             if "Chemical" in self.params.env_params.env_name:
    #                 if test_level == 0:
    #                     noise_variable_list = [1, 2] # number of noisy nodes = 2
    #                 elif test_level == 1:
    #                     noise_variable_list = [1, 2, 3, 4] # number of noisy nodes = 4
    #                 elif test_level == 2:
    #                     noise_variable_list = [1, 2, 3, 4, 5, 6] # number of noisy nodes = 6
    #             elif "Physical" in self.params.env_params.env_name:
    #                 if test_level == 0:
    #                     noise_variable_list = [1]
    #                 elif test_level == 1:
    #                     noise_variable_list = [1, 2]
    #                 elif test_level == 2:
    #                     noise_variable_list = [1, 2, 3]
    #             self.chemical_match_type_test = list(set(self.chemical_match_type_train) - set(noise_variable_list))
                
    #             if test_scale == 0:
    #                 return obs
    #             else:
    #                 for i in noise_variable_list:
    #                     # logger.info(f"i: {i}, noise_variable_list: {noise_variable_list}, obs[i]: {obs[i]}")
    #                     obs[i] = obs[i] * torch.randn_like(obs[i]) * test_scale
    #                 return obs
    #         else: return obs


_AVAILABLE_ENCODERS = {"identity": IdentityEncoder}


def make_encoder(params):
    encoder_type = params.encoder_params.encoder_type
    return _AVAILABLE_ENCODERS[encoder_type](params)
