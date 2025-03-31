import torch
import torch.nn.functional as F

from .inference_ours_masking import InferenceOursMask
from .inference_utils import forward_network, reset_layer
from loguru import logger


class InferenceDWM(InferenceOursMask):
    def __init__(self, encoder, params):
        logger.info("InferenceDWM")
        super(InferenceDWM, self).__init__(encoder, params)
        self.count = 0

    def update(self, obses, actions, next_obses, eval=False):
        self.is_eval = eval
        assert not self.training == self.is_eval
        features = self.encoder(obses)
        next_features = self.encoder(next_obses)
        pred_next_dist = self.forward_with_feature(features, actions)

        masked_pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_features)
        masked_pred_loss = masked_pred_loss.mean()
        full_pred_loss = torch.zeros_like(masked_pred_loss)
        loss = full_pred_loss + masked_pred_loss
        loss_detail = {"pred_loss": masked_pred_loss, 
                       "full_pred_loss": full_pred_loss
        }
        # logger.info(f"obses: {len(obses)}")
        # logger.info(f"actions: {actions.shape}")
        # logger.info(f"next_obses: {len(next_obses)}") 
        # # logger.info(f"obses: {obses['obj0']}")
        # # logger.info(f"next_obses[0]: {next_obses['obj0']}")
        # logger.info(f"features: {len(features)}")
        # logger.info(f"next_features: {len(next_features)}")
        # logger.info(f"features[0]: {features[0].shape}")
        # logger.info(f"next_features[0]: {next_features[0].shape}")
        action_single = actions[:, 0, :].unsqueeze(dim=-1)
        # logger.info(f"action_single: {action_single.shape}")
        next_features_stack_single = torch.stack(next_features)[:, :, 0, :]
        # logger.info(f"next_features_stack_single: {next_features_stack_single.shape}")
        # features_stack = torch.stack(features)
        # logger.info(f"features_stack: {features_stack.shape}")
        local_masks, log_probs = self.get_local_mask(features, action_single, training=True)
        # logger.info(f"log_probs: {log_probs.requires_grad}")
        # logger.info(f"local_masks: {local_masks.shape}")
        # logger.info(f"log_probs: {log_probs.min()}, {log_probs.max()}")
        # logger.info(f"log_probs: {log_probs.shape}")
        diff = torch.abs(next_features_stack_single - torch.stack(features)).sum(dim=-1)
        # logger.info(f"next_features_stack_single - torch.stack(features): {(next_features_stack_single - torch.stack(features)).shape}")
        # logger.info(f"torch.stack(features): {torch.stack(features).shape}")
        # logger.info(f"next_features_stack_single: {next_features_stack_single}")
        # logger.info(f"diff: {diff}")
        changed_nodes = (diff < 1e-3).nonzero(as_tuple=True)
        # logger.info(f"changed_nodes: {changed_nodes}")
        # logger.info(f"diff: {diff.shape}")
        # logger.info(f"changed_nodes: {changed_nodes}")
        batch_indices = changed_nodes[0]
        node_indices = changed_nodes[1]
        action_taken = action_single[node_indices].long() // self.params.env_params.chemical_env_params.num_colors
        # logger.info(f"action_taken: {action_taken}")
        # logger.info(f"batch_indices: {batch_indices[:3]}")
        # logger.info(f"node_indices: {node_indices[:3]}")
        # logger.info(f"action_taken: {action_taken[:3]}")
        # logger.info(f"log_probs: {log_probs.shape}")
        selected_log_probs = log_probs[0, node_indices, batch_indices, action_taken.squeeze(dim=-1)]
        # if self.params.inference_params.causal_coef == 0.0 or (self.count / self.params.training_params.total_steps < -1):
        #     selected_log_probs = torch.zeros_like(selected_log_probs)
        #     logger.info(f"self.params.inference_params.causal_coef: {self.params.inference_params.causal_coef}")
        log_probs_mean = self.params.inference_params.causal_coef * selected_log_probs.mean()
        loss_detail['log_probs_mean'] = log_probs_mean / (self.params.inference_params.causal_coef + 1e-8)
        
        if self.learn_codebook:
            ours_loss = self.local_causal_model.total_loss() + log_probs_mean
            loss = loss + ours_loss
            loss_detail['reg_loss']= self.local_causal_model.reg_loss.mean()
            loss_detail['vq_loss']= self.local_causal_model.vq_loss.mean()
            loss_detail['commit_loss']= self.local_causal_model.commit_loss.mean()

        if not eval:
            self.backprop(loss, loss_detail)
        self.count += 1

        return loss_detail
    
    def get_local_mask(self, feature, actions, training=True):
        # logger.info(f"features: {len(feature)}, features[0]: {feature[0].shape}")
        # logger.info(f"actions: {actions.shape}")
        if not self.continuous_action:
            actions = F.one_hot(actions.squeeze(dim=-1), self.action_dim).float()
        actions = torch.unbind(actions, dim=-2)
        
        local_masks = []
        log_probs = []
        current_pred_step = 0
        for action in actions:
            action_feature = self.extract_action_feature(action)
            state_feature = self.extract_state_feature(feature)

            bs = state_feature.size(1)
            local_mask, prob = self.local_causal_model(state_feature, action_feature, current_pred_step, training=training)
            local_masks.append(local_mask)
            log_probs.append(torch.log(prob + 1e-3))
            current_pred_step += 1
        local_masks = torch.stack(local_masks)
        log_probs = torch.stack(log_probs)
        
        return local_masks, log_probs

    def eval_local_mask(self, obses, actions):
        features = self.encoder(obses)
        if len(actions.shape) < 2:
            # logger.info(f"actions: {actions.shape}")
            actions = actions.view(-1, 1, 1)
        local_masks, log_probs = self.get_local_mask(features, actions, training=False)
        return local_masks, log_probs
        

    # def forward_with_feature(self, feature, actions, **kwargs):
    #     if not self.continuous_action:
    #         actions = F.one_hot(actions.squeeze(dim=-1), self.action_dim).float()
    #     actions = torch.unbind(actions, dim=-2)
        
    #     dists = []
    #     current_pred_step = 0
    #     for action in actions:
    #         dist = self.forward_step(feature, action, current_pred_step)
    #         feature = self.sample_from_distribution(dist)
    #         dists.append(dist)
    #         current_pred_step += 1
    #     dists = self.stack_dist(dists)

    #     return dists
    
    # def forward_step(self, feature, action, current_pred_step):
    #     if self.training:
    #         sampling_num = self.local_mask_sampling_num
    #     else: 
    #         sampling_num = self.eval_local_mask_sampling_num
        
    #     action_feature = self.extract_action_feature(action)
    #     state_feature = self.extract_state_feature(feature)
        
    #     bs = state_feature.size(1)
    #     # logger.info(f"bs: {bs}")
    #     if self.use_gt_global_mask:
    #         global_mask = self.gt_global_mask.clone().repeat(bs, 1, 1)
    #         prob = global_mask
    #         global_mask = global_mask.repeat(sampling_num, 1, 1, 1)
    #         local_mask = global_mask
    #     else:
    #         local_mask, prob = self.local_causal_model(state_feature, action_feature, current_pred_step, training=self.training)
    #         if not self.training:
    #             prob = (prob > 0.5).float()
    #         prob = prob.detach()
        
    #     local_mask = local_mask.permute(0, 2, 3, 1)
    #     local_mask = local_mask.unsqueeze(dim=-1)
    #     prob = prob.permute(1, 2, 0)
    #     prob = prob.unsqueeze(dim=-1)
    #     assert sampling_num == local_mask.size(0)

    #     return self.forward_with_local_mask(state_feature, action_feature, feature, local_mask, prob, current_pred_step)

    # def compute_causal_loss(self, prev_state_feature, new_state_feature, local_mask_prob, intervention_var, threshold=0):
    #     # 计算状态特征差异
    #     new_state_feature = torch.stack(new_state_feature)
    #     prev_state_feature = torch.stack(prev_state_feature)
    #     new_state_feature = new_state_feature.permute(1, 0, 2)
    #     prev_state_feature = prev_state_feature.permute(1, 0, 2)
        
    #     # 计算变化量并找到变化明显的节点
    #     diff = torch.abs(new_state_feature - prev_state_feature).sum(dim=-1)
    #     changed_nodes = (diff > threshold).nonzero(as_tuple=True)
                
    #     if len(changed_nodes[0]) == 0:
    #         return torch.tensor(0.0, device=prev_state_feature.device)
        
    #     # 获取批次索引和节点索引
    #     batch_indices = changed_nodes[0]
    #     node_indices = changed_nodes[1]
        
    #     # 使用干预变量作为源节点索引
    #     source_indices = intervention_var[batch_indices].long()  # 确保是整数索引
        
    #     # 提取对应的局部掩码概率
    #     # local_mask_prob[batch, target_node, source_node]
    #     relevant_probs = local_mask_prob[batch_indices, node_indices, source_indices]
        
    #     # 计算负对数损失
    #     loss = -torch.log(relevant_probs + 1e-8).mean()
        
    #     return loss

    # # 新增方法：基于因果信号优化local_mask生成概率
    # def optimize_causal_structure(self, prev_state_feature, new_state_feature, local_mask_prob, intervention_var, threshold=0.1):
    #     logger.info(f"local_mask_prob: {local_mask_prob.size()}")
    #     loss = self.compute_causal_loss(prev_state_feature, new_state_feature, local_mask_prob, intervention_var, threshold)
    #     loss.backward()
    #     return loss