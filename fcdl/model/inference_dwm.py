import torch
import torch.nn.functional as F
from copy import deepcopy

from .inference_ours_masking import InferenceOursMask
from .inference_utils import forward_network, reset_layer
from loguru import logger
import traceback


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
        # logger.info(f"actions before action_single: {actions.shape}")
        if not "Causal" in self.params.env_params.env_name:
            action_single = actions[:, 0, :].unsqueeze(dim=-1)
        else:
            action_single = actions[:, 0, :]
        # logger.info(f"action_single: {action_single.shape}")
        try:
            next_features_stack_single = torch.stack(next_features)[:, :, 0, :]
        except Exception as e:
            # logger.info(f"torch.stack(next_features): {next_features.shape}")
            next_features_stack_single = next_features[:, 0, :]
        # logger.info(f"features used for local mask: {len(features)}, features[0]: {features[0].shape}")
        # logger.info(f"action_single for local mask: {action_single.shape}")
        local_masks, log_probs = self.get_local_mask(features, action_single, training=True)
        if not "Causal" in self.params.env_params.env_name:
            diff = torch.abs(next_features_stack_single - torch.stack(features)).sum(dim=-1)
        else:
            diff = torch.abs(next_features_stack_single - features)
        # logger.info(f"next_features_stack_single: {next_features_stack_single.shape}")
        # logger.info(f"diff: {diff.shape}")
        # logger.info(f"diff: {diff}")
        logger.info(f"log_probs: {log_probs.shape}")
        changed_nodes = (diff < 1e-6).nonzero(as_tuple=True)
        unchanged_nodes = (diff >= 1e-3).nonzero(as_tuple=True)
        # logger.info(f"changed_nodes: {changed_nodes}")
        if not "Causal" in self.params.env_params.env_name:
            batch_indices = changed_nodes[0]
            node_indices = changed_nodes[1]
            if "Chemical" in self.params.env_params.env_name:
                action_taken = action_single[node_indices].long() // self.params.env_params.chemical_env_params.num_colors
            else:
                action_taken = action_single[node_indices].long().cpu() // 5
            selected_log_probs = log_probs[0, node_indices, batch_indices, action_taken.squeeze(dim=-1)]
            log_probs_mean = self.params.inference_params.causal_coef * selected_log_probs.mean()
        else:
            selected_log_probs_changed = log_probs[0, changed_nodes[0], changed_nodes[1], -4:]
            selected_log_probs_unchanged = log_probs[0, unchanged_nodes[0], unchanged_nodes[1], -4:]
            log_probs_mean = self.params.inference_params.causal_coef * (selected_log_probs_changed.mean() - selected_log_probs_unchanged.mean())
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
        if "Causal" in self.params.env_params.env_name:
            action_feature = self.extract_action_feature(actions)
            state_feature = self.extract_state_feature(feature)

            # logger.info(f"action_feature: {action_feature.shape}")
            # logger.info(f"state_feature: {state_feature.shape}")
            bs = state_feature.size(1)
            local_mask, prob = self.local_causal_model(state_feature, action_feature, current_pred_step, training=training)
            # backup_training = deepcopy(self.training)
            # self.training = False
            # dist = self.forward_step(feature, action, current_pred_step)
            # feature = self.sample_from_distribution(dist)
            # self.training = backup_training
            if current_pred_step == 0:
                local_masks.append(local_mask)
                log_probs.append(torch.log(prob + 1e-3))
            else:
                local_masks.append(torch.zeros_like(local_mask))
                log_probs.append(torch.zeros_like(prob))
            current_pred_step += 1
            local_masks = torch.stack(local_masks)
            log_probs = torch.stack(log_probs)
            
            return local_masks, log_probs
        else:
            for action in actions:
                action_feature = self.extract_action_feature(action)
                state_feature = self.extract_state_feature(feature)

                # logger.info(f"action_feature: {action_feature.shape}")
                # logger.info(f"state_feature: {state_feature.shape}")
                bs = state_feature.size(1)
                local_mask, prob = self.local_causal_model(state_feature, action_feature, current_pred_step, training=training)
                # backup_training = deepcopy(self.training)
                # self.training = False
                # dist = self.forward_step(feature, action, current_pred_step)
                # feature = self.sample_from_distribution(dist)
                # self.training = backup_training
                # if current_pred_step == 0:
                local_masks.append(local_mask)
                log_probs.append(torch.log(prob + 1e-3))
                # else:
                #     local_masks.append(torch.zeros_like(local_mask))
                #     log_probs.append(torch.zeros_like(prob))
                current_pred_step += 1
            local_masks = torch.stack(local_masks)
            log_probs = torch.stack(log_probs)
            
            return local_masks, log_probs

    def eval_local_mask(self, obses, actions):
        features = self.encoder(obses)
        if len(actions.shape) < 2:
            actions = actions.view(-1, 1, 1)
        # logger.info(f"actions in eval_local_mask: {actions}")
        # logger.info(f"features: {len(features)}, features[0]: {features[0].shape}")
        local_masks, log_probs = self.get_local_mask(features, actions, training=False)
        return local_masks, log_probs
    
    def reset_params(self):
        if self.params.env_params.env_name == "Causal":
            pass
        else:
            super(InferenceDWM, self).reset_params()
        