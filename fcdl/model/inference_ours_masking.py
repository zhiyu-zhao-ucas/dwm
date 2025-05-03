import torch
import torch.nn.functional as F
from loguru import logger

from .inference_ours_base import InferenceOursBase
from .inference_utils import forward_network, reset_layer

class InferenceOursMask(InferenceOursBase):
    def __init__(self, encoder, params):
        logger.info("InferenceOursMask")
        super(InferenceOursMask, self).__init__(encoder, params)

    def init_model(self):
        super(InferenceOursMask, self).init_model()

    def reset_params_sa_feature(self):
        for w, b in zip(self.sa_feature_weights, self.sa_feature_biases):
            for i in range(self.num_state_var):
                reset_layer(w[i], b[i])
    
    def forward_with_local_mask(self, state_feature, action_feature, feature, local_mask, prob, current_pred_step):
        s_dim = state_feature.size(0)
        sampling_num = local_mask.size(0)

        original_sa_feature = torch.cat([state_feature, action_feature], dim=0)
        original_sa_feature = original_sa_feature.repeat(s_dim, 1, 1, 1)

        if self.code_labeling:
            code_label = self.get_code_label(current_pred_step)
            code_label = code_label.repeat(s_dim, 1, 1)

        sampled_dist = []
        
        for i in range(sampling_num):
            sa_feature = original_sa_feature * local_mask[i]
            
            sa_feature = sa_feature.permute(0, 2, 1, 3)
            sa_feature = sa_feature.reshape(*sa_feature.shape[:2], -1)
            
            if self.code_labeling:
                sa_feature = torch.cat([sa_feature, code_label], dim=-1)
                # Code labeling is optional in practice
                # See Appendix C.3.2 for the discussion on the design choices

            sa_feature = forward_network(sa_feature, self.sa_feature_weights, self.sa_feature_biases)
            sa_feature = F.relu(sa_feature)

            sampled_dist.append(self.predict_from_sa_feature(sa_feature, feature))
        
        mean_dist = self.mean_dist(sampled_dist)

        return mean_dist
    
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
