"""
Example for interacting with a multi-process Physical environment.
"""

import numpy as np
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import List

from fcdl.env.physical_env import Physical
from fcdl.model.encoder import IdentityEncoder
from fcdl.utils.utils import get_env, update_obs_act_spec, set_seed_everywhere


# Create parameter structure to initialize the environment
@dataclass
class EnvParams:
    env_name: str = "Physical"
    num_env: int = 4  # Number of parallel environments
    
@dataclass
class PhysicalEnvParams:
    width: int = 5
    height: int = 5
    num_objects: int = 3
    num_rand_objects: int = 0
    render_type: str = "shapes"  # Options: grid, circles, shapes, cubes
    mode: str = "ZeroShotColor"
    dense_reward: bool = True
    max_steps: int = 50
    gt_local_mask: bool = False
    num_weights: int = None

@dataclass
class EncoderParams:
    encoder_type: str = "identity"

@dataclass
class Params:
    env_params: EnvParams = EnvParams()
    encoder_params: EncoderParams = EncoderParams()
    seed: int = 42
    device: str = "cpu"
    continuous_state: bool = False
    obs_keys: List[str] = None  # Add this attribute
    goal_keys: List[str] = None  # Add this attribute

    def __post_init__(self):
        self.env_params.physical_env_params = PhysicalEnvParams()
        # Initialize as empty lists if None
        if self.obs_keys is None:
            self.obs_keys = []
        if self.goal_keys is None:
            self.goal_keys = []


def create_encoder_params(env, num_envs):
    """Create parameters needed for the IdentityEncoder based on the environment."""
    params = Params()
    
    # Get observation specification from environment
    obs = env.reset()
    obs_spec = {}
    obs_keys = []
    
    # For vectorized environments, the observation is a dictionary with arrays
    # where the first dimension is the environment index
    for key, value in obs.items():
        # Store as specification the shape without batch dimension
        obs_spec[key] = value[0] if num_envs > 1 else value
        if (value[0] if num_envs > 1 else value).ndim == 1:
            obs_keys.append(key)
    
    # Add required attributes to params
    params.obs_spec = obs_spec
    params.obs_keys = obs_keys
    
    # Add feature inner dimensions
    feature_inner_dim = []
    for key in obs_keys:
        if key.startswith("obj") or key.startswith("target_obj"):
            feature_inner_dim.extend([1, 1])  # For x and y coordinates
    
    params.feature_inner_dim = np.array(feature_inner_dim)
    
    return params


def main():
    # Initialize parameters
    params = Params()
    set_seed_everywhere(params.seed)
    
    # Populate obs_keys and goal_keys based on environment config
    num_objects = params.env_params.physical_env_params.num_objects
    params.obs_keys = [f"obj{i}" for i in range(num_objects)]
    params.goal_keys = [f"target_obj{i}" for i in range(num_objects)]
    
    # Create environment using get_env
    env = get_env(params)
    num_envs = params.env_params.num_env
    is_vecenv = num_envs > 1
    
    # Update observation and action specs
    update_obs_act_spec(env, params)
    
    # Reset environment
    obs = env.reset()
    print(f"Created {num_envs} parallel environments")
    
    if is_vecenv:
        print(f"Observation structure (first environment): ")
        for key, value in obs.items():
            print(f"  {key}: shape={value.shape}")
    else:
        print("Initial state:", obs)
    
    # Create encoder parameters from environment
    encoder_params = create_encoder_params(env, num_envs)
    
    # Create IdentityEncoder
    encoder = IdentityEncoder(encoder_params)
    print(f"Encoder keys: {encoder.keys}")
    print(f"Encoder feature dimension: {encoder.feature_dim}")
    
    # Process observation through encoder
    encoded_obs = encoder(obs)
    print("Encoded observation sample:")
    if isinstance(encoded_obs, list):
        for i, tensor in enumerate(encoded_obs[:3]):  # Show first 3 tensors
            print(f"Tensor {i} shape: {tensor.shape}")
    else:
        print(f"Shape: {encoded_obs.shape}")
    
    # Initialize tracking variables
    episode_rewards = np.zeros(num_envs) if is_vecenv else 0
    episode_count = 0
    total_episodes = 10
    
    # Take some actions across all environments
    num_steps = 100
    for step in range(num_steps):
        # Sample random actions for all environments
        if is_vecenv:
            actions = np.random.randint(0, env.action_dim, size=num_envs)
        else:
            actions = np.random.randint(0, env.num_actions)
        
        # Print the action description (for the first environment if vectorized)
        if is_vecenv:
            obj_id = actions[0] // 5
            direction = actions[0] % 5
        else:
            obj_id = actions // 5
            direction = actions % 5
            
        direction_names = ["Stay", "Left", "Up", "Right", "Down"]
        print(f"\nStep {step+1}/{num_steps}: Moving object {obj_id} {direction_names[direction]} (first env)")
        
        # Take the action in all environments
        next_obs, rewards, dones, infos = env.step(actions)
        
        # Update episode rewards
        episode_rewards += rewards
        
        # Process observations through encoder
        encoded_next_obs = encoder(next_obs)
        
        # For vectorized environments, check which environments are done
        if is_vecenv:
            for i, done in enumerate(dones):
                if done:
                    print(f"Environment {i} completed episode with reward {episode_rewards[i]}")
                    episode_rewards[i] = 0
                    episode_count += 1
                    
            # Print some basic info about the step
            print(f"Rewards: min={rewards.min():.2f}, mean={rewards.mean():.2f}, max={rewards.max():.2f}")
            print(f"Completed episodes so far: {episode_count}")
            
            # Exit if we've completed enough episodes
            if episode_count >= total_episodes:
                print(f"Reached {total_episodes} completed episodes")
                break
        else:
            # Single environment handling
            print(f"Reward: {rewards}, Done: {dones}")
            
            if dones:
                print(f"Episode finished with reward {episode_rewards}")
                episode_rewards = 0
                episode_count += 1
                obs = env.reset()
                
                if episode_count >= total_episodes:
                    print(f"Reached {total_episodes} completed episodes")
                    break
        
        # Update observation
        obs = next_obs
        
        # Small delay to not overwhelm console output
        time.sleep(0.1)
    
    # Clean up
    env.close()
    print("Simulation complete!")


if __name__ == "__main__":
    main()
