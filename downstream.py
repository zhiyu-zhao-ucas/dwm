import logging
import os
import time

import numpy as np
import torch

import wandb
from tqdm import tqdm
from loguru import logger
from copy import deepcopy

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

from fcdl.env.chemical_env import Chemical
from fcdl.model.encoder import make_encoder
from fcdl.model.inference_gnn import InferenceGNN
from fcdl.model.inference_mlp import InferenceMLP
from fcdl.model.inference_ncd import InferenceNCD
from fcdl.model.inference_ours_masking import InferenceOursMask
from fcdl.model.inference_dwm import InferenceDWM
from fcdl.model.model_based import ModelBased, ModelBasedEntropy
from fcdl.model.random_policy import RandomPolicy
from fcdl.utils.replay_buffer import ReplayBuffer
from fcdl.utils.utils import (TrainingParams, get_env, get_single_env,
                             get_start_step_from_model_loading,
                             override_params_from_cli_args,
                             set_seed_everywhere, update_obs_act_spec)
from fcdl.utils.utils import postprocess_obs, calculate_entropy



def ood_evaluation(params, inference, obs_batch, actions_batch, next_obses_batch, info_batch, step):
    inference.eval()
    with torch.no_grad():
        if params.env_params.env_name == 'Chemical':
            ood_evaluation_chemical(params, inference, obs_batch, actions_batch, next_obses_batch, info_batch, step)
        else:
            pass

def ood_evaluation_chemical(params, inference, obs_batch, actions_batch, next_obses_batch, info_batch, step):
    test_params = params.env_params.chemical_env_params.test_params
    test_scales = [100,]
    inference.encoder.chemical_train = False
    test_detail = {}
    # "test1": number of noisy nodes = 2
    # "test2": number of noisy nodes = 4
    # "test3": number of noisy nodes = 6
    for i, test_param in enumerate(test_params):
        test_env_name = test_param.name
        inference.encoder.chemical_test_level = i
        for test_scale in test_scales:
            inference.encoder.chemical_test_scale = test_scale
            ood_eval_detail = inference.ood_prediction(obs_batch, actions_batch, next_obses_batch, info_batch)
            wandb_name = f"test/{test_env_name}/inference"
            for k, v in ood_eval_detail.items():
                # Convert numpy types to Python native types for JSON serialization
                if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                    v = float(v)
                test_detail[f"{wandb_name}/{k}"] = v
    
    # Save to file
    import os
    import json
    
    # Create ood_data directory if it doesn't exist
    if params.training_params.zero_shot:
        os.makedirs("downstream/zero_shot/ood_eval", exist_ok=True)
    else:
        os.makedirs("downstream/ood_eval", exist_ok=True)
    
    algo = params.training_params.inference_algo
    seed = params.seed
    if params.training_params.zero_shot:
        filename = f"downstream/zero_shot/ood_eval/{algo}-{seed}.json"
    else:
        filename = f"downstream/ood_eval/{algo}-{seed}.json"
    
    # Create a record with step information
    record = {
        "step": step,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    record.update(test_detail)
    
    # Load existing data if file exists
    existing_data = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except:
            pass
    
    existing_data.append(record)
    
    # Save to file using JSON format
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=2)
    
    if not getattr(params, "mute_wandb", False):
        wandb.log(test_detail, step+1)
    inference.encoder.chemical_train = True

def test_policy_evaluation(params, inference, policy, step):
    inference.eval()
    policy.eval()
    with torch.no_grad():
        if params.env_params.env_name == 'Chemical':
            test_policy_evaluation_chemical(params, inference, policy, step)
        else:
            pass

def test_policy_evaluation_chemical(params, inference, policy, step):
    test_params = params.env_params.chemical_env_params.test_params
    # "test1": number of noisy nodes = 2
    # "test2": number of noisy nodes = 4
    # "test3": number of noisy nodes = 6
    test_scales = [100,]
    num_env = params.env_params.num_env
    is_vecenv = num_env > 1
    training_params = params.training_params
    inference.encoder.chemical_train = False
    test_detail = {}
    for i, test_param in enumerate(test_params):
        test_env_name = test_param.name
        logging.info("Testing %s", test_env_name)
        env = get_env(params, test_idx=i)
        inference.encoder.chemical_test_level = i
        for test_scale in test_scales:
            episode_num = 0
            episode_reward = np.zeros(num_env) if is_vecenv else 0
            episode_reward_mean = []
            success = np.zeros(num_env, dtype=bool) if is_vecenv else False
            success_hist = []
            test_global_step = 0
            inference.encoder.chemical_test_scale = test_scale
            wandb_name = f"test/{test_env_name}/policy"
            if is_vecenv:
                obs = env.reset()
                if training_params.zero_shot:
                    total_episodes = training_params.zero_shot_total_test_episode_num
                else:
                    total_episodes = training_params.total_test_episode_num
                pbar = tqdm(total=total_episodes, desc=f"Testing {test_env_name}_s{test_scale}")
                while episode_num < total_episodes:
                    action = policy.act(obs, deterministic=True)
                    next_obs, env_reward, done, info = env.step(action)
                    episode_reward += env_reward
                    success = success | np.stack([info[i]["success"] for i in range(num_env)])
                    obs = next_obs
                    if done.any():
                        for i in range(num_env):
                            if not done[i]:
                                continue
                            success_hist.append(success[i])
                            episode_reward_mean.append(episode_reward[i])
                            episode_reward[i] = 0
                            success[i] = False
                            episode_num += 1
                            pbar.update(1)
                    test_global_step += 1
                    pbar.set_postfix({"reward": np.mean(episode_reward_mean), "success": np.mean(success_hist) if success_hist else 0})
                pbar.close()
            else:
                obs = env.reset()
                total_episodes = training_params.total_test_episode_num
                pbar = tqdm(total=total_episodes, desc=f"Testing {test_env_name}_s{test_scale}")
                while episode_num < total_episodes:
                    action = policy.act(obs, deterministic=True)
                    next_obs, env_reward, done, info = env.step(action)
                    episode_reward += env_reward
                    success = success or info["success"]
                    obs = next_obs
                    if done:
                        success_hist.append(success)
                        episode_reward_mean.append(episode_reward)
                        episode_reward = 0
                        success = False
                        episode_num += 1
                        obs = env.reset()
                        pbar.update(1)
                        pbar.set_postfix({"reward": np.mean(episode_reward_mean), "success": np.mean(success_hist)})
                    test_global_step += 1
                pbar.close()
            episode_reward_mean = np.mean(episode_reward_mean, axis=0)
            success_ratio = np.mean(success_hist, axis=0)
            test_detail[f"{wandb_name}/episode_reward_mean"] = episode_reward_mean
            test_detail[f"{wandb_name}/success_ratio"] = success_ratio
        if is_vecenv:
            env.close()
            
    # Save to file
    import os
    import json
    
    # Create directory if it doesn't exist
    zero_shot = getattr(params.training_params, "zero_shot", False)
    if zero_shot:
        os.makedirs("downstream/zero_shot/reward", exist_ok=True)
    else:
        os.makedirs("downstream/reward", exist_ok=True)
    
    algo = params.training_params.inference_algo
    seed = params.seed
    load_inference = getattr(params.training_params, "load_inference", "none")
    if zero_shot:
        filename = f"downstream/zero_shot/reward/{algo}-{seed}-zero_shot.json"
    else:
        filename = f"downstream/reward/{algo}-{seed}-.json"
    
    # Create a record with step information
    record = {
        "step": step,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    record.update(test_detail)
    
    # Load existing data if file exists
    existing_data = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except:
            pass
    
    existing_data.append(record)
    
    # Save to file using JSON format
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=2)
    
    if not getattr(params, "mute_wandb", False):
        wandb.log(test_detail, step+1)
    inference.encoder.chemical_train = True

def train(params):
    device = torch.device("cuda:{}".format(params.cuda_id) if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available()
    if not params.seed == -1:
        set_seed_everywhere(params.seed)
    params.device = device
    env_name = params.env_params.env_name
    loglevel = getattr(params, "loglevel", "WARNING")
    n_samples = 0
    logging.basicConfig(level=loglevel)
    
    # Wandb
    env_specific_type = "default"
    if env_name == "Chemical":
        params.obs_keys, params.goal_keys = [], []
        for i in range(params.env_params.chemical_env_params.num_objects):
            params.obs_keys.append(f"obj{i}")
            params.goal_keys.append(f"target_obj{i}")

        env_specific_type = params.env_params.chemical_env_params.local_causal_rule
    
    # Check if wandb should be muted
    params.mute_wandb = getattr(params.training_params, "mute_wandb", False)
    
    if not params.mute_wandb:
        # wandb.init(project=f'{env_name}-{env_specific_type}',
        #         name=f'{params.training_params.inference_algo}-{time.strftime("%m%d_%H-%M-%S")}',
        #         config=dict(params),
        #         dir=params.wandb_dir)
        wandb.init(project=f'{env_name}-{env_specific_type}',
                name=f'{params.training_params.inference_algo}-{params.seed}-downstream-{params.training_params.load_inference}',
                config=dict(params),
                dir=params.wandb_dir)
    else:
        # Create a dummy run ID for directory naming when wandb is muted
        class DummyRun:
            def __init__(self):
                self.id = f"muted-{params.training_params.inference_algo}-{params.seed}-downstream-{params.training_params.load_inference}-{time.strftime('%m%d_%H-%M-%S')}"
    if not params.mute_wandb:
        params.rslts_dir = params.wandb_dir + f"{env_name}/{wandb.run.id}/"
    else:
        params.rslts_dir = os.path.join(params.wandb_dir, f"{env_name}/{DummyRun().id}/")
    os.makedirs(params.rslts_dir)
    torch.save(dict(params), os.path.join(params.rslts_dir, "params"))
    load_dir = None
    if getattr(params.training_params, "load_id", None) is not None:
        load_dir = params.wandb_dir + f"{env_name}/{params.training_params.load_id}/"
        load_params = torch.load(os.path.join(load_dir, "params"))
        assert params.training_params.inference_algo == load_params["training_params"]["inference_algo"]
        if getattr(params.training_params, "load_inference", None) is not None:
            params.training_params.load_inference = \
                os.path.join(load_dir, "trained_models", f"inference_{params.training_params.load_inference}")
        if getattr(params.training_params, "load_model_based", None) is not None:
            raise NotImplementedError
        if getattr(params.training_params, "load_policy", None) is not None:
            raise NotImplementedError
        if getattr(params.training_params, "load_replay_buffer", None) is not None:
            raise NotImplementedError

    # init environment
    num_env = params.env_params.num_env
    is_vecenv = num_env > 1
    if is_vecenv and env_name == "Chemical":
        single_env = get_single_env(params)
        single_env.save_mlps()
        del single_env
    env = get_env(params, load_dir=load_dir)
    if isinstance(env, Chemical):
        torch.save(env.get_save_information(), os.path.join(params.rslts_dir, "chemical_env_params"))
    # init model
    update_obs_act_spec(env, params)
    encoder = make_encoder(params)

    inference_algo = params.training_params.inference_algo

    logger.info(f"params.mute_wandb: {params.mute_wandb}")
    logger.info(f"Using inference algorithm: {inference_algo}")
    if inference_algo == "mlp":
        Inference = InferenceMLP
    elif inference_algo == "gnn":
        Inference = InferenceGNN
    elif inference_algo == "ncd":
        Inference = InferenceNCD
    elif inference_algo == "oracle":
        assert params.inference_params.use_gt_global_mask
        assert not params.ours_params.code_labeling
        Inference = InferenceOursMask
    elif "ours" in inference_algo:
        Inference = InferenceOursMask
    elif inference_algo == "dwm":
        Inference = InferenceDWM
    elif "dwm" in inference_algo:
        Inference = InferenceDWM
    else:
        raise NotImplementedError
    if inference_algo == "ncd":
        model_path = params.training_params.load_inference
        params_path = model_path.replace('/'.join(model_path.split("/")[-2:]), "params")
        logger.info(f"Loading params from {params_path}")
        params_dict = torch.load(params_path)
        original_params = deepcopy(params)
        for key, value in params_dict.items():
            setattr(params, key, value)
        params.cuda_id = original_params.cuda_id
        params.device = original_params.device
        logger.info(f"params.cuda_id: {params.cuda_id}, original_params.cuda_id: {original_params.cuda_id}")
        # params.training_params.load_inference = ""
        # logger.info(f"params.training_params.load_inference: {params.training_params.load_inference}")
        inference = Inference(encoder, params)
        inference.load(model_path, device)
        params = original_params
        logger.info(f"params.mute_wandb: {params.mute_wandb}")
    else:
        inference = Inference(encoder, params)

    if not params.mute_wandb:
        wandb.watch(inference, log="gradients", log_freq=100)

    rl_algo = params.training_params.rl_algo
    is_task_learning = rl_algo == "model_based"
    init_policy = RandomPolicy(params)
    if rl_algo == "random":
        policy = RandomPolicy(params)
    elif rl_algo == "model_based":
        policy = ModelBased(encoder, inference, params)
    else:
        raise NotImplementedError


    training_params = params.training_params
    inference_params = params.inference_params
    policy_params = params.policy_params

    replay_buffer = ReplayBuffer(params)

    # init saving
    model_dir = os.path.join(params.rslts_dir, "trained_models")
    os.makedirs(model_dir, exist_ok=True)
    buffer_dir = os.path.join(params.rslts_dir, "replay_buffer")
    os.makedirs(buffer_dir, exist_ok=True)

    start_step = get_start_step_from_model_loading(params)
    total_steps = training_params.total_steps
    inference_gradient_steps = training_params.inference_gradient_steps
    train_prop = inference_params.train_prop

    # init episode variables
    episode_num = 0
    obs = env.reset()

    done = np.zeros(num_env, dtype=bool) if is_vecenv else False
    success = np.zeros(num_env, dtype=bool) if is_vecenv else False
    episode_reward = np.zeros(num_env) if is_vecenv else 0
    episode_step = np.zeros(num_env) if is_vecenv else 0
    is_train = (np.random.rand(num_env) if is_vecenv else np.random.rand()) < train_prop
    
    if training_params.zero_shot:
        params.stage = 'test'
        test_policy_evaluation(params, inference, policy, 0)
    else:
        for step in tqdm(range(start_step, total_steps)):
            params.step = step
            is_init_stage = step < training_params.init_steps
            # print(f"{step + 1}/{total_steps}, is_init_stage: {is_init_stage}")
            if not params.mute_wandb:
                wandb.log({"init_stage": float(is_init_stage)}, step+1)
            loss_details = {"inference": [],
                            "inference_eval": [],
                            "inference_eval_rl": [],
                            "policy": []}
            train_loss_detail = {"pred_loss": 0}
            eval_loss_detail = {"pred_loss": 0}

            # env interaction and transition saving
            params.stage = 'train'
            # reset in the beginning of an episode
            if is_vecenv and done.any():
            
                success = success | np.stack([info[i]["success"] for i in range(num_env)])
                if not params.mute_wandb:
                    wandb.log({
                        "policy_stat/episode_reward": episode_reward.mean(),
                        "episode_num": episode_num,
                        "policy_stat/success": float(success.mean())
                    }, step+1)
                for i, done_ in enumerate(done):
                    if not done_:
                        continue
                    is_train[i] = np.random.rand() < train_prop

                    episode_reward[i] = 0
                    episode_step[i] = 0
                    success[i] = False
                    episode_num += 1
            elif not is_vecenv and done:
                obs = env.reset()

                if not params.mute_wandb:
                    if is_task_learning:
                        wandb.log({
                            "policy_stat/episode_reward": episode_reward,
                            "policy_stat/success": float(success),
                            "episode_num": episode_num,
                        }, step+1)
                    else:
                        wandb.log({
                                "policy_stat/episode_reward": episode_reward,
                                "episode_num": episode_num,
                        }, step+1)
                is_train = np.random.rand() < train_prop
                episode_reward = 0
                episode_step = 0
                success = False
                episode_num += 1

            # get action
            inference.eval()
            policy.eval()
            if is_init_stage:
                action = init_policy.act(obs)
            else:
                action = policy.act(obs)
            next_obs, env_reward, done, info = env.step(action)
            n_samples += num_env
            if not params.mute_wandb:
                wandb.log({"n_samples": n_samples}, step+1)
            
            if is_task_learning and not is_vecenv:
                success = success or info["success"]

            inference_reward = np.zeros(num_env) if is_vecenv else 0
            episode_reward += env_reward if is_task_learning else inference_reward
            episode_step += 1

            replay_buffer.add(obs, action, env_reward, next_obs, done, info, is_train)

            obs = next_obs


            # training and logging
            if not is_init_stage:

                if inference_gradient_steps > 0 and (step + 1) % training_params.inference_update_freq == 0:
                    inference.train()
                    for i_grad_step in range(inference_gradient_steps):
                        obs_batch, actions_batch, next_obses_batch, _ = \
                            replay_buffer.sample_inference(inference_params.batch_size, "train")
                        # logger.info(f"start training inference, step: {step}, i_grad_step: {i_grad_step}, inference: {inference}")
                        train_loss_detail = inference.update(obs_batch, actions_batch, next_obses_batch)
                        loss_details["inference"].append(train_loss_detail)

                    inference.eval()
                    if (step + 1) % inference_params.eval_freq == 0:
                        obs_batch, actions_batch, next_obses_batch, info_batch = \
                            replay_buffer.sample_inference(inference_params.eval_batch_size, use_part="eval")
                        eval_loss_detail = inference.eval_prediction(obs_batch, actions_batch, next_obses_batch, info_batch)
                        loss_details["inference_eval"].append(eval_loss_detail)
                    
                    if (step + 1) % training_params.ood_eval_freq == 0:
                        obs_batch, actions_batch, next_obses_batch, info_batch = \
                            replay_buffer.sample_ood_eval(training_params.ood_eval_batch_size, use_part="all")
                        ood_evaluation(params, inference, obs_batch, actions_batch, next_obses_batch, info_batch, step)
                
                for module_name, module_loss_detail in loss_details.items():
                    if not module_loss_detail:
                        continue
                    # list of dict to dict of list
                    if isinstance(module_loss_detail, list):
                        keys = set().union(*[dic.keys() for dic in module_loss_detail])
                        module_loss_detail = {k: [dic[k].item() for dic in module_loss_detail if k in dic]
                                            for k in keys if k not in ["priority"]}
                    for loss_name, loss_values in module_loss_detail.items():
                        if not params.mute_wandb:
                            wandb.log({
                                        "{}/{}".format(module_name, loss_name): np.mean(loss_values),
                            }, step+1)

                if (step + 1) % training_params.saving_freq == 0:
                    easy_step = int((step + 1) / 1000)
                    if inference_gradient_steps > 0:
                        inference.save(os.path.join(model_dir, f"inference_{easy_step}k"))
                        # inference.save(os.path.join(model_dir, "inference_{}".format(step + 1)))
                    replay_buffer.save(buffer_dir, easy_step)
            
            if (step + 1) % training_params.test_freq == 0:
                params.stage = 'test'
                test_policy_evaluation(params, inference, policy, step)
                params.stage = 'train'

if __name__ == "__main__":
    params = TrainingParams(training_params_fname="policy_params.json", train=True)
    override_params_from_cli_args(params)
    train(params)
