import sys
import os
import torch
import torch.optim as optim
import numpy as np
import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.pokemon_red_gym import PokemonRedGym
from src.env.rewards import RewardSystem
from src.agent.hierarchical_agent import HierarchicalAgent
from src.utils.memory_buffer import PrioritizedReplayBuffer
from src.utils.logger import Logger

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "hyperparameters.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train():
    cfg = load_config()
    ENV_CFG = cfg['environment']
    NAV_CFG = cfg['specialists']['navigation']
    BAT_CFG = cfg['specialists']['battle']
    TRN_CFG = cfg['training']
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- STARTING TRAINING ON {DEVICE} ---")

    # 1. SETUP
    env = PokemonRedGym(
        headless=False, 
        emulation_speed=6,
        max_steps=ENV_CFG['max_steps']
    )
    
    reward_sys = RewardSystem()
    
    agent = HierarchicalAgent(
        action_dim=env.action_space.n, 
        device=DEVICE,
        nav_lr=float(NAV_CFG['learning_rate']),
        battle_lr=float(BAT_CFG['learning_rate'])
    )
    # Load Weights (Agent loads Director + Specialists)
    agent.load(checkpoint_dir="checkpoints", tag="latest")
    
    # --- OPTIMIZER SETUP ---
    # The Specialists have their own optimizers for their specialized layers.
    # But who trains the Vision Encoder inside the Director?
    # Strategy: We create a separate optimizer for the Encoder, 
    # or we simply add the encoder params to the specialists' optimizers.
    # Let's use a SHARED OPTIMIZER for the encoder for stability.
    
    vision_optimizer = optim.AdamW(
        agent.director.vision.parameters(), 
        lr=1e-4
    )
    
    # Memory Buffers
    # Note: We removed the Director Buffer. Logic doesn't need replay.
    BUFFER_SIZE = TRN_CFG['buffer_size']
    nav_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, (1, 84, 84), alpha=0.2, device=DEVICE)
    battle_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, (1, 84, 84), alpha=0.6, device=DEVICE)

    logger = Logger()
    
    # Params
    TOTAL_STEPS = 5_000_000
    BATCH_SIZE = TRN_CFG['batch_size']
    SAVE_FREQ = 2_000
    FAST_FREQ = 4
    
    epsilon = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 100_000 
    
    # 2. TRAINING LOOP
    obs, info = env.reset()
    reward_sys.reset()
    
    episode_reward = 0
    pbar = tqdm(range(1, TOTAL_STEPS + 1))
    
    for step in pbar:
        # --- A. SELECT ACTION ---
        # The Director Logic runs inside get_action
        action, specialist_idx = agent.get_action(obs, info, epsilon)
        
        # --- B. EXECUTE ---
        next_obs, _, terminated, truncated, next_info = env.step(action)
        ram_reward = reward_sys.compute_reward(next_info, env.pyboy.memory, next_obs, action)
        
        done = terminated or truncated
        episode_reward += ram_reward

        # --- C. STORE ---
        # Store based on who was active
        if specialist_idx == 0:
            nav_buffer.add(obs, action, ram_reward, next_obs, done)
        elif specialist_idx == 1:
            battle_buffer.add(obs, action, ram_reward, next_obs, done)

        # --- D. TRAIN ---
        if step > 1000 and step % FAST_FREQ == 0:
            metrics = {}
            
            # --- TRAIN NAVIGATION ---
            if len(nav_buffer) > BATCH_SIZE:
                s, a, r, ns, d, w, idx = nav_buffer.sample(BATCH_SIZE)
                
                # 1. Forward Pass through Shared Encoder
                # We do this here so we can capture gradients for the encoder
                s_feat = agent.director.vision(s)
                ns_feat = agent.director.vision(ns)
                
                # 2. Specialist Loss
                # Note: You need to modify NavAgent.train_step to return LOSS only, not step optimizer
                # Or use a custom update routine here:
                
                # Get Q-values/Loss from Nav Brain
                nav_loss, stats = agent.nav_brain.train_step_return_loss(s_feat, a, r, ns_feat, d)
                
                # 3. Backprop (End-to-End)
                vision_optimizer.zero_grad()
                agent.nav_brain.optimizer.zero_grad()
                
                nav_loss.backward()
                
                # 4. Step Both Optimizers
                vision_optimizer.step()
                agent.nav_brain.optimizer.step()
                
                nav_buffer.update_priorities(idx, [nav_loss.item() + 1e-5] * BATCH_SIZE)
                metrics['loss/nav'] = nav_loss.item()

            # --- TRAIN BATTLE ---
            if len(battle_buffer) > BATCH_SIZE:
                s, a, r, ns, d, w, idx = battle_buffer.sample(BATCH_SIZE)
                
                s_feat = agent.director.vision(s)
                ns_feat = agent.director.vision(ns)
                
                battle_loss = agent.battle_brain.train_step_return_loss(s_feat, a, r, ns_feat, d)
                
                vision_optimizer.zero_grad()
                agent.battle_brain.optimizer.zero_grad()
                
                battle_loss.backward()
                
                vision_optimizer.step()
                agent.battle_brain.optimizer.step()
                
                battle_buffer.update_priorities(idx, [battle_loss.item() + 1e-5] * BATCH_SIZE)
                metrics['loss/battle'] = battle_loss.item()

            # Log Metrics
            metrics['policy/epsilon'] = epsilon
            metrics.update(agent.director.blackboard.__dict__) # Log Blackboard state
            if len(metrics) > 0:
                logger.log_step(metrics, step)

        # --- E. UPDATE & RESET ---
        obs = next_obs
        info = next_info
        
        if epsilon > EPSILON_END:
            epsilon -= (1.0 - EPSILON_END) / EPSILON_DECAY

        pbar.set_description(f"Rew: {episode_reward:.2f} | Goal: {agent.director.current_goal}")

        if done:
            episode_reward = 0
            obs, info = env.reset()
            reward_sys.reset()
            
        if step % SAVE_FREQ == 0:
            agent.save("checkpoints", tag="latest")

    logger.close()
    env.close()

if __name__ == "__main__":
    train()