import sys
import os
import torch
import torch.optim as optim
import numpy as np
import yaml # NEW
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
    # 1. LOAD CONFIG
    cfg = load_config()
    print(f"--- LOADED CONFIG ---")
    
    # Extract Configs for easier access
    ENV_CFG = cfg['environment']
    DIR_CFG = cfg['director']
    NAV_CFG = cfg['specialists']['navigation']
    BAT_CFG = cfg['specialists']['battle']
    TRN_CFG = cfg['training']
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- STARTING TRAINING ON {DEVICE} ---")

    # 2. SETUP
    env = PokemonRedGym(
        headless=False, 
        emulation_speed=6,
        max_steps=ENV_CFG['max_steps'] # From YAML
    )
    
    reward_sys = RewardSystem()
    
    agent = HierarchicalAgent(
        action_dim=env.action_space.n, 
        device=DEVICE,
        nav_lr=float(NAV_CFG['learning_rate']),    # From YAML
        battle_lr=float(BAT_CFG['learning_rate'])  # From YAML
    )
    agent.load(checkpoint_dir="checkpoints", tag="latest")
    
    # Director Optimizer (From YAML)
    director_optimizer = optim.AdamW(
        agent.director.parameters(), 
        lr=float(DIR_CFG['learning_rate'])
    )
    
    # Memory Buffers
    BUFFER_SIZE = TRN_CFG['buffer_size']
    nav_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, (1, 84, 84), alpha=0.2, device=DEVICE)
    battle_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, (1, 84, 84), alpha=0.6, device=DEVICE)
    director_buffer = PrioritizedReplayBuffer(10_000, (1, 84, 84), alpha=0.2, device=DEVICE)

    logger = Logger()
    
    # Loop Params
    TOTAL_STEPS = 5_000_000
    BATCH_SIZE = TRN_CFG['batch_size']
    GAMMA = TRN_CFG['gamma']
    SAVE_FREQ = 2_000
    
    # Frequencies
    FAST_FREQ = 4
    SLOW_FREQ = DIR_CFG['update_interval'] # From YAML (128)

    # Epsilon
    epsilon = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 100_000 
    
    # 3. TRAINING LOOP
    obs, info = env.reset()
    reward_sys.reset()
    
    episode_reward = 0
    pbar = tqdm(range(1, TOTAL_STEPS + 1))
    
    for step in pbar:
        # --- A. SELECT ACTION ---
        action, specialist_idx = agent.get_action(obs, info, epsilon)
        
        # --- B. EXECUTE ---
        next_obs, _, terminated, truncated, next_info = env.step(action)
        ram_reward = reward_sys.compute_reward(next_info, env.pyboy.memory, next_obs, action)
        
        done = terminated or truncated
        episode_reward += ram_reward

        # --- C. STORE ---
        if specialist_idx == 0:
            nav_buffer.add(obs, action, ram_reward, next_obs, done)
        else:
            battle_buffer.add(obs, action, ram_reward, next_obs, done)
        director_buffer.add(obs, specialist_idx, ram_reward, next_obs, done)

        # --- D. TRAIN ---
        if step > 1000 and step % FAST_FREQ == 0:
            metrics = {}
            
            # Fast Loop (Nav)
            if len(nav_buffer) > BATCH_SIZE:
                s, a, r, ns, d, w, idx = nav_buffer.sample(BATCH_SIZE)
                with torch.no_grad():
                    s_feat = agent.director.encoder(s)
                    ns_feat = agent.director.encoder(ns)
                nav_loss, nav_stats = agent.nav_brain.train_step(s_feat, a, r, ns_feat, d)
                nav_buffer.update_priorities(idx, [nav_loss + 1e-5] * BATCH_SIZE)
                metrics['loss/nav'] = nav_loss
                metrics.update(nav_stats)

            # Fast Loop (Battle)
            if len(battle_buffer) > BATCH_SIZE:
                s, a, r, ns, d, w, idx = battle_buffer.sample(BATCH_SIZE)
                with torch.no_grad():
                    s_feat = agent.director.encoder(s)
                    ns_feat = agent.director.encoder(ns)
                battle_loss = agent.battle_brain.train_step(s_feat, a, r, ns_feat, d)
                battle_buffer.update_priorities(idx, [battle_loss + 1e-5] * BATCH_SIZE)
                metrics['loss/battle'] = battle_loss

            # Slow Loop (Director) - Uses YAML update_interval
            if step % SLOW_FREQ == 0 and len(director_buffer) > BATCH_SIZE:
                s, a, r, ns, d, w, idx = director_buffer.sample(BATCH_SIZE)
                logits, _ = agent.director(s)
                q_pred = logits.gather(1, a.unsqueeze(1).long()).squeeze(1)
                with torch.no_grad():
                    next_logits, _ = agent.director(ns)
                    max_next_q = next_logits.max(1)[0]
                    q_target = r + GAMMA * max_next_q * (1 - d)
                
                director_loss = torch.nn.functional.mse_loss(q_pred, q_target)
                director_optimizer.zero_grad()
                director_loss.backward()
                director_optimizer.step()
                director_buffer.update_priorities(idx, [director_loss.item() + 1e-5] * BATCH_SIZE)
                metrics['loss/director'] = director_loss.item()

            metrics['policy/epsilon'] = epsilon
            metrics['buffer/nav_fill'] = len(nav_buffer) / BUFFER_SIZE
            metrics['buffer/battle_fill'] = len(battle_buffer) / BUFFER_SIZE
            # inside the FAST_FREQ block, where metrics is a dict
            metrics['reward/step'] = ram_reward


            if len(metrics) > 0:
                logger.log_step(metrics, step)

        # --- E. UPDATE & RESET ---
        obs = next_obs
        info = next_info
        
        if epsilon > EPSILON_END:
            epsilon -= (1.0 - EPSILON_END) / EPSILON_DECAY

        pbar.set_description(f"Rew: {episode_reward:.2f} | Brain: {'Nav' if specialist_idx==0 else 'Bat'}")

        if done:
            episode_reward = 0
            obs, info = env.reset() # This now triggers the max_steps check from YAML
            reward_sys.reset()
            
        if step % SAVE_FREQ == 0:
            agent.save("checkpoints", tag="latest")
            print("Saved Latest checkpoint at step", step)
            if step % (SAVE_FREQ * 5) == 0:
                 agent.save("checkpoints", tag=f"step_{step}")
                 print("Saved Backup checkpoint at step", step)

    logger.close()
    env.close()

if __name__ == "__main__":
    train()
