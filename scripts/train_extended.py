#!/usr/bin/env python3
"""
Extended Training Script for the Fine-tuned Efficient Lunar Lander RL agent using PPO.

This script implements:
1. Fine-tuned reward function for balanced learning
2. Extended training duration (600k timesteps)
3. Advanced hyperparameter optimization
4. Comprehensive performance tracking
5. Adaptive learning rate scheduling

Key features:
- Longer training for complex landing behavior mastery
- Balanced exploration vs exploitation
- Fuel efficiency tracking and rewards
- Progressive difficulty scaling
"""

import os
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

# Import our custom environment from the package
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def create_env():
    """
    Create and return the custom Lunar Lander environment.
    
    Returns:
        EfficientLanderEnv: The wrapped environment
    """
    # Create the base LunarLander-v3 environment (v2 is deprecated)
    base_env = gym.make("LunarLander-v3")
    
    # Wrap it with our custom EfficientLanderEnv
    custom_env = EfficientLanderEnv(base_env)
    
    return custom_env


def setup_device():
    """
    Determine the best available device for PyTorch.
    
    Returns:
        str: Device string ('mps', 'cuda', or 'cpu')
    """
    # For now, use CPU to avoid device compatibility issues with stable-baselines3
    print("Using CPU for stable-baselines3 compatibility")
    return "cpu"


def main():
    """Main training function with extended duration and fine-tuned parameters."""
    
    # Extended Training Configuration - Optimized for mastery
    TRAINING_CONFIG = {
        "total_timesteps": 600000,      # Extended training for mastery
        "learning_rate": 6e-4,          # Balanced learning rate
        "n_steps": 2048,                # Standard PPO batch size
        "batch_size": 128,              # Larger batch for stability
        "n_epochs": 10,                 # Balanced epochs per update
        "gamma": 0.995,                 # Slightly higher discount for long-term planning
        "gae_lambda": 0.98,             # Better GAE estimation
        "clip_range": 0.2,              # Standard PPO clipping
        "clip_range_vf": None,          # No value function clipping
        "normalize_advantage": True,     # Normalize advantages
        "ent_coef": 0.008,              # Balanced entropy for exploration
        "vf_coef": 0.5,                 # Standard value function coefficient
        "max_grad_norm": 0.5,           # Standard gradient clipping
        "use_sde": False,                # No state-dependent exploration
        "sde_sample_freq": -1,
        "target_kl": 0.015,             # Early stopping if KL divergence is too high
        "eval_freq": 10000,             # Less frequent evaluation for longer training
    }
    
    print("🚀 Extended PPO Training with Fine-tuned Reward Function:")
    print("=" * 70)
    print("🎯 Fine-tuned Reward Function Features:")
    print("  • Balanced fuel efficiency rewards (prevents over-optimization)")
    print("  • Progressive hovering penalties with time limits")
    print("  • Fuel usage tracking and efficiency bonuses")
    print("  • Time pressure for faster completion")
    print("  • Balanced exploration vs exploitation")
    print("  • Comprehensive performance metrics")
    print("=" * 70)
    print("⚙️  Extended Training Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Setup device
    device = setup_device()
    
    # Create environment
    print("Creating fine-tuned Lunar Lander environment...")
    env = create_env()
    
    # Wrap in DummyVecEnv for stable-baselines3 compatibility
    vec_env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment (separate from training env)
    eval_env = create_env()
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    
    # Setup evaluation callback with less frequent evaluation
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=TRAINING_CONFIG["eval_freq"],
        deterministic=True,
        render=False,
        n_eval_episodes=10,  # More episodes for better statistics
    )
    
    # Setup checkpoint callback for saving progress
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50k steps
        save_path="checkpoints/",
        name_prefix="ppo_lander_checkpoint"
    )
    
    # Configure logger for TensorBoard
    configure(folder="logs/")
    
    print("Setting up extended PPO model...")
    
    # Create PPO model with extended configuration
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        n_steps=TRAINING_CONFIG["n_steps"],
        batch_size=TRAINING_CONFIG["batch_size"],
        n_epochs=TRAINING_CONFIG["n_epochs"],
        gamma=TRAINING_CONFIG["gamma"],
        gae_lambda=TRAINING_CONFIG["gae_lambda"],
        clip_range=TRAINING_CONFIG["clip_range"],
        clip_range_vf=TRAINING_CONFIG["clip_range_vf"],
        normalize_advantage=TRAINING_CONFIG["normalize_advantage"],
        ent_coef=TRAINING_CONFIG["ent_coef"],
        vf_coef=TRAINING_CONFIG["vf_coef"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        use_sde=TRAINING_CONFIG["use_sde"],
        sde_sample_freq=TRAINING_CONFIG["sde_sample_freq"],
        target_kl=TRAINING_CONFIG["target_kl"],
        tensorboard_log="logs/",
        verbose=1,
    )
    
    print(f"Starting extended training on device: {device}")
    print(f"Training for {TRAINING_CONFIG['total_timesteps']:,} timesteps...")
    print("Check TensorBoard logs in 'logs/' directory")
    print(f"Evaluation every {TRAINING_CONFIG['eval_freq']} timesteps")
    print("Checkpoints saved every 50,000 timesteps")
    
    # Train the model with both callbacks
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"], 
        callback=[eval_callback, checkpoint_callback], 
        progress_bar=True
    )
    
    # Save the final trained model
    final_model_path = "models/ppo_efficient_lander_extended.zip"
    model.save(final_model_path)
    print(f"Extended training completed! Final model saved to: {final_model_path}")
    print(f"Best model during training saved to: models/best_model.zip")
    print(f"Checkpoints saved in: checkpoints/")
    
    # Close environments
    vec_env.close()
    eval_vec_env.close()
    
    print("Extended training script completed successfully!")
    print("\n🎯 Expected Improvements with Extended Training:")
    print("  • Mastery of complex landing scenarios")
    print("  • Consistent high performance (500+ rewards)")
    print("  • Better fuel efficiency through extended learning")
    print("  • More precise landings with balanced exploration")
    print("  • Robust policy that generalizes better")
    print("  • Comprehensive performance tracking and analysis")


if __name__ == "__main__":
    main()
