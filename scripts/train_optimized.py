#!/usr/bin/env python3
"""
Optimized training script for the Improved Efficient Lunar Lander RL agent using PPO.

This script is specifically designed for the new balanced reward function that:
1. Provides continuous positive feedback
2. Rewards progress toward landing
3. Gives substantial bonuses for successful landings
4. Encourages fuel efficiency without being too harsh

Key optimizations:
- Higher learning rate for faster learning
- Smaller batch sizes for more frequent updates
- Better exploration parameters
- Focused on the new reward structure
"""

import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
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
    """Main training function with optimized parameters for the new reward function."""
    
    # Training Configuration - Optimized for the new balanced reward function
    TRAINING_CONFIG = {
        "total_timesteps": 300000,      # Shorter but more focused training
        "learning_rate": 5e-4,          # Higher learning rate for faster learning
        "n_steps": 2048,                # Standard PPO batch size
        "batch_size": 64,               # Standard batch size
        "n_epochs": 8,                  # Fewer epochs for more frequent updates
        "gamma": 0.99,                  # Standard discount factor
        "gae_lambda": 0.95,             # Standard GAE
        "clip_range": 0.2,              # Standard PPO clipping
        "clip_range_vf": None,          # No value function clipping
        "normalize_advantage": True,     # Normalize advantages
        "ent_coef": 0.01,               # Higher entropy for better exploration
        "vf_coef": 0.5,                 # Standard value function coefficient
        "max_grad_norm": 0.5,           # Standard gradient clipping
        "use_sde": False,                # No state-dependent exploration
        "sde_sample_freq": -1,
        "target_kl": None,              # No early stopping
        "eval_freq": 5000,              # Frequent evaluation
    }
    
    print("🚀 Optimized PPO Training for New Reward Function:")
    print("=" * 60)
    print("🎯 New Reward Function Features:")
    print("  • Continuous positive feedback for progress")
    print("  • Fuel efficiency bonuses (not penalties)")
    print("  • Substantial landing bonuses (200+ points)")
    print("  • Progress rewards for safe descent")
    print("  • Precision landing rewards")
    print("=" * 60)
    print("⚙️  Training Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Setup device
    device = setup_device()
    
    # Create environment
    print("Creating improved Lunar Lander environment...")
    env = create_env()
    
    # Wrap in DummyVecEnv for stable-baselines3 compatibility
    vec_env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment (separate from training env)
    eval_env = create_env()
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    
    # Setup evaluation callback with frequent evaluation
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=TRAINING_CONFIG["eval_freq"],
        deterministic=True,
        render=False,
        n_eval_episodes=5,  # Evaluate on 5 episodes
    )
    
    # Configure logger for TensorBoard
    configure(folder="logs/")
    
    print("Setting up optimized PPO model...")
    
    # Create PPO model with optimized configuration
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
    
    print(f"Starting optimized training on device: {device}")
    print(f"Training for {TRAINING_CONFIG['total_timesteps']:,} timesteps...")
    print("Check TensorBoard logs in 'logs/' directory")
    print(f"Evaluation every {TRAINING_CONFIG['eval_freq']} timesteps")
    
    # Train the model
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"], 
        callback=eval_callback, 
        progress_bar=True
    )
    
    # Save the final trained model
    final_model_path = "models/ppo_efficient_lander_optimized.zip"
    model.save(final_model_path)
    print(f"Optimized training completed! Final model saved to: {final_model_path}")
    print(f"Best model during training saved to: models/best_model.zip")
    
    # Close environments
    vec_env.close()
    eval_vec_env.close()
    
    print("Optimized training script completed successfully!")
    print("\n🎯 Expected Improvements with New Reward Function:")
    print("  • Much higher episode rewards (200+ for successful landings)")
    print("  • Better fuel efficiency through positive reinforcement")
    print("  • More precise landings near the center")
    print("  • Faster learning due to continuous positive feedback")
    print("  • Better exploration due to balanced rewards")


if __name__ == "__main__":
    main()
