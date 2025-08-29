#!/usr/bin/env python3
"""
Precision-Focused Training Script for the Lunar Lander RL agent using PPO.

This script is specifically designed for the new precision-focused reward function that:
1. STRONGLY rewards landing in the center of the objective
2. Penalizes movement after landing to prevent sliding
3. Rewards vertical alignment with the center objective
4. Encourages fuel efficiency and precise control

Key features:
- Higher learning rate for faster adaptation to precision requirements
- Smaller batch sizes for more frequent updates
- Focused on center landing and movement prevention
- Comprehensive monitoring of precision metrics
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
    """Main training function with precision-focused parameters."""
    
    # Precision-Focused Training Configuration
    TRAINING_CONFIG = {
        "total_timesteps": 400000,      # Extended training for precision mastery
        "learning_rate": 7e-4,          # Higher learning rate for precision learning
        "n_steps": 1024,                # Smaller batch for more frequent updates
        "batch_size": 64,               # Standard batch size
        "n_epochs": 8,                  # Fewer epochs for more frequent updates
        "gamma": 0.995,                 # Slightly higher discount for long-term planning
        "gae_lambda": 0.98,             # Better GAE estimation
        "clip_range": 0.2,              # Standard PPO clipping
        "clip_range_vf": None,          # No value function clipping
        "normalize_advantage": True,     # Normalize advantages
        "ent_coef": 0.012,              # Higher entropy for better exploration
        "vf_coef": 0.5,                 # Standard value function coefficient
        "max_grad_norm": 0.5,           # Standard gradient clipping
        "use_sde": False,                # No state-dependent exploration
        "sde_sample_freq": -1,
        "target_kl": 0.02,              # Early stopping if KL divergence is too high
        "eval_freq": 5000,              # Frequent evaluation
    }
    
    print("🎯 Precision-Focused PPO Training:")
    print("=" * 70)
    print("🎯 New Precision-Focused Reward Features:")
    print("  • STRONG rewards for landing in the center (500+ bonus)")
    print("  • Penalizes movement after landing (prevents sliding)")
    print("  • Rewards vertical alignment with center objective")
    print("  • More precise landing zone requirements")
    print("  • Fuel efficiency tracking and rewards")
    print("  • Movement prevention after touchdown")
    print("=" * 70)
    print("⚙️  Training Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Setup device
    device = setup_device()
    
    # Create environment
    print("Creating precision-focused Lunar Lander environment...")
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
        n_eval_episodes=10,  # More episodes for better statistics
    )
    
    # Configure logger for TensorBoard
    configure(folder="logs/")
    
    print("Setting up precision-focused PPO model...")
    
    # Create PPO model with precision-focused configuration
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
    
    print(f"Starting precision-focused training on device: {device}")
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
    final_model_path = "models/ppo_efficient_lander_precision.zip"
    model.save(final_model_path)
    print(f"Precision-focused training completed! Final model saved to: {final_model_path}")
    print(f"Best model during training saved to: models/best_model.zip")
    
    # Close environments
    vec_env.close()
    eval_vec_env.close()
    
    print("Precision-focused training script completed successfully!")
    print("\n🎯 Expected Improvements with Precision-Focused Reward Function:")
    print("  • Much higher precision landing in the center")
    print("  • No more sliding/movement after landing")
    print("  • Better vertical alignment during approach")
    print("  • Consistent center landing performance")
    print("  • Improved fuel efficiency through precision")
    print("  • Movement prevention after touchdown")


if __name__ == "__main__":
    main()
