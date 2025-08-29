#!/usr/bin/env python3
"""
Enhanced training script for the Efficient Lunar Lander RL agent using PPO.

This script includes:
1. Better hyperparameters for improved learning
2. Longer training duration
3. More frequent evaluation
4. Customizable training parameters
5. Better logging and monitoring
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
    # MPS support in stable-baselines3 can be problematic
    print("Using CPU for stable-baselines3 compatibility")
    return "cpu"


def main():
    """Main training function with enhanced parameters."""
    
    # Training Configuration - Adjust these parameters as needed
    TRAINING_CONFIG = {
        "total_timesteps": 500000,      # Increased from 200k to 500k
        "learning_rate": 1e-4,          # Reduced for more stable learning
        "n_steps": 4096,                # Increased batch size for better stability
        "batch_size": 128,              # Increased batch size
        "n_epochs": 15,                 # More epochs per update
        "gamma": 0.995,                 # Slightly higher discount factor
        "gae_lambda": 0.98,             # Better GAE estimation
        "clip_range": 0.15,             # Smaller clip range for stability
        "clip_range_vf": 0.15,          # Value function clipping
        "normalize_advantage": True,     # Normalize advantages
        "ent_coef": 0.005,              # Reduced entropy for more focused learning
        "vf_coef": 0.25,                # Reduced value function coefficient
        "max_grad_norm": 0.3,           # Smaller gradient clipping
        "use_sde": False,                # No state-dependent exploration
        "sde_sample_freq": -1,
        "target_kl": 0.01,              # Early stopping if KL divergence is too high
        "eval_freq": 5000,              # Evaluate more frequently
    }
    
    print("🚀 Enhanced PPO Training Configuration:")
    print("=" * 50)
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Setup device
    device = setup_device()
    
    # Create environment
    print("Creating custom Lunar Lander environment...")
    env = create_env()
    
    # Wrap in DummyVecEnv for stable-baselines3 compatibility
    vec_env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment (separate from training env)
    eval_env = create_env()
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    
    # Setup evaluation callback with more frequent evaluation
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=TRAINING_CONFIG["eval_freq"],  # More frequent evaluation
        deterministic=True,
        render=False,
        n_eval_episodes=5,  # Evaluate on 5 episodes for better statistics
    )
    
    # Configure logger for TensorBoard
    configure(folder="logs/")
    
    print("Setting up enhanced PPO model...")
    
    # Create PPO model with enhanced configuration
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
    
    print(f"Starting enhanced training on device: {device}")
    print(f"Training for {TRAINING_CONFIG['total_timesteps']:,} timesteps...")
    print("Check TensorBoard logs in 'logs/' directory")
    print("Evaluation every", TRAINING_CONFIG["eval_freq"], "timesteps")
    
    # Train the model
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"], 
        callback=eval_callback, 
        progress_bar=True
    )
    
    # Save the final trained model
    final_model_path = "models/ppo_efficient_lander_enhanced.zip"
    model.save(final_model_path)
    print(f"Enhanced training completed! Final model saved to: {final_model_path}")
    print(f"Best model during training saved to: models/best_model.zip")
    
    # Close environments
    vec_env.close()
    eval_vec_env.close()
    
    print("Enhanced training script completed successfully!")
    print("\n💡 Tips for further improvement:")
    print("  - Try different learning rates: 5e-5 to 5e-4")
    print("  - Experiment with batch sizes: 64, 128, 256")
    print("  - Adjust entropy coefficient: 0.001 to 0.01")
    print("  - Modify clip range: 0.1 to 0.3")
    print("  - Increase total timesteps for even better performance")


if __name__ == "__main__":
    main()
