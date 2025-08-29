#!/usr/bin/env python3
"""
Main training script for the Efficient Lunar Lander RL agent using PPO.

This script:
1. Creates the custom EfficientLanderEnv environment
2. Sets up PPO training with PyTorch backend
3. Configures TensorBoard logging
4. Trains the agent with evaluation callbacks
5. Saves the final model
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
    """Main training function."""

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

    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=10000,  # Evaluate every 10k steps
        deterministic=True,
        render=False,
    )

    # Configure logger for TensorBoard
    configure(folder="logs/")

    print("Setting up PPO model...")

    # Create PPO model with custom configuration
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="logs/",
        verbose=1,
    )
    


    print(f"Starting training on device: {device}")
    print("Training for 200,000 timesteps...")
    print("Check TensorBoard logs in 'logs/' directory")

    # Train the model
    model.learn(total_timesteps=200000, callback=eval_callback, progress_bar=True)

    # Save the final trained model
    final_model_path = "models/ppo_efficient_lander.zip"
    model.save(final_model_path)
    print(f"Training completed! Final model saved to: {final_model_path}")
    print(f"Best model during training saved to: models/best_model.zip")

    # Close environments
    vec_env.close()
    eval_vec_env.close()

    print("Training script completed successfully!")


if __name__ == "__main__":
    main()
