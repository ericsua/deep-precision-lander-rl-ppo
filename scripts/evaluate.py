#!/usr/bin/env python3
"""
Evaluation script for the trained Efficient Lunar Lander RL agent.

This script:
1. Loads a trained PPO model
2. Creates the custom environment with human rendering
3. Runs 10 evaluation episodes
4. Displays the total reward for each episode
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def create_eval_env():
    """
    Create the custom environment for evaluation with human rendering.

    Returns:
        EfficientLanderEnv: The wrapped environment with human rendering
    """
    # Create the base environment with human rendering
    base_env = gym.make("LunarLander-v3", render_mode="human")

    # Wrap it with our custom EfficientLanderEnv
    custom_env = EfficientLanderEnv(base_env)

    return custom_env


def evaluate_model(model_path, num_episodes=10):
    """
    Evaluate a trained model by running multiple episodes.

    Args:
        model_path (str): Path to the saved model file
        num_episodes (int): Number of episodes to run for evaluation
    """

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Available models in 'models/' directory:")
        if os.path.exists("models/"):
            for file in os.listdir("models/"):
                if file.endswith(".zip"):
                    print(f"  - models/{file}")
        return

    print(f"Loading model from: {model_path}")

    # Load the trained model
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create evaluation environment
    print("Creating evaluation environment...")
    env = create_eval_env()

    # Run evaluation episodes
    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("Watch the agent's performance in the rendered window!")
    print("=" * 50)

    episode_rewards = []

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        # Reset environment
        obs, info = env.reset()
        total_reward = 0
        step_count = 0

        # Run episode
        while True:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Check if episode is done
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step_count}")

        # Small pause between episodes
        import time

        time.sleep(1)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Number of episodes: {num_episodes}")
    print(f"Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Worst episode reward: {min(episode_rewards):.2f}")

    # Close environment
    env.close()

    print("\nEvaluation completed!")


def main():
    """Main evaluation function."""

    print("Efficient Lunar Lander - Model Evaluation")
    print("=" * 50)

    # Check for available models
    models_dir = "models/"
    best_model_path = os.path.join(models_dir, "best_model.zip")
    final_model_path = os.path.join(models_dir, "ppo_efficient_lander.zip")

    # Try to use best_model.zip first (from EvalCallback), then fall back to final model
    if os.path.exists(best_model_path):
        print("Found best model from training (best_model.zip)")
        model_path = best_model_path
    elif os.path.exists(final_model_path):
        print("Found final trained model (ppo_efficient_lander.zip)")
        model_path = final_model_path
    else:
        print("No trained models found!")
        print("Please run train.py first to train a model.")
        return

    # Run evaluation
    evaluate_model(model_path, num_episodes=10)


if __name__ == "__main__":
    main()
