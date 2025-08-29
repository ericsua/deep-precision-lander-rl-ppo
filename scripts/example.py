#!/usr/bin/env python3
"""
Example script demonstrating the custom Lunar Lander environment.

This script shows how to:
1. Create the custom environment
2. Take random actions
3. Observe the custom reward function in action
"""

import gymnasium as gym
import numpy as np
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def main():
    """Demonstrate the custom environment with random actions."""

    print("🚀 Deep Precision Lander RL - Example")
    print("=" * 50)

    # Create the custom environment
    print("Creating custom Lunar Lander environment...")
    base_env = gym.make("LunarLander-v3")
    env = EfficientLanderEnv(base_env)

    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Fuel efficiency bonus: {env.fuel_efficiency_bonus}")
    print(f"Landing bonus: {env.landing_bonus}")

    # Run a few episodes with random actions
    num_episodes = 3
    max_steps = 100

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")

        obs, info = env.reset()
        total_reward = 0
        step_count = 0

        for step in range(max_steps):
            # Take a random action
            action = env.action_space.sample()

            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1

            # Print action and reward for first few steps
            if step < 5:
                action_names = ["No-op", "Main Engine", "Left Engine", "Right Engine"]
                print(
                    f"  Step {step + 1}: Action={action_names[action]} (Reward: {reward:.3f})"
                )

            if terminated or truncated:
                break

        print(f"Episode {episode + 1} completed:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Final position: x={obs[0]:.3f}, y={obs[1]:.3f}")

    # Close environment
    env.close()

    print("\n" + "=" * 50)
    print("Example completed! The new balanced reward function:")
    print("• Provides continuous positive feedback for progress")
    print("• Rewards fuel efficiency with bonuses (not penalties)")
    print("• Gives substantial landing bonuses (200+ points)")
    print("• Encourages safe and precise landings")


if __name__ == "__main__":
    main()
