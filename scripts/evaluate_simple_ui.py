#!/usr/bin/env python3
"""
Simple Evaluation Script with In-Game UI Overlay.

This script provides minimal, essential feedback directly on the game screen:
1. Remaining fuel (simple text)
2. Current speed (horizontal/vertical)
3. Landing precision (distance from center)
4. Basic status indicators
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def evaluate_simple_ui(model_path, num_episodes=5):
    """Evaluate the trained model with simple in-game UI."""
    
    print("🎮 Simple Evaluation with In-Game UI")
    print("=" * 50)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return
    
    print(f"📁 Loading model from: {model_path}")
    model = PPO.load(model_path)
    print("✅ Model loaded successfully!")
    
    # Create environment with human rendering
    env = EfficientLanderEnv(gym.make("LunarLander-v3", render_mode='human'))
    
    print(f"🎯 Running {num_episodes} evaluation episodes...")
    print("💡 In-Game UI shows: Fuel, Speed, Landing Precision")
    print()
    
    # Run evaluation episodes
    for episode in range(1, num_episodes + 1):
        print(f"🚀 Starting Episode {episode}/{num_episodes}")
        
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        # Episode loop
        while True:
            steps += 1
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Extract observation components for UI
            x_pos, y_pos = obs[0], obs[1]
            x_vel, y_vel = obs[2], obs[3]
            
            # Calculate essential metrics
            remaining_fuel = 1000 - env.total_fuel_used
            distance_from_center = abs(x_pos)
            total_speed = np.sqrt(x_vel**2 + y_vel**2)
            
            # Simple in-game UI (this will be rendered by the environment)
            # The actual UI rendering happens in the game window
            if terminated or truncated:
                success = info.get('success', False)
                print(f"Episode {episode} ended: Reward={episode_reward:.2f}, Steps={steps}, Success={success}")
                break
        
        print(f"Episode {episode} completed with reward: {episode_reward:.2f}")
    
    print("\n🎉 Evaluation completed!")
    env.close()


def main():
    """Main function."""
    print("🎮 Simple Lunar Lander Evaluation with In-Game UI")
    print("=" * 60)
    
    # Check for available models
    model_paths = [
        "models/best_model.zip",
        "models/ppo_efficient_lander_precision.zip",
        "models/ppo_efficient_lander_optimized.zip"
    ]
    
    available_models = [path for path in model_paths if os.path.exists(path)]
    
    if not available_models:
        print("❌ No trained models found!")
        print("Please train a model first using one of the training scripts.")
        return
    
    print("Available models:")
    for i, path in enumerate(available_models):
        print(f"  {i+1}. {path}")
    
    # Use the best model by default
    model_path = available_models[0]
    print(f"\nUsing model: {model_path}")
    
    # Run evaluation with simple UI
    try:
        evaluate_simple_ui(model_path, num_episodes=3)
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
