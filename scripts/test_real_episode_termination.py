#!/usr/bin/env python3
"""
Real environment test script to demonstrate episode termination when touching ground.

This script actually runs the environment to test:
1. Episode ends immediately when touching ground
2. No fuel depletion after landing
3. Consistent behavior
"""

import gymnasium as gym
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def test_real_episode_termination():
    """Test episode termination with the real environment."""
    
    print("🎯 Testing Real Environment Episode Termination")
    print("=" * 60)
    
    # Create environment
    env = EfficientLanderEnv(gym.make("LunarLander-v3"))
    
    print("✅ Environment created successfully!")
    print("🎯 Testing: Episode should end IMMEDIATELY when touching ground")
    print()
    
    # Test 1: Run a few episodes to see the behavior
    for episode in range(3):
        print(f"🧪 Episode {episode + 1}")
        print("-" * 30)
        
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 1000
        
        for step in range(max_steps):
            steps += 1
            
            # Take a random action (or no-op) to see what happens
            action = 0  # No-op action
            
            # Take step in the real environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            
            # Get position information
            x_pos, y_pos = obs[0], obs[1]
            left_leg, right_leg = obs[6], obs[7]
            
            # Print every 50 steps or when something interesting happens
            if (step % 50 == 0) or (y_pos < 0.2) or terminated or truncated:
                print(f"  Step {steps:3d}: Position ({x_pos:6.2f}, {y_pos:6.2f}), "
                      f"Legs: ({left_leg}, {right_leg}), Reward: {reward:6.2f}")
            
            # Check if episode ended
            if terminated or truncated:
                print(f"  ✅ Episode ended at step {steps}")
                print(f"  📊 Final reward: {episode_reward:.2f}")
                print(f"  🎯 Landing type: {info.get('landing_type', 'unknown')}")
                print(f"  ✅ Success: {info.get('success', False)}")
                print(f"  🚀 Fuel used: {env.total_fuel_used}")
                break
        
        print()
    
    print("✅ Real environment test completed!")
    print("\n🎯 What to look for:")
    print("  • Episode should end when y_pos <= 0.05 or legs touch ground")
    print("  • No more steps after landing")
    print("  • Clear success/failure determination")
    print("  • Fuel usage tracking")


if __name__ == "__main__":
    test_real_episode_termination()
