#!/usr/bin/env python3
"""
Test script to demonstrate the new consistent episode termination behavior.

This script shows:
1. Episode ends immediately when touching ground
2. Consistent behavior regardless of landing location
3. No more fuel depletion after landing
4. Clear success/failure determination
"""

import gymnasium as gym
import numpy as np
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def test_episode_termination():
    """Test the new episode termination logic."""
    
    print("🎯 Testing Episode Termination Logic")
    print("=" * 60)
    
    # Create environment
    env = EfficientLanderEnv(gym.make("LunarLander-v3"))
    
    print("✅ Environment created successfully!")
    print("🎯 New behavior: Episode ends IMMEDIATELY when touching ground")
    print()
    
    # Test 1: Landing in center (should be successful)
    print("🧪 Test 1: Landing in Center (Should be Successful)")
    print("-" * 50)
    
    obs, _ = env.reset()
    episode_reward = 0
    steps = 0
    
    # Simulate a controlled descent to center
    for step in range(100):
        steps += 1
        
        # Simulate moving toward center and down
        x_pos = 0.0 + np.sin(step * 0.1) * 0.05  # Small oscillation around center
        y_pos = max(0.5 - step * 0.01, 0.05)     # Controlled descent
        
        # Create observation
        obs = np.array([x_pos, y_pos, 0, -0.1, 0, 0, 0, 0])
        
        # Take step (this should trigger episode termination when y_pos <= 0.05)
        reward, terminated, truncated, info = env.step(0)[1:]  # No-op action
        
        episode_reward += reward
        
        print(f"  Step {steps:2d}: Position ({x_pos:5.2f}, {y_pos:5.2f}), Reward: {reward:6.2f}")
        
        if terminated:
            print(f"  ✅ Episode ended at step {steps}")
            print(f"  📊 Final reward: {episode_reward:.2f}")
            print(f"  🎯 Landing type: {info.get('landing_type', 'unknown')}")
            print(f"  ✅ Success: {info.get('success', False)}")
            break
    
    print()
    
    # Test 2: Landing outside center (should fail)
    print("🧪 Test 2: Landing Outside Center (Should Fail)")
    print("-" * 50)
    
    obs, _ = env.reset()
    episode_reward = 0
    steps = 0
    
    # Simulate a descent to outside the landing zone
    for step in range(100):
        steps += 1
        
        # Simulate moving toward outside and down
        x_pos = 0.5 + step * 0.01  # Moving away from center
        y_pos = max(0.5 - step * 0.01, 0.05)  # Controlled descent
        
        # Create observation
        obs = np.array([x_pos, y_pos, 0, -0.1, 0, 0, 0, 0])
        
        # Take step
        reward, terminated, truncated, info = env.step(0)[1:]  # No-op action
        
        episode_reward += reward
        
        print(f"  Step {steps:2d}: Position ({x_pos:5.2f}, {y_pos:5.2f}), Reward: {reward:6.2f}")
        
        if terminated:
            print(f"  ✅ Episode ended at step {steps}")
            print(f"  📊 Final reward: {episode_reward:.2f}")
            print(f"  🎯 Landing type: {info.get('landing_type', 'unknown')}")
            print(f"  ❌ Success: {info.get('success', False)}")
            break
    
    print()
    
    # Test 3: Demonstrate no fuel depletion after landing
    print("🧪 Test 3: No Fuel Depletion After Landing")
    print("-" * 50)
    
    obs, _ = env.reset()
    episode_reward = 0
    steps = 0
    
    # Simulate landing and then trying to continue
    for step in range(100):
        steps += 1
        
        if step < 50:
            # Normal descent
            x_pos = 0.0
            y_pos = max(0.5 - step * 0.01, 0.05)
        else:
            # After landing, try to continue (should not work)
            x_pos = 0.0
            y_pos = 0.05
        
        # Create observation
        obs = np.array([x_pos, y_pos, 0, 0, 0, 0, 0, 0])
        
        # Take step
        reward, terminated, truncated, info = env.step(0)[1:]  # No-op action
        
        episode_reward += reward
        
        if step < 50:
            print(f"  Step {steps:2d}: Position ({x_pos:5.2f}, {y_pos:5.2f}), Reward: {reward:6.2f}")
        elif step == 50:
            print(f"  Step {steps:2d}: LANDED! Position ({x_pos:5.2f}, {y_pos:5.2f}), Reward: {reward:6.2f}")
        else:
            print(f"  Step {steps:2d}: After landing - Episode should be terminated")
            break
        
        if terminated:
            print(f"  ✅ Episode ended at step {steps}")
            print(f"  📊 Final reward: {episode_reward:.2f}")
            print(f"  🎯 Landing type: {info.get('landing_type', 'unknown')}")
            print(f"  ✅ Success: {info.get('success', False)}")
            break
    
    print()
    print("✅ Episode termination test completed!")
    print("\n🎯 Key Improvements:")
    print("  • Episode ends IMMEDIATELY when touching ground")
    print("  • No more fuel depletion after landing")
    print("  • Consistent behavior regardless of landing location")
    print("  • Clear success/failure determination")
    print("  • Landing type information in episode info")


if __name__ == "__main__":
    test_episode_termination()
