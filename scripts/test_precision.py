#!/usr/bin/env python3
"""
Test script to demonstrate the new precision-focused reward function features.

This script shows:
1. Center landing rewards
2. Movement prevention after landing
3. Vertical alignment bonuses
4. Precision landing zone requirements
"""

import gymnasium as gym
import numpy as np
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def test_precision_features():
    """Test the new precision-focused reward function features."""
    
    print("🎯 Testing Precision-Focused Reward Function Features")
    print("=" * 60)
    
    # Create environment
    env = EfficientLanderEnv(gym.make("LunarLander-v3"))
    
    print("✅ Environment created successfully!")
    print(f"Center precision bonus: {env.center_precision_bonus}")
    print(f"Vertical alignment bonus: {env.vertical_alignment_bonus}")
    print(f"Landing bonus: {env.landing_bonus}")
    print()
    
    # Test vertical alignment bonus
    print("🧪 Testing Vertical Alignment Bonus:")
    test_positions = [
        (0.0, 1.0),    # Perfect center, high up
        (0.1, 1.0),    # Slightly off center, high up
        (0.2, 1.0),    # More off center, high up
        (0.0, 0.3),    # Perfect center, low (should get no bonus)
        (0.1, 0.3),    # Off center, low (should get no bonus)
    ]
    
    for x_pos, y_pos in test_positions:
        obs = np.array([x_pos, y_pos, 0, 0, 0, 0, 0, 0])
        bonus = env._calculate_vertical_alignment_bonus(obs)
        print(f"  Position ({x_pos:4.1f}, {y_pos:4.1f}): {bonus:6.2f} bonus")
    
    print()
    
    # Test center precision bonus
    print("🎯 Testing Center Precision Bonus:")
    test_distances = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    
    for distance in test_distances:
        obs = np.array([distance, 0.05, 0, 0, 0, 0, 0, 0])
        bonus = env._calculate_landing_bonus(obs)
        print(f"  Distance from center {distance:4.2f}: {bonus:6.1f} total bonus")
    
    print()
    
    # Test movement prevention
    print("🚫 Testing Movement Prevention After Landing:")
    print("  This feature tracks landing position and penalizes movement")
    print("  Movement penalty increases with each step after landing")
    print("  Landing position tracking resets on environment reset")
    
    # Simulate landing and movement
    env.reset()
    obs = np.array([0.0, 0.05, 0, 0, 0, 0, 0, 0])  # Landed at center
    
    # Simulate landing
    env.has_landed = True
    env.landing_position = 0.0
    
    # Test movement penalties
    for i in range(5):
        # Simulate moving slightly
        new_x = 0.0 + (i + 1) * 0.02
        obs = np.array([new_x, 0.05, 0, 0, 0, 0, 0, 0])
        
        # Calculate reward with movement
        reward = env._calculate_landing_encouragement(obs)
        print(f"  Step {i+1}: Position {new_x:5.2f}, Movement penalty: {reward:6.2f}")
    
    print()
    print("✅ Precision features test completed!")
    print("\n🎯 Key Improvements:")
    print("  • Center landing gets 500+ bonus (vs 200 base)")
    print("  • Vertical alignment rewards approach from center")
    print("  • Movement after landing is heavily penalized")
    print("  • More precise landing zone requirements")
    print("  • Better fuel efficiency tracking")


if __name__ == "__main__":
    test_precision_features()
