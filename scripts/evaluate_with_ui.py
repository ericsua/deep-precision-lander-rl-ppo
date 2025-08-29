#!/usr/bin/env python3
"""
Enhanced Evaluation Script with Comprehensive UI Feedback.

This script provides real-time feedback during evaluation including:
1. Fuel gauge and remaining fuel
2. Speed indicators (horizontal/vertical velocity)
3. Landing precision (distance from center)
4. Episode progress and metrics
5. Real-time position and angle information
6. Landing zone indicators
"""

import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def create_fuel_gauge(fuel_used, max_fuel=1000):
    """Create a visual fuel gauge."""
    remaining = max_fuel - fuel_used
    percentage = (remaining / max_fuel) * 100
    
    # Create visual gauge
    gauge_length = 20
    filled_length = int((remaining / max_fuel) * gauge_length)
    empty_length = gauge_length - filled_length
    
    gauge = "█" * filled_length + "░" * empty_length
    
    # Color coding
    if percentage > 70:
        color = "🟢"  # Green
    elif percentage > 30:
        color = "🟡"  # Yellow
    else:
        color = "🔴"  # Red
    
    return f"{color} Fuel: {gauge} {remaining:3d}/{max_fuel} ({percentage:5.1f}%)"


def create_speed_indicator(x_vel, y_vel):
    """Create speed indicators with direction."""
    # Horizontal speed
    if abs(x_vel) < 0.1:
        x_indicator = "🟢 Stable"
    elif x_vel > 0:
        x_indicator = f"➡️ Right {x_vel:5.2f}"
    else:
        x_indicator = f"⬅️ Left {abs(x_vel):5.2f}"
    
    # Vertical speed
    if abs(y_vel) < 0.1:
        y_indicator = "🟢 Stable"
    elif y_vel > 0:
        y_indicator = f"⬆️ Up {y_vel:5.2f}"
    else:
        y_indicator = f"⬇️ Down {abs(y_vel):5.2f}"
    
    return f"Speed: {x_indicator} | {y_indicator}"


def create_precision_indicator(x_pos, y_pos):
    """Create landing precision indicator."""
    distance_from_center = abs(x_pos)
    
    if y_pos > 0.5:
        # Still high up
        if distance_from_center < 0.1:
            precision = "🎯 Perfect Center"
        elif distance_from_center < 0.2:
            precision = "✅ Near Center"
        elif distance_from_center < 0.3:
            precision = "⚠️  Approaching Center"
        else:
            precision = "❌ Far from Center"
    else:
        # Close to ground
        if distance_from_center < 0.05:
            precision = "🎯 PERFECT LANDING!"
        elif distance_from_center < 0.1:
            precision = "✅ Excellent Landing"
        elif distance_from_center < 0.2:
            precision = "⚠️  Good Landing"
        elif distance_from_center < 0.3:
            precision = "❌ Poor Landing"
        else:
            precision = "💥 Outside Landing Zone!"
    
    return f"Precision: {precision} (Distance: {distance_from_center:5.3f})"


def create_position_display(x_pos, y_pos, angle, ang_vel):
    """Create position and orientation display."""
    # Position info
    position = f"Position: ({x_pos:6.2f}, {y_pos:6.2f})"
    
    # Angle info (convert to degrees for readability)
    angle_deg = np.degrees(angle)
    if abs(angle_deg) < 5:
        orientation = "🟢 Upright"
    elif abs(angle_deg) < 15:
        orientation = "🟡 Slightly Tilted"
    elif abs(angle_deg) < 30:
        orientation = "🟠 Tilted"
    else:
        orientation = "🔴 Heavily Tilted"
    
    # Angular velocity
    ang_vel_deg = np.degrees(ang_vel)
    if abs(ang_vel_deg) < 5:
        rotation = "🟢 Stable"
    elif abs(ang_vel_deg) < 15:
        rotation = "🟡 Slow Rotation"
    else:
        rotation = "🔴 Fast Rotation"
    
    return f"{position} | {orientation} | Rotation: {rotation}"


def create_landing_zone_indicator(x_pos, y_pos):
    """Create landing zone indicator."""
    in_landing_zone = (abs(x_pos) < 0.3 and y_pos < 0.5)
    
    if in_landing_zone:
        if y_pos < 0.1:
            zone_status = "🟢 IN LANDING ZONE - Very Close!"
        elif y_pos < 0.3:
            zone_status = "🟡 IN LANDING ZONE - Approaching"
        else:
            zone_status = "🟠 IN LANDING ZONE - High Up"
    else:
        if abs(x_pos) > 0.3:
            zone_status = "🔴 OUTSIDE LANDING ZONE - Too Far Horizontally"
        elif y_pos > 0.5:
            zone_status = "🔴 OUTSIDE LANDING ZONE - Too High"
        else:
            zone_status = "🔴 OUTSIDE LANDING ZONE"
    
    return f"Landing Zone: {zone_status}"


def create_episode_summary(episode_num, total_reward, steps, fuel_used, success):
    """Create episode summary display."""
    print("\n" + "="*80)
    print(f"🎯 EPISODE {episode_num} SUMMARY")
    print("="*80)
    
    # Success indicator
    if success:
        print("✅ SUCCESSFUL LANDING!")
    else:
        print("❌ FAILED LANDING")
    
    # Key metrics
    print(f"📊 Total Reward: {total_reward:8.2f}")
    print(f"⏱️  Steps Taken: {steps}")
    print(f"🚀 Fuel Used: {fuel_used}")
    
    # Performance rating
    if total_reward > 1000:
        rating = "🏆 EXCELLENT"
    elif total_reward > 500:
        rating = "🥇 GREAT"
    elif total_reward > 200:
        rating = "🥈 GOOD"
    elif total_reward > 0:
        rating = "🥉 FAIR"
    else:
        rating = "💥 POOR"
    
    print(f"⭐ Performance: {rating}")
    print("="*80)


def evaluate_with_ui(model_path, num_episodes=5, render_mode='human'):
    """Evaluate the trained model with comprehensive UI feedback."""
    
    print("🎮 Enhanced Evaluation with Comprehensive UI Feedback")
    print("=" * 80)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return
    
    print(f"📁 Loading model from: {model_path}")
    model = PPO.load(model_path)
    print("✅ Model loaded successfully!")
    
    # Create environment
    env = EfficientLanderEnv(gym.make("LunarLander-v3", render_mode=render_mode))
    
    print(f"🎯 Running {num_episodes} evaluation episodes...")
    print("💡 UI Features:")
    print("  • Fuel gauge with color coding")
    print("  • Speed indicators with direction")
    print("  • Landing precision tracking")
    print("  • Real-time position and orientation")
    print("  • Landing zone status")
    print("  • Episode progress and metrics")
    print()
    
    # Run evaluation episodes
    for episode in range(1, num_episodes + 1):
        print(f"\n🚀 Starting Episode {episode}/{num_episodes}")
        print("-" * 60)
        
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
            
            # Extract observation components
            x_pos, y_pos = obs[0], obs[1]
            x_vel, y_vel = obs[2], obs[3]
            angle, ang_vel = obs[4], obs[5]
            left_leg, right_leg = obs[6], obs[7]
            
            # Create UI displays
            fuel_display = create_fuel_gauge(env.total_fuel_used)
            speed_display = create_speed_indicator(x_vel, y_vel)
            precision_display = create_precision_indicator(x_pos, y_pos)
            position_display = create_position_display(x_pos, y_pos, angle, ang_vel)
            zone_display = create_landing_zone_indicator(x_pos, y_pos)
            
            # Clear screen and display UI (simple approach)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"🎮 Episode {episode}/{num_episodes} | Step {steps:3d}")
            print("=" * 60)
            print(fuel_display)
            print(speed_display)
            print(precision_display)
            print(position_display)
            print(zone_display)
            print("-" * 60)
            print(f"📊 Current Reward: {reward:6.2f} | Total: {episode_reward:8.2f}")
            print(f"🎯 Legs: Left={left_leg}, Right={right_leg}")
            
            # Check if episode ended
            if terminated or truncated:
                success = info.get('success', False)
                create_episode_summary(episode, episode_reward, steps, env.total_fuel_used, success)
                break
            
            # Small delay for readability
            time.sleep(0.1)
    
    # Final summary
    print("\n🎉 Evaluation completed!")
    print("💡 The UI provided real-time feedback on:")
    print("  • Fuel consumption and efficiency")
    print("  • Landing precision and approach")
    print("  • Speed and stability during flight")
    print("  • Landing zone positioning")
    print("  • Episode progress and performance")
    
    env.close()


def main():
    """Main function."""
    print("🎮 Enhanced Lunar Lander Evaluation with UI Feedback")
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
    
    # Run evaluation with UI
    try:
        evaluate_with_ui(model_path, num_episodes=3)
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
