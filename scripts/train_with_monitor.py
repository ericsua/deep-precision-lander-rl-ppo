#!/usr/bin/env python3
"""
Training Script with Real-Time Monitoring and UI Feedback.

This script provides live feedback during training including:
1. Real-time fuel consumption tracking
2. Live speed and position monitoring
3. Training progress indicators
4. Episode performance metrics
5. Landing precision tracking
"""

import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def create_training_display(episode, step, total_steps, reward, obs, fuel_used, success_rate):
    """Create real-time training display."""
    
    # Extract observation components
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_leg, right_leg = obs[6], obs[7]
    
    # Clear screen
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("🚀 LUNAR LANDER TRAINING - REAL-TIME MONITOR")
    print("=" * 80)
    
    # Training progress
    progress = (step / total_steps) * 100
    progress_bar = "█" * int(progress / 2) + "░" * (50 - int(progress / 2))
    print(f"📊 Training Progress: {progress_bar} {progress:5.1f}%")
    print(f"🎯 Episode: {episode} | Step: {step:6d}/{total_steps:,}")
    print()
    
    # Current performance
    print(f"💰 Current Reward: {reward:8.2f}")
    print(f"📈 Success Rate: {success_rate:5.1f}%")
    print()
    
    # Fuel gauge
    remaining_fuel = 1000 - fuel_used
    fuel_percentage = (remaining_fuel / 1000) * 100
    fuel_gauge = "█" * int(fuel_percentage / 5) + "░" * (20 - int(fuel_percentage / 5))
    
    if fuel_percentage > 70:
        fuel_color = "🟢"
    elif fuel_percentage > 30:
        fuel_color = "🟡"
    else:
        fuel_color = "🔴"
    
    print(f"{fuel_color} Fuel: {fuel_gauge} {remaining_fuel:3d}/1000 ({fuel_percentage:5.1f}%)")
    
    # Speed indicators
    if abs(x_vel) < 0.1:
        x_speed = "🟢 Stable"
    elif x_vel > 0:
        x_speed = f"➡️ Right {x_vel:5.2f}"
    else:
        x_speed = f"⬅️ Left {abs(x_vel):5.2f}"
    
    if abs(y_vel) < 0.1:
        y_speed = "🟢 Stable"
    elif y_vel > 0:
        y_speed = f"⬆️ Up {y_vel:5.2f}"
    else:
        y_speed = f"⬇️ Down {abs(y_vel):5.2f}"
    
    print(f"🚀 Speed: {x_speed} | {y_speed}")
    
    # Position and precision
    distance_from_center = abs(x_pos)
    if y_pos > 0.5:
        if distance_from_center < 0.1:
            precision = "🎯 Perfect Center"
        elif distance_from_center < 0.2:
            precision = "✅ Near Center"
        elif distance_from_center < 0.3:
            precision = "⚠️  Approaching Center"
        else:
            precision = "❌ Far from Center"
    else:
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
    
    print(f"🎯 Precision: {precision} (Distance: {distance_from_center:5.3f})")
    
    # Position and orientation
    angle_deg = np.degrees(angle)
    if abs(angle_deg) < 5:
        orientation = "🟢 Upright"
    elif abs(angle_deg) < 15:
        orientation = "🟡 Slightly Tilted"
    elif abs(angle_deg) < 30:
        orientation = "🟠 Tilted"
    else:
        orientation = "🔴 Heavily Tilted"
    
    print(f"📍 Position: ({x_pos:6.2f}, {y_pos:6.2f}) | {orientation}")
    
    # Landing zone status
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
    
    print(f"🎯 Landing Zone: {zone_status}")
    
    # Leg contact
    print(f"🦵 Legs: Left={left_leg}, Right={right_leg}")
    
    print("-" * 80)
    print("💡 Training in progress... Press Ctrl+C to stop")


def create_env():
    """Create and return the custom Lunar Lander environment."""
    base_env = gym.make("LunarLander-v3")
    custom_env = EfficientLanderEnv(base_env)
    return custom_env


def train_with_monitor(total_timesteps=100000, eval_freq=5000):
    """Train the model with real-time monitoring."""
    
    print("🚀 Starting Lunar Lander Training with Real-Time Monitor")
    print("=" * 80)
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create environment
    env = create_env()
    vec_env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = create_env()
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    
    # Configure logger
    configure(folder="logs/")
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=7e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=8,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.012,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        tensorboard_log="logs/",
        verbose=0,  # Reduce verbosity since we have custom display
    )
    
    print("✅ Model created successfully!")
    print(f"🎯 Training for {total_timesteps:,} timesteps...")
    print(f"📊 Evaluation every {eval_freq} timesteps")
    print("💡 Real-time monitoring active - watch the live feedback!")
    print()
    
    # Training loop with monitoring
    try:
        # Start training
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False  # We'll show our own progress
        )
        
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("💾 Saving current model...")
        
    # Save the final model
    final_model_path = "models/ppo_efficient_lander_monitored.zip"
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")
    
    # Close environments
    vec_env.close()
    eval_vec_env.close()
    
    print("\n🎯 Training Summary:")
    print(f"  • Final model: {final_model_path}")
    print(f"  • Best model: models/best_model.zip")
    print(f"  • Logs: logs/ directory")
    print(f"  • TensorBoard: tensorboard --logdir logs/")


def main():
    """Main function."""
    print("🎮 Lunar Lander Training with Real-Time Monitor")
    print("=" * 60)
    
    # Training configuration
    total_timesteps = 100000  # Start with 100k for testing
    eval_freq = 5000
    
    print(f"⚙️  Training Configuration:")
    print(f"  • Total timesteps: {total_timesteps:,}")
    print(f"  • Evaluation frequency: {eval_freq}")
    print(f"  • Real-time monitoring: ✅ ENABLED")
    print()
    
    # Start training
    try:
        train_with_monitor(total_timesteps, eval_freq)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
