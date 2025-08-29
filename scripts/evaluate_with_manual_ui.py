#!/usr/bin/env python3
"""
Evaluation Script with Manual UI Overlay.

This script manually creates a pygame window and overlays the UI on top of the game.
"""

import os
import time
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def create_ui_window():
    """Create a separate pygame window for UI overlay."""
    pygame.init()
    
    # Create a small UI window that floats on top
    ui_width, ui_height = 400, 200
    ui_screen = pygame.display.set_mode((ui_width, ui_height), pygame.NOFRAME | pygame.SRCALPHA)
    pygame.display.set_caption("Lunar Lander UI")
    
    # Make window stay on top
    try:
        # This might not work on all systems, but worth trying
        ui_screen.set_alpha(0.9)
    except:
        pass
    
    return ui_screen


def draw_ui_overlay(ui_screen, obs, fuel_used, episode_steps):
    """Draw UI overlay on the separate window."""
    # Clear with semi-transparent background
    ui_screen.fill((0, 0, 0, 180))  # Semi-transparent black
    
    # Extract observation components
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    
    # Calculate UI metrics
    remaining_fuel = 1000 - fuel_used
    distance_from_center = abs(x_pos)
    total_speed = np.sqrt(x_vel**2 + y_vel**2)
    
    # Font setup
    font_size = 18
    try:
        font = pygame.font.Font(None, font_size)
    except:
        font = pygame.font.Font(None, 24)
    
    # Colors
    white = (255, 255, 255)
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    red = (255, 0, 0)
    blue = (0, 150, 255)
    dark_gray = (40, 40, 40)
    
    # UI Elements
    ui_elements = []
    
    # 1. Fuel Gauge (top left)
    fuel_color = green if remaining_fuel > 700 else yellow if remaining_fuel > 300 else red
    fuel_text = f"FUEL: {remaining_fuel}/1000"
    fuel_surface = font.render(fuel_text, True, fuel_color)
    fuel_rect = fuel_surface.get_rect()
    fuel_rect.topleft = (10, 10)
    ui_elements.append((fuel_surface, fuel_rect))
    
    # 2. Speed Indicator (top right)
    speed_color = green if total_speed < 0.5 else yellow if total_speed < 1.0 else red
    speed_text = f"SPEED: {total_speed:.2f}"
    speed_surface = font.render(speed_text, True, speed_color)
    speed_rect = speed_surface.get_rect()
    speed_rect.topright = (ui_screen.get_width() - 10, 10)
    ui_elements.append((speed_surface, speed_rect))
    
    # 3. Landing Precision (center)
    if y_pos > 0.5:
        precision_text = f"DISTANCE: {distance_from_center:.3f}"
        precision_color = green if distance_from_center < 0.1 else yellow if distance_from_center < 0.2 else red
    else:
        if distance_from_center < 0.05:
            precision_text = "🎯 PERFECT!"
            precision_color = green
        elif distance_from_center < 0.1:
            precision_text = "✅ EXCELLENT"
            precision_color = green
        elif distance_from_center < 0.2:
            precision_text = "⚠️  GOOD"
            precision_color = yellow
        elif distance_from_center < 0.3:
            precision_text = "❌ POOR"
            precision_color = red
        else:
            precision_text = "💥 OUTSIDE!"
            precision_color = red
    
    precision_surface = font.render(precision_text, True, precision_color)
    precision_rect = precision_surface.get_rect()
    precision_rect.centerx = ui_screen.get_width() // 2
    precision_rect.y = 40
    ui_elements.append((precision_surface, precision_rect))
    
    # 4. Position Info (bottom left)
    pos_text = f"POS: ({x_pos:.2f}, {y_pos:.2f})"
    pos_surface = font.render(pos_text, True, blue)
    pos_rect = pos_surface.get_rect()
    pos_rect.bottomleft = (10, ui_screen.get_height() - 10)
    ui_elements.append((pos_surface, pos_rect))
    
    # 5. Episode Info (bottom right)
    episode_text = f"STEPS: {episode_steps}"
    episode_surface = font.render(episode_text, True, white)
    episode_rect = episode_surface.get_rect()
    episode_rect.bottomright = (ui_screen.get_width() - 10, ui_screen.get_height() - 10)
    ui_elements.append((episode_surface, episode_rect))
    
    # Draw all UI elements
    for element_surface, position in ui_elements:
        ui_screen.blit(element_surface, position)
    
    # Update the UI window
    pygame.display.flip()


def evaluate_with_manual_ui(model_path, num_episodes=3):
    """Evaluate the trained model with manual UI overlay."""
    
    print("🎮 Evaluation with Manual UI Overlay")
    print("=" * 50)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return
    
    print(f"📁 Loading model from: {model_path}")
    model = PPO.load(model_path)
    print("✅ Model loaded successfully!")
    
    # Create environment
    env = EfficientLanderEnv(gym.make("LunarLander-v3", render_mode='human'))
    
    # Create UI window
    ui_screen = create_ui_window()
    
    print(f"🎯 Running {num_episodes} evaluation episodes...")
    print("💡 UI window should appear with real-time info")
    print("🎮 Game window will show the lunar lander")
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
            
            # Update UI overlay
            draw_ui_overlay(ui_screen, obs, env.total_fuel_used, steps)
            
            # Handle pygame events for UI window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        env.close()
                        return
            
            # Check if episode ended
            if terminated or truncated:
                success = info.get('success', False)
                print(f"Episode {episode} ended: Reward={episode_reward:.2f}, Steps={steps}, Success={success}")
                break
            
            # Small delay for UI updates
            time.sleep(0.05)
        
        print(f"Episode {episode} completed with reward: {episode_reward:.2f}")
    
    print("\n🎉 Evaluation completed!")
    
    # Clean up
    pygame.quit()
    env.close()


def main():
    """Main function."""
    print("🎮 Lunar Lander Evaluation with Manual UI Overlay")
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
    
    # Run evaluation with manual UI
    try:
        evaluate_with_manual_ui(model_path, num_episodes=3)
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
