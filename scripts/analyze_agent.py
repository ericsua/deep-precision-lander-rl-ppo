#!/usr/bin/env python3
"""
Comprehensive Agent Behavior Analysis Script for the Trained Lunar Lander RL Agent.

This script analyzes:
1. Agent's landing strategies and patterns
2. Fuel efficiency analysis
3. Precision landing statistics
4. Behavioral patterns and decision making
5. Performance metrics and trends
6. Reward function effectiveness

Key features:
- Detailed episode analysis
- Fuel usage patterns
- Landing precision metrics
- Behavioral clustering
- Performance visualization
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import our custom environment from the package
from deep_precision_lander_rl_ppo import EfficientLanderEnv


def create_env(render_mode=None):
    """
    Create and return the custom Lunar Lander environment.
    
    Args:
        render_mode: Optional render mode for visualization
        
    Returns:
        EfficientLanderEnv: The wrapped environment
    """
    # Create the base LunarLander-v3 environment
    base_env = gym.make("LunarLander-v3", render_mode=render_mode)
    
    # Wrap it with our custom EfficientLanderEnv
    custom_env = EfficientLanderEnv(base_env)
    
    return custom_env


def load_model(model_path):
    """
    Load a trained PPO model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        PPO: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    return model


def analyze_episode(env, model, episode_num, max_steps=1000):
    """
    Analyze a single episode in detail.
    
    Args:
        env: The environment
        model: The trained model
        episode_num: Episode number for tracking
        max_steps: Maximum steps per episode
        
    Returns:
        dict: Detailed episode analysis
    """
    obs, _ = env.reset()
    episode_data = {
        'episode': episode_num,
        'actions': [],
        'rewards': [],
        'positions': [],
        'velocities': [],
        'angles': [],
        'fuel_usage': 0,
        'total_reward': 0,
        'steps': 0,
        'landed': False,
        'crashed': False,
        'final_position': None,
        'landing_precision': None,
        'hovering_time': 0,
        'in_landing_zone': False
    }
    
    for step in range(max_steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Track action (convert numpy array to int if needed)
        if hasattr(action, 'item'):
            action = action.item()
        episode_data['actions'].append(action)
        
        # Track fuel usage
        if action in [1, 2, 3]:  # Main engine, left engine, right engine
            episode_data['fuel_usage'] += 1
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track data
        episode_data['rewards'].append(reward)
        episode_data['total_reward'] += reward
        episode_data['steps'] += 1
        
        # Track position and state
        x_pos, y_pos = obs[0], obs[1]
        x_vel, y_vel = obs[2], obs[3]
        angle, ang_vel = obs[4], obs[5]
        
        episode_data['positions'].append((x_pos, y_pos))
        episode_data['velocities'].append((x_vel, y_vel))
        episode_data['angles'].append((angle, ang_vel))
        
        # Check if in landing zone
        in_landing_zone = (abs(x_pos) < 0.25 and y_pos < 0.4)
        if in_landing_zone:
            if not episode_data['in_landing_zone']:
                episode_data['in_landing_zone'] = True
            episode_data['hovering_time'] += 1
        
        # Check episode end
        if terminated:
            episode_data['landed'] = info.get('success', False)
            episode_data['crashed'] = not info.get('success', False)
            episode_data['final_position'] = (x_pos, y_pos)
            episode_data['landing_precision'] = abs(x_pos)  # Distance from center
            break
        elif truncated:
            episode_data['final_position'] = (x_pos, y_pos)
            episode_data['landing_precision'] = abs(x_pos)
            break
    
    return episode_data


def run_analysis_episodes(model_path, num_episodes=50):
    """
    Run multiple analysis episodes and collect comprehensive data.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to analyze
        
    Returns:
        list: List of episode analysis data
    """
    print(f"Running {num_episodes} analysis episodes...")
    
    # Load model
    model = load_model(model_path)
    
    # Create environment
    env = create_env()
    vec_env = DummyVecEnv([lambda: env])
    
    # Run analysis episodes
    all_episodes = []
    for i in range(num_episodes):
        episode_data = analyze_episode(env, model, i + 1)
        all_episodes.append(episode_data)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_episodes} episodes")
    
    # Close environment
    vec_env.close()
    
    return all_episodes


def analyze_performance(episodes):
    """
    Analyze overall performance from episode data.
    
    Args:
        episodes: List of episode analysis data
        
    Returns:
        dict: Performance analysis results
    """
    print("\n📊 PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    total_episodes = len(episodes)
    successful_landings = sum(1 for ep in episodes if ep['landed'])
    crashes = sum(1 for ep in episodes if ep['crashed'])
    timeouts = total_episodes - successful_landings - crashes
    
    print(f"Total Episodes: {total_episodes}")
    print(f"Successful Landings: {successful_landings} ({100*successful_landings/total_episodes:.1f}%)")
    print(f"Crashes: {crashes} ({100*crashes/total_episodes:.1f}%)")
    print(f"Timeouts: {timeouts} ({100*timeouts/total_episodes:.1f}%)")
    
    # Reward analysis
    rewards = [ep['total_reward'] for ep in episodes]
    print(f"\nReward Statistics:")
    print(f"  Average Reward: {np.mean(rewards):.2f}")
    print(f"  Best Reward: {np.max(rewards):.2f}")
    print(f"  Worst Reward: {np.min(rewards):.2f}")
    print(f"  Reward Std Dev: {np.std(rewards):.2f}")
    
    # Fuel efficiency analysis
    fuel_usage = [ep['fuel_usage'] for ep in episodes]
    print(f"\nFuel Efficiency:")
    print(f"  Average Fuel Used: {np.mean(fuel_usage):.1f}")
    print(f"  Most Fuel Used: {np.max(fuel_usage)}")
    print(f"  Least Fuel Used: {np.min(fuel_usage)}")
    print(f"  Fuel Efficiency Score: {1000/np.mean(fuel_usage):.1f}")
    
    # Landing precision analysis
    successful_episodes = [ep for ep in episodes if ep['landed']]
    if successful_episodes:
        precision_scores = [ep['landing_precision'] for ep in successful_episodes]
        print(f"\nLanding Precision (Successful Landings):")
        print(f"  Average Distance from Center: {np.mean(precision_scores):.3f}")
        print(f"  Best Precision: {np.min(precision_scores):.3f}")
        print(f"  Precision Score: {100 * np.exp(-5 * np.mean(precision_scores)):.1f}")
    
    # Episode length analysis
    episode_lengths = [ep['steps'] for ep in episodes]
    print(f"\nEpisode Length:")
    print(f"  Average Steps: {np.mean(episode_lengths):.1f}")
    print(f"  Shortest Episode: {np.min(episode_lengths)}")
    print(f"  Longest Episode: {np.max(episode_lengths)}")
    
    # Hovering analysis
    hovering_times = [ep['hovering_time'] for ep in episodes]
    print(f"\nHovering Behavior:")
    print(f"  Average Hovering Time: {np.mean(hovering_times):.1f} steps")
    print(f"  Max Hovering Time: {np.max(hovering_times)} steps")
    
    return {
        'total_episodes': total_episodes,
        'success_rate': successful_landings / total_episodes,
        'avg_reward': np.mean(rewards),
        'avg_fuel_usage': np.mean(fuel_usage),
        'avg_precision': np.mean(precision_scores) if successful_episodes else None,
        'avg_episode_length': np.mean(episode_lengths),
        'avg_hovering_time': np.mean(hovering_times)
    }


def analyze_behavioral_patterns(episodes):
    """
    Analyze behavioral patterns and decision making.
    
    Args:
        episodes: List of episode analysis data
    """
    print("\n🧠 BEHAVIORAL PATTERN ANALYSIS")
    print("=" * 50)
    
    # Action distribution analysis
    all_actions = []
    for ep in episodes:
        all_actions.extend(ep['actions'])
    
    action_counts = Counter(all_actions)
    action_names = ['No-op', 'Main Engine', 'Left Engine', 'Right Engine']
    
    print("Action Distribution:")
    for action, count in action_counts.items():
        percentage = 100 * count / len(all_actions)
        print(f"  {action_names[action]}: {count} ({percentage:.1f}%)")
    
    # Landing strategy analysis
    successful_episodes = [ep for ep in episodes if ep['landed']]
    if successful_episodes:
        print(f"\nLanding Strategy Analysis ({len(successful_episodes)} successful landings):")
        
        # Analyze approach patterns
        approach_speeds = []
        approach_angles = []
        
        for ep in successful_episodes:
            if len(ep['velocities']) > 10:
                # Look at velocity when approaching landing zone
                approach_speeds.append(abs(ep['velocities'][-10][1]))  # Y velocity
                approach_angles.append(abs(ep['angles'][-10][0]))      # Angle
        
        if approach_speeds:
            print(f"  Average Approach Speed: {np.mean(approach_speeds):.2f}")
            print(f"  Average Approach Angle: {np.mean(approach_angles):.2f}")
    
    # Fuel usage patterns
    fuel_efficient_episodes = [ep for ep in episodes if ep['fuel_usage'] < 50]
    moderate_fuel_episodes = [ep for ep in episodes if 50 <= ep['fuel_usage'] < 100]
    high_fuel_episodes = [ep for ep in episodes if ep['fuel_usage'] >= 100]
    
    print(f"\nFuel Usage Patterns:")
    print(f"  High Efficiency (<50 fuel): {len(fuel_efficient_episodes)} episodes")
    print(f"  Moderate Efficiency (50-100 fuel): {len(moderate_fuel_episodes)} episodes")
    print(f"  High Usage (≥100 fuel): {len(high_fuel_episodes)} episodes")


def create_visualizations(episodes, save_path="analysis_plots"):
    """
    Create visualizations of the analysis results.
    
    Args:
        episodes: List of episode analysis data
        save_path: Directory to save plots
    """
    print(f"\n📈 Creating visualizations in '{save_path}' directory...")
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Reward distribution
    plt.figure(figsize=(10, 6))
    rewards = [ep['total_reward'] for ep in episodes]
    plt.hist(rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Total Reward')
    plt.ylabel('Number of Episodes')
    plt.title('Reward Distribution Across Episodes')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}/reward_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Fuel usage vs reward
    plt.figure(figsize=(10, 6))
    fuel_usage = [ep['fuel_usage'] for ep in episodes]
    plt.scatter(fuel_usage, rewards, alpha=0.6, color='green')
    plt.xlabel('Fuel Used')
    plt.ylabel('Total Reward')
    plt.title('Fuel Usage vs Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}/fuel_vs_reward.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Landing precision for successful landings
    successful_episodes = [ep for ep in episodes if ep['landed']]
    if successful_episodes:
        plt.figure(figsize=(10, 6))
        precision_scores = [ep['landing_precision'] for ep in successful_episodes]
        plt.hist(precision_scores, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Distance from Center')
        plt.ylabel('Number of Landings')
        plt.title('Landing Precision Distribution (Successful Landings)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_path}/landing_precision.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Episode length distribution
    plt.figure(figsize=(10, 6))
    episode_lengths = [ep['steps'] for ep in episodes]
    plt.hist(episode_lengths, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Episode Length (Steps)')
    plt.ylabel('Number of Episodes')
    plt.title('Episode Length Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}/episode_lengths.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved successfully!")


def main():
    """Main analysis function."""
    print("🔍 COMPREHENSIVE AGENT BEHAVIOR ANALYSIS")
    print("=" * 60)
    
    # Check for available models
    model_paths = [
        "models/best_model.zip",
        "models/ppo_efficient_lander_extended.zip",
        "models/ppo_efficient_lander_landing_focused.zip",
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
    
    # Run analysis
    try:
        episodes = run_analysis_episodes(model_path, num_episodes=50)
        
        # Analyze performance
        performance = analyze_performance(episodes)
        
        # Analyze behavioral patterns
        analyze_behavioral_patterns(episodes)
        
        # Create visualizations
        create_visualizations(episodes)
        
        print("\n✅ Analysis completed successfully!")
        print(f"Results saved in 'analysis_plots' directory")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
