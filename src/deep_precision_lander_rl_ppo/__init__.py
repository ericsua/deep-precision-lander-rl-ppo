"""
Deep Precision Lander RL - PPO

A Reinforcement Learning project for training a fuel-efficient and precise Lunar Lander agent
using Proximal Policy Optimization (PPO).
"""

__version__ = "0.1.0"
__author__ = "Hackathon Team"
__email__ = "team@company.com"

from .custom_lander_env import EfficientLanderEnv

__all__ = ["EfficientLanderEnv"]
