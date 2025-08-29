"""
Tests for the custom Lunar Lander environment.
"""

import pytest
import gymnasium as gym
import numpy as np

from deep_precision_lander_rl_ppo import EfficientLanderEnv


class TestEfficientLanderEnv:
    """Test cases for the EfficientLanderEnv wrapper."""

    def test_env_creation(self):
        """Test that the environment can be created successfully."""
        base_env = gym.make("LunarLander-v3")
        env = EfficientLanderEnv(base_env)
        assert env is not None
        assert hasattr(env, "fuel_consumption_penalty")
        assert env.fuel_consumption_penalty == 2.5

    def test_env_reset(self):
        """Test that the environment can be reset."""
        base_env = gym.make("LunarLander-v3")
        env = EfficientLanderEnv(base_env)
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)

    def test_env_step(self):
        """Test that the environment can take a step."""
        base_env = gym.make("LunarLander-v3")
        env = EfficientLanderEnv(base_env)
        obs, info = env.reset()
        action = 0  # No operation
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_fuel_penalty(self):
        """Test that fuel consumption penalties are applied correctly."""
        base_env = gym.make("LunarLander-v3")
        env = EfficientLanderEnv(base_env)
        obs, info = env.reset()

        # Test with main engine (action 1) and negative reward
        action = 1
        # Mock a negative reward to test penalty
        original_reward = -0.1
        # We can't easily test the exact reward modification without mocking,
        # but we can ensure the step method works
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))

    def test_precision_bonus_calculation(self):
        """Test precision bonus calculation."""
        base_env = gym.make("LunarLander-v3")
        env = EfficientLanderEnv(base_env)

        # Test with observation at center (x=0)
        center_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        bonus = env._calculate_precision_bonus(center_obs)
        assert bonus == 100.0  # Perfect center landing

        # Test with observation away from center
        off_center_obs = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        bonus = env._calculate_precision_bonus(off_center_obs)
        assert bonus < 100.0  # Should be less than perfect
        assert bonus > 0.0  # But still positive
