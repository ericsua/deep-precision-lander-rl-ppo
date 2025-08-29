import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EfficientLanderEnv(gym.Wrapper):
    """
    Custom wrapper for LunarLander-v3 that implements fuel-efficient and precise landing objectives.

    This wrapper modifies the reward function to:
    1. Heavily penalize fuel consumption (2.5x multiplier)
    2. Reward precision landing based on distance from center
    """

    def __init__(self, env):
        super().__init__(env)
        self.original_env = env

        # Store the original action space for reference
        self.original_action_space = env.action_space

        # Track fuel consumption for reward modification
        self.fuel_consumption_penalty = 2.5

    def step(self, action):
        """
        Override the step method to implement custom reward logic.

        Args:
            action: The action taken by the agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Take the original step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply fuel consumption penalty
        modified_reward = self._apply_fuel_penalty(action, reward)

        # Apply precision landing bonus if episode is terminated successfully
        if terminated and info.get("success", False):
            modified_reward += self._calculate_precision_bonus(obs)

        return obs, modified_reward, terminated, truncated, info

    def _apply_fuel_penalty(self, action, original_reward):
        """
        Apply fuel consumption penalty by multiplying negative rewards when engines are fired.

        Args:
            action: The action taken (0: noop, 1: main engine, 2: left engine, 3: right engine)
            original_reward: The original reward from the environment

        Returns:
            Modified reward with fuel penalty applied
        """
        # Check if any engine is firing (actions 1, 2, 3)
        if action in [1, 2, 3]:
            # If the original reward is negative (fuel consumption), multiply by penalty factor
            if original_reward < 0:
                # Heavily penalize fuel consumption by 2.5x
                return original_reward * self.fuel_consumption_penalty
            else:
                # If reward is positive, still apply some penalty for using fuel
                return original_reward * 0.8

        return original_reward

    def _calculate_precision_bonus(self, obs):
        """
        Calculate precision landing bonus based on horizontal distance from center.

        Args:
            obs: The observation from the environment

        Returns:
            Bonus reward for precision landing
        """
        # Extract x position from observation (index 0 is x position)
        x_pos = obs[0]

        # Calculate distance from center (0.0 is the center between flags)
        distance_from_center = abs(x_pos)

        # Perfect landing bonus: 100 points for landing at center, decreasing with distance
        # Use exponential decay for smooth bonus calculation
        precision_bonus = 100.0 * np.exp(-5.0 * distance_from_center)

        return precision_bonus

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render(mode)
