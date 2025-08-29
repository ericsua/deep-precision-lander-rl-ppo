import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EfficientLanderEnv(gym.Wrapper):
    """
    Balanced custom wrapper for LunarLander-v3 that implements fuel-efficient and precise landing objectives.

    This wrapper implements a forgiving but effective reward function that:
    1. Encourages fuel efficiency without being too harsh
    2. Rewards progress toward landing
    3. Gives substantial bonuses for successful landings
    4. Provides continuous positive feedback for good behavior
    5. Encourages actual landing completion (not just hovering)
    6. Balances exploration vs exploitation for consistent performance
    7. Is forgiving of mistakes while rewarding good behavior
    """

    def __init__(self, env):
        super().__init__(env)
        self.original_env = env

        # Store the original action space for reference
        self.original_action_space = env.action_space

        # Balanced reward configuration - more forgiving
        self.fuel_efficiency_bonus = 0.02      # Small bonus for not using fuel
        self.progress_bonus = 0.2              # Moderate progress rewards
        self.landing_bonus = 200.0             # Good landing bonus
        self.precision_multiplier = 1.2        # Moderate precision multiplier
        
        # Landing encouragement parameters - more forgiving
        self.near_landing_zone = False
        self.hovering_penalty = 0.0
        self.time_in_landing_zone = 0
        self.max_hover_time = 100              # More generous hovering time
        
        # Performance tracking
        self.episode_steps = 0
        self.total_fuel_used = 0

    def step(self, action, **kwargs):
        """
        Override the step method to implement balanced reward logic.

        Args:
            action: The action taken by the agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Increment episode steps
        self.episode_steps += 1
        
        # Track fuel usage
        if action in [1, 2, 3]:  # Main engine, left engine, right engine
            self.total_fuel_used += 1
        
        # Take the original step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Start with the original reward
        modified_reward = reward

        # Add small fuel efficiency bonus
        if action == 0:  # No operation
            modified_reward += self.fuel_efficiency_bonus

        # Add moderate progress bonus
        progress_bonus = self._calculate_progress_bonus(obs)
        modified_reward += progress_bonus

        # Check if we're in the landing zone and apply landing encouragement
        landing_encouragement = self._calculate_landing_encouragement(obs)
        modified_reward += landing_encouragement

        # Apply landing bonus if episode is terminated successfully
        if terminated and info.get("success", False):
            landing_bonus = self._calculate_landing_bonus(obs)
            modified_reward += landing_bonus
            
            # Additional bonus for fuel efficiency
            if self.total_fuel_used < 80:  # Very fuel efficient
                modified_reward += 50.0
            elif self.total_fuel_used < 150:  # Moderately fuel efficient
                modified_reward += 25.0

        # Add moderate penalty for crashes
        if terminated and not info.get("success", False):
            modified_reward -= 30.0  # Reduced penalty for crashes

        # Add very mild time pressure (encourage faster completion)
        if self.episode_steps > 800:
            modified_reward -= 0.05 * (self.episode_steps - 800)

        return obs, modified_reward, terminated, truncated, info

    def _calculate_progress_bonus(self, obs):
        """
        Calculate moderate progress bonus based on how close the lander is to landing safely.
        
        Args:
            obs: The observation from the environment
            
        Returns:
            Progress bonus reward
        """
        # Extract relevant observations
        x_pos, y_pos = obs[0], obs[1]           # Position
        x_vel, y_vel = obs[2], obs[3]           # Velocity
        angle, ang_vel = obs[4], obs[5]         # Angle and angular velocity
        left_leg, right_leg = obs[6], obs[7]    # Leg contact
        
        progress_bonus = 0.0
        
        # Moderate bonus for being close to the ground
        if 0.1 < y_pos < 1.0:
            progress_bonus += 0.1
        
        # Bonus for being near the center horizontally
        if abs(x_pos) < 0.4:
            progress_bonus += 0.05
        elif abs(x_pos) < 0.6:
            progress_bonus += 0.02
        
        # Bonus for controlled descent
        if abs(y_vel) < 1.2:
            progress_bonus += 0.05
        
        # Bonus for being upright
        if abs(angle) < 0.4:
            progress_bonus += 0.05
        elif abs(angle) < 0.6:
            progress_bonus += 0.02
        
        # Bonus for low angular velocity
        if abs(ang_vel) < 0.5:
            progress_bonus += 0.05
        
        return progress_bonus

    def _calculate_landing_encouragement(self, obs):
        """
        Calculate moderate reward to encourage actual landing completion.
        
        Args:
            obs: The observation from the environment
            
        Returns:
            Landing encouragement reward
        """
        x_pos, y_pos = obs[0], obs[1]
        x_vel, y_vel = obs[2], obs[3]
        left_leg, right_leg = obs[6], obs[7]
        
        encouragement = 0.0
        
        # Check if we're in the landing zone (more generous)
        in_landing_zone = (abs(x_pos) < 0.4 and y_pos < 0.6)
        
        if in_landing_zone:
            # Track time in landing zone
            if not self.near_landing_zone:
                self.near_landing_zone = True
                self.hovering_penalty = 0.0
                self.time_in_landing_zone = 0
            
            self.time_in_landing_zone += 1
            
            # Bonus for being very close to the ground
            if y_pos < 0.2:
                encouragement += 1.0
            
            # Bonus for having legs near the ground
            if left_leg or right_leg:
                encouragement += 2.0
            
            # Bonus for very slow descent (controlled landing)
            if abs(y_vel) < 0.6:
                encouragement += 0.5
            
            # Very mild penalty for hovering too long
            if y_pos > 0.1:  # Still hovering above ground
                if self.time_in_landing_zone > self.max_hover_time:
                    self.hovering_penalty += 0.05  # Much smaller penalty
                    encouragement -= self.hovering_penalty
                else:
                    # Small bonus for controlled hovering
                    encouragement += 0.05
            
            # Good bonus for actually touching the ground
            if y_pos <= 0.05:
                encouragement += 10.0
                # Reset hovering penalty on successful landing
                self.hovering_penalty = 0.0
        else:
            # Reset when leaving landing zone
            self.near_landing_zone = False
            self.hovering_penalty = 0.0
            self.time_in_landing_zone = 0
        
        return encouragement

    def _calculate_landing_bonus(self, obs):
        """
        Calculate balanced landing bonus based on precision and safety.
        
        Args:
            obs: The observation from the environment
            
        Returns:
            Landing bonus reward
        """
        # Extract position for precision calculation
        x_pos = obs[0]
        y_pos = obs[1]
        
        # Base landing bonus
        landing_bonus = self.landing_bonus
        
        # Precision bonus based on horizontal distance from center
        distance_from_center = abs(x_pos)
        precision_bonus = self.landing_bonus * 0.3 * np.exp(-2.5 * distance_from_center)
        
        # Safety bonus for landing gently
        if y_pos > 0.05:  # Not too close to ground
            safety_bonus = 50.0
        else:
            safety_bonus = 25.0
        
        # Small speed bonus for completing quickly
        speed_bonus = max(0, 25 - self.episode_steps * 0.02)
        
        total_bonus = landing_bonus + precision_bonus + safety_bonus + speed_bonus
        
        return total_bonus

    def reset(self, **kwargs):
        """Reset the environment and tracking variables."""
        self.near_landing_zone = False
        self.hovering_penalty = 0.0
        self.time_in_landing_zone = 0
        self.episode_steps = 0
        self.total_fuel_used = 0
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render(mode)
