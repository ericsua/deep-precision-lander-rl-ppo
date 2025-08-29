import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EfficientLanderEnv(gym.Wrapper):
    """
    Precision-focused custom wrapper for LunarLander-v3 that implements fuel-efficient and precise landing objectives.

    This wrapper implements a precision-focused reward function that:
    1. Encourages fuel efficiency without being too harsh
    2. Rewards progress toward landing
    3. Gives substantial bonuses for successful landings
    4. STRONGLY rewards landing in the center of the objective
    5. Penalizes movement after landing to prevent sliding
    6. Rewards vertical alignment with the center objective
    7. Provides continuous positive feedback for good behavior
    8. Balances exploration vs exploitation for consistent performance
    """

    def __init__(self, env):
        super().__init__(env)
        self.original_env = env

        # Store the original action space for reference
        self.original_action_space = env.action_space

        # Precision-focused reward configuration
        self.fuel_efficiency_bonus = 0.02      # Small bonus for not using fuel
        self.progress_bonus = 0.15             # Moderate progress rewards
        self.landing_bonus = 300.0             # Good landing bonus
        self.center_precision_bonus = 500.0     # Large bonus for center landing
        self.vertical_alignment_bonus = 2.0     # Bonus for vertical alignment
        
        # Landing encouragement parameters
        self.near_landing_zone = False
        self.hovering_penalty = 0.0
        self.time_in_landing_zone = 0
        self.max_hover_time = 80               # Reasonable hovering time
        
        # Performance tracking
        self.episode_steps = 0
        self.total_fuel_used = 0
        self.has_landed = False
        self.landing_position = None
        self.movement_after_landing = 0

    def step(self, action, **kwargs):
        """
        Override the step method to implement precision-focused reward logic.

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

        # Add progress bonus
        progress_bonus = self._calculate_progress_bonus(obs)
        modified_reward += progress_bonus

        # Add vertical alignment bonus (reward for being in line with center)
        vertical_alignment_bonus = self._calculate_vertical_alignment_bonus(obs)
        modified_reward += vertical_alignment_bonus

        # Check if we're in the landing zone and apply landing encouragement
        landing_encouragement = self._calculate_landing_encouragement(obs)
        modified_reward += landing_encouragement

        # Apply landing bonus if episode is terminated successfully
        if terminated and info.get("success", False):
            landing_bonus = self._calculate_landing_bonus(obs)
            modified_reward += landing_bonus
            
            # Additional bonus for fuel efficiency
            if self.total_fuel_used < 80:  # Very fuel efficient
                modified_reward += 100.0
            elif self.total_fuel_used < 150:  # Moderately fuel efficient
                modified_reward += 50.0

        # Add penalty for crashes
        if terminated and not info.get("success", False):
            modified_reward -= 50.0  # Moderate penalty for crashes

        # Add very mild time pressure
        if self.episode_steps > 800:
            modified_reward -= 0.05 * (self.episode_steps - 800)

        return obs, modified_reward, terminated, truncated, info

    def _calculate_progress_bonus(self, obs):
        """
        Calculate progress bonus based on how close the lander is to landing safely.
        
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
        
        # Bonus for being close to the ground
        if 0.1 < y_pos < 1.0:
            progress_bonus += 0.1
        
        # Bonus for being near the center horizontally (more precise)
        if abs(x_pos) < 0.2:
            progress_bonus += 0.1
        elif abs(x_pos) < 0.4:
            progress_bonus += 0.05
        elif abs(x_pos) < 0.6:
            progress_bonus += 0.02
        
        # Bonus for controlled descent
        if abs(y_vel) < 1.0:
            progress_bonus += 0.05
        
        # Bonus for being upright
        if abs(angle) < 0.3:
            progress_bonus += 0.05
        elif abs(angle) < 0.5:
            progress_bonus += 0.02
        
        # Bonus for low angular velocity
        if abs(ang_vel) < 0.4:
            progress_bonus += 0.05
        
        return progress_bonus

    def _calculate_vertical_alignment_bonus(self, obs):
        """
        Calculate bonus for being vertically aligned with the center objective.
        
        Args:
            obs: The observation from the environment
            
        Returns:
            Vertical alignment bonus reward
        """
        x_pos, y_pos = obs[0], obs[1]
        
        # Reward for being vertically above the center (x close to 0)
        # This encourages the agent to approach from above the center
        if y_pos > 0.5:  # Only when high enough to be approaching
            center_alignment = max(0, 1.0 - abs(x_pos) * 2.0)  # 1.0 at center, 0.0 at edges
            alignment_bonus = self.vertical_alignment_bonus * center_alignment
            
            # Extra bonus for being very close to center line
            if abs(x_pos) < 0.1:
                alignment_bonus += 0.5
            
            return alignment_bonus
        
        return 0.0

    def _calculate_landing_encouragement(self, obs):
        """
        Calculate reward to encourage precise landing completion.
        
        Args:
            obs: The observation from the environment
            
        Returns:
            Landing encouragement reward
        """
        x_pos, y_pos = obs[0], obs[1]
        x_vel, y_vel = obs[2], obs[3]
        left_leg, right_leg = obs[6], obs[7]
        
        encouragement = 0.0
        
        # Check if we're in the landing zone (more precise)
        in_landing_zone = (abs(x_pos) < 0.3 and y_pos < 0.5)
        
        if in_landing_zone:
            # Track time in landing zone
            if not self.near_landing_zone:
                self.near_landing_zone = True
                self.hovering_penalty = 0.0
                self.time_in_landing_zone = 0
            
            self.time_in_landing_zone += 1
            
            # Bonus for being very close to the ground
            if y_pos < 0.15:
                encouragement += 1.5
            
            # Bonus for having legs near the ground
            if left_leg or right_leg:
                encouragement += 3.0
            
            # Bonus for very slow descent (controlled landing)
            if abs(y_vel) < 0.5:
                encouragement += 0.8
            
            # Progressive penalty for hovering too long
            if y_pos > 0.08:  # Still hovering above ground
                if self.time_in_landing_zone > self.max_hover_time:
                    self.hovering_penalty += 0.1
                    encouragement -= self.hovering_penalty
                else:
                    # Small bonus for controlled hovering
                    encouragement += 0.05
            
            # Strong bonus for actually touching the ground
            if y_pos <= 0.05:
                encouragement += 15.0
                # Reset hovering penalty on successful landing
                self.hovering_penalty = 0.0
                
                # Track landing position to prevent movement after landing
                if not self.has_landed:
                    self.has_landed = True
                    self.landing_position = x_pos
                else:
                    # Penalize movement after landing
                    if abs(x_pos - self.landing_position) > 0.01:
                        self.movement_after_landing += 1
                        encouragement -= 2.0 * self.movement_after_landing  # Increasing penalty
        else:
            # Reset when leaving landing zone
            self.near_landing_zone = False
            self.hovering_penalty = 0.0
            self.time_in_landing_zone = 0
        
        return encouragement

    def _calculate_landing_bonus(self, obs):
        """
        Calculate precision-focused landing bonus based on center alignment.
        
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
        
        # PRECISION BONUS: Heavily reward landing in the center
        distance_from_center = abs(x_pos)
        
        # Exponential precision bonus - much higher for center landings
        if distance_from_center < 0.05:  # Perfect center landing
            precision_bonus = self.center_precision_bonus
        elif distance_from_center < 0.1:  # Very close to center
            precision_bonus = self.center_precision_bonus * 0.8
        elif distance_from_center < 0.2:  # Close to center
            precision_bonus = self.center_precision_bonus * 0.5
        elif distance_from_center < 0.3:  # Reasonable landing
            precision_bonus = self.center_precision_bonus * 0.2
        else:  # Far from center
            precision_bonus = 0.0
        
        # Safety bonus for landing gently
        if y_pos > 0.05:  # Not too close to ground
            safety_bonus = 75.0
        else:
            safety_bonus = 50.0
        
        # Speed bonus for completing quickly
        speed_bonus = max(0, 30 - self.episode_steps * 0.02)
        
        # Penalty for movement after landing
        movement_penalty = 0.0
        if self.has_landed and self.movement_after_landing > 0:
            movement_penalty = 50.0 * self.movement_after_landing
        
        total_bonus = landing_bonus + precision_bonus + safety_bonus + speed_bonus - movement_penalty
        
        return total_bonus

    def reset(self, **kwargs):
        """Reset the environment and tracking variables."""
        self.near_landing_zone = False
        self.hovering_penalty = 0.0
        self.time_in_landing_zone = 0
        self.episode_steps = 0
        self.total_fuel_used = 0
        self.has_landed = False
        self.landing_position = None
        self.movement_after_landing = 0
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render(mode)
