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
    9. ENDS EPISODE IMMEDIATELY when touching ground (consistent behavior)
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
        
        # Episode control
        self.episode_terminated = False
        self.episode_truncated = False

    def step(self, action, **kwargs):
        """
        Override the step method to implement precision-focused reward logic with controlled episode termination.

        Args:
            action: The action taken by the agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # If we've already terminated, return the final state
        if self.episode_terminated or self.episode_truncated:
            return self.last_obs, 0.0, True, False, self.last_info
        
        # Increment episode steps
        self.episode_steps += 1
        
        # Track fuel usage
        if action in [1, 2, 3]:  # Main engine, left engine, right engine
            self.total_fuel_used += 1
        
        # Take the original step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store for potential reuse
        self.last_obs = obs
        self.last_info = info
        
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

        # CRITICAL: Check if we've touched the ground and should end episode
        x_pos, y_pos = obs[0], obs[1]
        left_leg, right_leg = obs[6], obs[7]
        
        # Check if we've touched the ground (either through position or leg contact)
        # More strict ground detection to ensure episode ends
        touched_ground = (y_pos <= 0.05) or (left_leg and right_leg and y_pos <= 0.08)
        
        if touched_ground and not self.episode_terminated:
            # We've touched the ground - end the episode immediately
            self.episode_terminated = True
            
            # Calculate final landing bonus
            landing_bonus = self._calculate_landing_bonus(obs)
            modified_reward += landing_bonus
            
            # Additional bonus for fuel efficiency
            if self.total_fuel_used < 80:  # Very fuel efficient
                modified_reward += 100.0
            elif self.total_fuel_used < 150:  # Moderately fuel efficient
                modified_reward += 50.0
            
            # Determine if it's a successful landing (in landing zone)
            in_landing_zone = (abs(x_pos) < 0.3 and y_pos < 0.5)
            if in_landing_zone:
                info['success'] = True
                info['landing_type'] = 'successful'
            else:
                info['success'] = False
                info['landing_type'] = 'outside_zone'
            
            # Add penalty for crashes (if not in landing zone)
            if not in_landing_zone:
                modified_reward -= 50.0  # Moderate penalty for landing outside
            
            # Force episode termination
            self.episode_terminated = True
            return obs, modified_reward, True, False, info
        
        # If episode was terminated by the original environment, handle it
        if terminated:
            # This means the original environment ended the episode
            # Apply landing bonus if it was successful
            if info.get('success', False):
                landing_bonus = self._calculate_landing_bonus(obs)
                modified_reward += landing_bonus
                
                # Additional bonus for fuel efficiency
                if self.total_fuel_used < 80:
                    modified_reward += 100.0
                elif self.total_fuel_used < 150:
                    modified_reward += 50.0
            else:
                # Crashed or other failure
                modified_reward -= 50.0
            
            # Mark as terminated to prevent further steps
            self.episode_terminated = True
        
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
        self.episode_terminated = False
        self.episode_truncated = False
        self.last_obs = None
        self.last_info = None
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        """Render the environment with custom UI overlay."""
        # Get the original render
        render_result = self.env.render(mode)
        
        # Add custom UI overlay if we have a surface to draw on
        if mode == "human" and hasattr(self.env, 'viewer') and self.env.viewer is not None:
            try:
                # Try to get the pygame surface from the viewer
                if hasattr(self.env.viewer, 'window'):
                    surface = self.env.viewer.window
                    self._draw_ui_overlay(surface)
                elif hasattr(self.env.viewer, 'surf'):
                    surface = self.env.viewer.surf
                    self._draw_ui_overlay(surface)
            except Exception as e:
                # If UI overlay fails, just continue with normal rendering
                pass
        
        return render_result
    
    def _draw_ui_overlay(self, surface):
        """Draw custom UI overlay on the pygame surface."""
        try:
            import pygame
            
            # Get current observation for UI data
            if hasattr(self, 'last_obs') and self.last_obs is not None:
                obs = self.last_obs
            else:
                return
            
            # Extract observation components
            x_pos, y_pos = obs[0], obs[1]
            x_vel, y_vel = obs[2], obs[3]
            
            # Calculate UI metrics
            remaining_fuel = 1000 - self.total_fuel_used
            distance_from_center = abs(x_pos)
            total_speed = np.sqrt(x_vel**2 + y_vel**2)
            
            # Font setup
            font_size = 16
            try:
                font = pygame.font.Font(None, font_size)
            except:
                font = pygame.font.Font(None, 24)  # Fallback font size
            
            # Colors
            white = (255, 255, 255)
            green = (0, 255, 0)
            yellow = (255, 255, 0)
            red = (255, 0, 0)
            blue = (0, 100, 255)
            black = (0, 0, 0)
            
            # UI Elements
            ui_elements = []
            
            # 1. Fuel Gauge (top left)
            fuel_color = green if remaining_fuel > 700 else yellow if remaining_fuel > 300 else red
            fuel_text = f"Fuel: {remaining_fuel}/1000"
            fuel_surface = font.render(fuel_text, True, fuel_color, black)
            ui_elements.append((fuel_surface, (10, 10)))
            
            # 2. Speed Indicator (top right)
            speed_color = green if total_speed < 0.5 else yellow if total_speed < 1.0 else red
            speed_text = f"Speed: {total_speed:.2f}"
            speed_surface = font.render(speed_text, True, speed_color, black)
            speed_rect = speed_surface.get_rect()
            speed_rect.topright = (surface.get_width() - 10, 10)
            ui_elements.append((speed_surface, speed_rect))
            
            # 3. Landing Precision (center top)
            if y_pos > 0.5:
                precision_text = f"Distance: {distance_from_center:.3f}"
                precision_color = green if distance_from_center < 0.1 else yellow if distance_from_center < 0.2 else red
            else:
                if distance_from_center < 0.05:
                    precision_text = "PERFECT LANDING!"
                    precision_color = green
                elif distance_from_center < 0.1:
                    precision_text = "Excellent Landing"
                    precision_color = green
                elif distance_from_center < 0.2:
                    precision_text = "Good Landing"
                    precision_color = yellow
                elif distance_from_center < 0.3:
                    precision_text = "Poor Landing"
                    precision_color = red
                else:
                    precision_text = "Outside Zone!"
                    precision_color = red
            
            precision_surface = font.render(precision_text, True, precision_color, black)
            precision_rect = precision_surface.get_rect()
            precision_rect.centerx = surface.get_width() // 2
            precision_rect.y = 10
            ui_elements.append((precision_surface, precision_rect))
            
            # 4. Position Info (bottom left)
            pos_text = f"Pos: ({x_pos:.2f}, {y_pos:.2f})"
            pos_surface = font.render(pos_text, True, blue, black)
            ui_elements.append((pos_surface, (10, surface.get_height() - 30)))
            
            # 5. Episode Info (bottom right)
            episode_text = f"Steps: {self.episode_steps}"
            episode_surface = font.render(episode_text, True, white, black)
            episode_rect = episode_surface.get_rect()
            episode_rect.bottomright = (surface.get_width() - 10, surface.get_height() - 10)
            ui_elements.append((episode_surface, episode_rect))
            
            # Draw all UI elements
            for element_surface, position in ui_elements:
                if isinstance(position, pygame.Rect):
                    surface.blit(element_surface, position)
                else:
                    surface.blit(element_surface, position)
                    
        except Exception as e:
            # If anything goes wrong with UI rendering, just continue
            pass
