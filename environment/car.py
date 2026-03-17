import numpy as np
import sys
try:
    import pygame
except ImportError:
    import pygame_ce as pygame
    sys.modules['pygame'] = pygame
import math

class Car:
    """
    Car with physics simulation and distance sensors (like LIDAR)
    """
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.acceleration = 0
        
        # Car properties
        self.max_speed = 10
        self.friction = 0.95
        self.turn_speed = 5
        
        # Dimensions
        self.width = 40
        self.height = 20
        
        # Sensors (5 distance sensors like LIDAR)
        self.sensor_angles = [-60, -30, 0, 30, 60]
        self.sensor_length = 300
        self.sensor_readings = [0] * len(self.sensor_angles)
        
        # State
        self.alive = True
        self.distance_traveled = 0
        self.collision = False
        
    def update(self, action, obstacles):
        """
        Update car state based on action
        Actions: 0=nothing, 1=forward, 2=left, 3=right, 4=brake
        """
        if not self.alive:
            return
            
        # Apply action
        if action == 1:  # Forward
            self.acceleration = 0.4 # Relaxed pace
        elif action == 2:  # Left
            self.angle -= self.turn_speed
            self.acceleration = 0.2 # Auto-gas when turning
        elif action == 3:  # Right
            self.angle += self.turn_speed
            self.acceleration = 0.2 # Auto-gas when turning
        elif action == 4:  # Brake
            self.acceleration = -0.6
        else:  # Nothing
            self.acceleration = -0.05 # Very slight drift
            
        # Update speed
        self.speed += self.acceleration
        self.speed *= self.friction
        # Removed minimum threshold for more natural slow movement
        self.speed = max(2.0, min(self.speed, self.max_speed))
        
        # Update position
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
        
        self.distance_traveled += abs(self.speed)
        
        # Update sensors
        self.update_sensors(obstacles)
        
        # Check collision with more precision
        self.check_collision(obstacles)
        
    def update_sensors(self, obstacles):
        """
        Cast rays to detect obstacles (ML input)
        """
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = math.radians(self.angle + sensor_angle)
            
            # Cast ray
            for distance in range(0, self.sensor_length, 5):
                x = self.x + distance * math.cos(angle)
                y = self.y + distance * math.sin(angle)
                
                # Check if ray hits obstacle
                if self.point_in_obstacles(x, y, obstacles):
                    self.sensor_readings[i] = distance / self.sensor_length
                    break
            else:
                self.sensor_readings[i] = 1.0  # Max distance
                
    def point_in_obstacles(self, x, y, obstacles):
        """Check if point collides with obstacles"""
        for obstacle in obstacles:
            if obstacle.collidepoint(x, y):
                return True
        return False
        
    def check_collision(self, obstacles):
        """Check if any corner of the car collides with obstacles"""
        points = self.get_car_corners()
        for px, py in points:
            if self.point_in_obstacles(px, py, obstacles):
                self.alive = False
                self.collision = True
                return
                
    def get_rect(self):
        """Get car bounding box (simplified)"""
        return pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )
        
    def get_state(self):
        """
        Get state vector for ML model
        Returns: normalized sensor readings + speed + angle features
        """
        state = list(self.sensor_readings)
        state.append(self.speed / self.max_speed)
        state.append(math.sin(math.radians(self.angle)))
        state.append(math.cos(math.radians(self.angle)))
        return np.array(state, dtype=np.float32)
        
    def draw(self, screen):
        """Draw a more detailed car model"""
        if not self.alive:
            main_color = (200, 50, 50)  # Dull Red if crashed
            detail_color = (150, 40, 40)
        else:
            main_color = (0, 120, 255)  # Professional Blue
            detail_color = (0, 90, 200)
            
        # Get rotated corners for the main body
        points = self.get_car_corners()
        
        # 1. Draw Wheels
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        wheel_w, wheel_h = 10, 6
        wheel_offsets = [
            (-self.width//2 + 8, -self.height//2 - 2), # Top Left
            (self.width//2 - 12, -self.height//2 - 2), # Top Right
            (-self.width//2 + 8, self.height//2 - 4),  # Bottom Left
            (self.width//2 - 12, self.height//2 - 4)   # Bottom Right
        ]
        
        for ox, oy in wheel_offsets:
            # Rotate wheel offset
            rx = ox * cos_a - oy * sin_a
            ry = ox * sin_a + oy * cos_a
            wheel_rect = pygame.Rect(0, 0, wheel_w, wheel_h)
            wheel_surf = pygame.Surface((wheel_w, wheel_h), pygame.SRCALPHA)
            pygame.draw.rect(wheel_surf, (20, 20, 20), (0, 0, wheel_w, wheel_h), border_radius=2)
            # Rotate the wheel surface itself to match car angle
            rot_wheel = pygame.transform.rotate(wheel_surf, -self.angle)
            screen.blit(rot_wheel, (self.x + rx - rot_wheel.get_width()//2, self.y + ry - rot_wheel.get_height()//2))

        # 2. Draw Car Body
        pygame.draw.polygon(screen, main_color, points)
        pygame.draw.polygon(screen, detail_color, points, 2) # Outline
        
        # 3. Draw Windshield
        # Calculate windshield points (front part of the car)
        windshield_ox = self.width // 4
        windshield_h = self.height - 6
        w_points = [
            (windshield_ox, -windshield_h//2),
            (windshield_ox + 8, -windshield_h//2),
            (windshield_ox + 8, windshield_h//2),
            (windshield_ox, windshield_h//2)
        ]
        rot_w_points = []
        for ox, oy in w_points:
            rx = ox * cos_a - oy * sin_a
            ry = ox * sin_a + oy * cos_a
            rot_w_points.append((self.x + rx, self.y + ry))
        
        pygame.draw.polygon(screen, (100, 200, 255), rot_w_points) # Light blue glass
        
        # 4. Draw Headlights (at the front)
        if self.alive:
            light_ox = self.width // 2 - 2
            light_oy = self.height // 2 - 4
            for side in [-1, 1]:
                lx = light_ox * cos_a - (side * light_oy) * sin_a
                ly = light_ox * sin_a + (side * light_oy) * cos_a
                pygame.draw.circle(screen, (255, 255, 200), (int(self.x + lx), int(self.y + ly)), 3)
                # Subtle beam effect
                s = pygame.Surface((20, 20), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 255, 100, 50), (10, 10), 10)
                screen.blit(s, (self.x + lx - 10, self.y + ly - 10))

        # 5. Draw Sensors (Keep them green but subtle)
        if self.alive:
            for i, sensor_angle in enumerate(self.sensor_angles):
                angle = math.radians(self.angle + sensor_angle)
                distance = self.sensor_readings[i] * self.sensor_length
                end_x = self.x + distance * math.cos(angle)
                end_y = self.y + distance * math.sin(angle)
                pygame.draw.line(screen, (0, 255, 0, 100), (self.x, self.y), (end_x, end_y), 1)
                
    def get_car_corners(self):
        """Get car corner points for drawing"""
        rad = math.radians(self.angle)
        
        # Calculate corners
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        dx = self.width / 2
        dy = self.height / 2
        
        corners = [
            (-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)
        ]
        
        rotated = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            rotated.append((self.x + rx, self.y + ry))
            
        return rotated
