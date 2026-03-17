import sys
try:
    import pygame
except ImportError:
    import pygame_ce as pygame
    sys.modules['pygame'] = pygame
import random

class Track:
    """
    Road track with obstacles (walls, boundaries)
    """
    def __init__(self, width=800, height=600, randomize=False):
        self.randomize = randomize
        self.width = width
        self.height = height
        self.obstacles = []
        self.checkpoints = []
        self.create_track()
        
    def create_track(self):
        """Create a vertical single-lane road with obstacles"""
        lane_x_left = 300
        lane_x_right = 500
        lane_width = lane_x_right - lane_x_left
        sim_h = self.height
        
        # Road boundaries (Left and Right walls)
        self.obstacles = [
            pygame.Rect(0, 0, lane_x_left, sim_h),  # Left "off-road"
            pygame.Rect(lane_x_right, 0, self.width - lane_x_right, sim_h),  # Right "off-road"
        ]
        
        # Obstacles inside the lane
        if getattr(self, 'randomize', False):
            self.obstacles.extend(self.generate_random_obstacles(lane_x_left, lane_x_right))
        else:
            self.obstacles.extend([
                pygame.Rect(lane_x_left + 20, 500, 60, 30),
                pygame.Rect(lane_x_right - 80, 350, 60, 30),
                pygame.Rect(lane_x_left + 50, 200, 40, 40),
                pygame.Rect(lane_x_right - 100, 50, 80, 30),
            ])
            
        # Checkpoints along the lane (moving upwards)
        self.checkpoints = []
        for y in range(sim_h - 100, 0, -100):
            self.checkpoints.append(pygame.Rect(lane_x_left, y, lane_width, 20))

    def generate_random_obstacles(self, left_x, right_x):
        """Generate random obstacles avoiding the start zone"""
        obs = []
        for _ in range(random.randint(3, 6)):
            x = random.randint(left_x + 10, right_x - 60)
            y = random.randint(50, 550) # Avoid very bottom (start)
            w = random.randint(30, 60)
            h = random.randint(30, 60)
            obs.append(pygame.Rect(x, y, w, h))
        return obs
        
    def draw(self, screen):
        """Draw track with game-like visuals"""
        # Grass background
        screen.fill((34, 139, 34)) # Forest Green
        
    def draw(self, screen):
        """Draw vertical track with game-like visuals"""
        # Grass background
        screen.fill((34, 139, 34)) # Forest Green
        
        # Outer border (Grass borders)
        # Left grass
        pygame.draw.rect(screen, (34, 139, 34), (0, 0, 300, self.height))
        # Right grass
        pygame.draw.rect(screen, (34, 139, 34), (500, 0, self.width - 500, self.height))
        
        # Asphalt (Main Road area)
        pygame.draw.rect(screen, (40, 40, 40), (300, 0, 200, self.height))
        
        # Lane markings (dashed white lines) - Vertical
        lane_color = (200, 200, 200)
        for y in range(0, self.height, 40):
            pygame.draw.line(screen, lane_color, (400, y), (400, y + 20), 2)
            
        # Draw road shoulders (white lines)
        pygame.draw.line(screen, (255, 255, 255), (300, 0), (300, self.height), 3)
        pygame.draw.line(screen, (255, 255, 255), (500, 0), (500, self.height), 3)

        # Obstacles (walls with a more concrete look)
        for obstacle in self.obstacles:
            # Concrete wall color
            pygame.draw.rect(screen, (100, 100, 110), obstacle)
            # Wall top detail (inner shadow)
            pygame.draw.rect(screen, (120, 120, 130), 
                            (obstacle.x + 2, obstacle.y + 2, obstacle.width - 4, obstacle.height - 4))
            pygame.draw.rect(screen, (40, 40, 50), obstacle, 2)
            
        # Checkpoints (Hidden - no longer drawn)
        # for checkpoint in self.checkpoints:
        #     pygame.draw.rect(screen, (0, 255, 100), checkpoint, 1)
