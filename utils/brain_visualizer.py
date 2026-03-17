import sys
try:
    import pygame
except ImportError:
    import pygame_ce as pygame
    sys.modules['pygame'] = pygame
import numpy as np

class BrainVisualizer:
    """
    Visualizes the neural network structure and real-time activations.
    """
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.node_radius = 6
        self.layer_gap = width // 6
        self.font = pygame.font.Font(None, 24)
        
    def draw(self, screen, activations):
        """
        Draw a simplified network architecture with clear activations.
        """
        if not activations:
            return

        # Background for the brain panel
        pygame.draw.rect(screen, (15, 15, 20), self.rect)
        pygame.draw.rect(screen, (0, 255, 150), self.rect, 1) # Thin green border
        
        # Title
        title = self.font.render("NEURAL NETWORK BRAIN", True, (0, 255, 150))
        screen.blit(title, (self.rect.x + 20, self.rect.y + 20))

        # We show 3 key layers: Input, Hidden (combined), Output
        # Mapping activations to a smaller subset for "Understandable" UI
        num_layers = len(activations)
        display_layers = []
        
        # Input Layer (Sensors)
        display_layers.append(activations[0][0][:8]) # First 8 nodes
        # Middle Layer (Representing the 'thinking')
        mid_idx = num_layers // 2
        display_layers.append(activations[mid_idx][0][:6]) # Repr 6 nodes
        # Output Layer (Actions)
        display_layers.append(activations[-1][0]) # All actions

        layer_positions = []
        layer_names = ["SENSORS", "THINKING", "DECISION"]
        
        for i, layer_act in enumerate(display_layers):
            layer_x = self.rect.x + 80 + i * (self.rect.width // 3.5)
            num_nodes = len(layer_act)
            
            node_gap = 40
            total_h = (num_nodes - 1) * node_gap
            start_y = self.rect.y + (self.rect.height // 2) - (total_h // 2) + 20
            
            nodes = []
            for j in range(num_nodes):
                node_y = start_y + j * node_gap
                val = layer_act[j]
                # Scale ReLu activations for intuitive brightness
                # 0 is dim, >0 is bright
                brightness = min(255, max(40, int(val * 255))) 
                nodes.append(((layer_x, node_y), brightness))
            
            layer_positions.append(nodes)
            
            # Layer Labels
            name_surf = self.font.render(layer_names[i], True, (100, 100, 110))
            screen.blit(name_surf, (layer_x - name_surf.get_width()//2, self.rect.y + self.rect.height - 40))

        # Draw Clean Connections
        for i in range(len(layer_positions) - 1):
            for start_pos, start_bright in layer_positions[i]:
                for end_pos, end_bright in layer_positions[i+1]:
                    # Only draw if nodes are active
                    if start_bright > 60:
                        alpha = min(150, (start_bright + end_bright) // 4)
                        pygame.draw.line(screen, (0, 255, 150, alpha), start_pos, end_pos, 1)

        # Draw Nodes
        for nodes in layer_positions:
            for pos, brightness in nodes:
                # Activation Glow
                if brightness > 100:
                    pygame.draw.circle(screen, (0, 255, 150, 50), pos, 10)
                # Neuron
                color = (brightness, brightness, brightness) if brightness < 100 else (0, brightness, 150)
                pygame.draw.circle(screen, color, pos, 6)
                pygame.draw.circle(screen, (0, 255, 150), pos, 6, 1) # Outline
