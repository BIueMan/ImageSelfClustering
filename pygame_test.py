import pygame
import numpy as np
from funcrions.pygame_label import *

# Initialize Pygame
pygame.init()

# Set the size of the Pygame window
window_size = (400, 400)

# Create the Pygame window
screen = pygame.display.set_mode(window_size)

# Generate a random image array
n = 4
m = 5
x = 50
y = 50
color = 3
image_array = np.random.randint(0, 255, size=(n, m, x, y, color)).astype(np.uint8)

# Display the big image
display_big_image(image_array, screen)

# Update the display
pygame.display.flip()

# Wait for the user to close the window
while True:
    event = pygame.event.wait()
    if event.type == pygame.QUIT:
        break

# Quit Pygame
pygame.quit()
