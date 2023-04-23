import pygame
import numpy as np
import cv2
import colorsys

def display_big_image(image_array, screen):
    # rotate image for plot / pygame axis are yx and not xy
    image_array = np.rot90(image_array.copy(), k=1)
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            image_array[i,j,:,:,:] = np.rot90(image_array[i,j,:,:,:], k=1)

    # Extract dimensions of image_array
    grid_width = image_array.shape[0]
    grid_height = image_array.shape[1]
    small_width = image_array.shape[2]
    small_height = image_array.shape[3]
    big_width = small_width * grid_width
    big_height = small_height * grid_height
    
    # Create a surface for the big image
    big_image = pygame.Surface((big_width, big_height))
    
    # Loop through the small images and blit them onto the big image
    for i in range(grid_width):
        for j in range(grid_height):
            small_image = image_array[i,j,:,:,:]
            small_image = np.flip(small_image, axis=-1)
            big_image.blit(pygame.surfarray.make_surface(small_image), (i*small_width, j*small_height))
    
    # Blit the big image onto the screen
    screen.blit(big_image, (0,0))

def select_filter_ui(small_splited, screen, class_num = 1, init_sigment = None):
    # params
    filter_britnes = 0.75

    # create the filter
    def get_color_range(n):
        hue_values = [i*(360/n) for i in range(n)]
        colors = [colorsys.hls_to_rgb(h/360, 0.5, 1.0) for h in hue_values]
        return colors
    filter_list = np.vstack([np.array([0,0,0]), 
                             (np.array(get_color_range(class_num))*filter_britnes*255).astype(np.uint8)])
    filter_list = np.flip(filter_list, axis=1) # flip color (pygame work with bgr and not rgb) 
    selected_filter = 1

    # create text for bottuns
    end_of_image_loc = small_splited.shape[1]*small_splited.shape[2]
    text_size = 20
    font = pygame.font.SysFont("Arial", 18)
    text = font.render("Select Filter:", True, (255, 255, 255))
    screen.blit(text, (end_of_image_loc, 0))

    # create buttons
    button_rect_list = []
    SPACE_BETWEEN_BUTTONS = 20
    for i in range(filter_list.shape[0]):
        ## create button
        x = end_of_image_loc
        y = text_size + i * SPACE_BETWEEN_BUTTONS
        w = 100 
        h = SPACE_BETWEEN_BUTTONS
        button_rect_list.append(pygame.Rect(x,y,w,h))

        ## plot button with color
        def create_button(rect, color):
            color = np.flip(color)
            pygame.draw.rect(screen, color, rect) # draw color
            pygame.draw.rect(screen, (255, 255 ,255), rect, 2) # draw frame
        create_button(button_rect_list[-1] ,filter_list[i,:])
    pygame.draw.rect(screen, (0,   0 ,  255), button_rect_list[selected_filter], 2)

    # init loop
    filtered_small_splited = small_splited.copy()
    if init_sigment is None:
        flipped_images = np.zeros([small_splited.shape[0], small_splited.shape[1]])
    else:
        flipped_images = init_sigment
        for (i, j), value in np.ndenumerate(flipped_images):
            #### add filter
            filtered_small_splited[i, j, :, :, :] = np.clip(small_splited[i, j, :, :, :].copy() + filter_list[value], 0, 255).astype(np.uint8)
    display_big_image(filtered_small_splited, screen)
    pygame.display.flip()
    ## loop
    flag_exit = False
    mouse_hold_button = False
    while not flag_exit:
        for event in pygame.event.get(): # wait for mouse event
            ### if click exit
            if event.type == pygame.QUIT:
                flag_exit = True
                break
            ### if click mouse been hold
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_hold_button = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_hold_button = False
            
            ### if mouse been hold
            if mouse_hold_button:
                pos = pygame.mouse.get_pos()
                ### check if click on buttons
                for ii in range(len(button_rect_list)):
                    if button_rect_list[ii].collidepoint(pos):
                        pygame.draw.rect(screen, (255, 255 ,255), button_rect_list[selected_filter], 2)
                        pygame.draw.rect(screen, (0  ,   0 ,255), button_rect_list[ii], 2)
                        selected_filter = ii
                        pygame.display.flip()
                        break

                ### check if click out of image
                if (pos[0] > small_splited.shape[1]*small_splited.shape[2] or 
                    pos[1] > small_splited.shape[0]*small_splited.shape[3]):
                    continue

                ### Get the index of the clicked small image
                i = pos[1] // small_splited.shape[2]
                j = small_splited.shape[1] - pos[0] // small_splited.shape[3] - 1

                ### filter image
                if flipped_images[i,j] != selected_filter:
                    #### add filter
                    filtered_small_splited[i, j, :, :, :] = np.clip(small_splited[i, j, :, :, :].copy() + filter_list[selected_filter], 0, 255).astype(np.uint8)
                    flipped_images[i,j] = selected_filter

                ### Redraw the big image with the flipped images colored red
                display_big_image(filtered_small_splited, screen)
                pygame.display.flip()
    
    # Quit Pygame
    pygame.quit()
    return np.flip(flipped_images, axis=1)


