import numpy as np
import tensorflow as tf
import pygame
import sys
from pygame.locals import *

pygame.init()
window_size = 400
screen = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption('Draw a Digit')

white = (255, 255, 255)
black = (0, 0, 0)

model = tf.keras.models.load_model('mnist_model_tanmay.keras')


def preprocess_image(image):

    image_array = pygame.surfarray.array3d(image)
    
    gray_image = np.mean(image_array, axis=2)
    
    gray_image_surface = pygame.transform.scale(pygame.surfarray.make_surface(gray_image), (28, 28))
    
    image_array = pygame.surfarray.array3d(gray_image_surface).swapaxes(0, 1)
    image_array = image_array.astype('float32') / 255.0

    image_array = 1 - image_array

    image_array = np.squeeze(image_array)

    image_array = np.expand_dims(image_array, axis=0)  

    return image_array

def predict_digit(image):
    
   
    processed_image = preprocess_image(image)
    processed_image = processed_image[:, :, :, 0]
    prediction = model.predict(processed_image)
    return np.argmax(prediction), np.max(prediction)

def main():
    drawing = False
    screen.fill(white)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == MOUSEBUTTONUP:
                drawing = False
            elif event.type == MOUSEMOTION and drawing:
                pygame.draw.circle(screen, black, event.pos, 10)

            elif event.type == KEYDOWN:
                if event.key == K_RETURN:  

                    predicted_digit, confidence = predict_digit(screen)
                    font = pygame.font.Font(None, 25)
                    text = font.render(f"Prediction: {predicted_digit}, Confidence: {confidence:.4f}", True, black)
                    screen.blit(text, (10, 10))
                    pygame.display.update()

                    print(f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.4f}")
                elif event.key == K_c:  
                    screen.fill(white)

        pygame.display.update()

if __name__ == '__main__':
    main()
