import json
from PIL import Image, ImageDraw
import numpy as np
import cv2

def detect_board(path) -> dict:
    # Load the image and convert it to grayscale
    image = Image.open(path).convert('L')
    width, height = image.size

    # Apply a threshold to create a binary image
    _, binary_image = cv2.threshold(np.array(image), 150, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and select the largest one (assuming it's the board)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Approximate the contour to a polygon
    epsilon = 0.05 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Calculate the bounding box of the board
    x, y, w, h = cv2.boundingRect(approx)

    # Calculate the square size
    square_size = min(w, h) // 8

    return {'x': x, 'y': y, 'square': square_size}