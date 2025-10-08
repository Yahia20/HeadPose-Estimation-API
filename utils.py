import cv2
import numpy as np
import os
import uuid
from math import cos, sin
from fastapi import UploadFile

def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size=100):
    yaw = -yaw
    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)  # X: Red
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)  # Y: Green
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)  # Z: Blue
    return img

def map_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def generate_unique_filename(original_filename):
    extension = os.path.splitext(original_filename)[1]
    return f"{uuid.uuid4()}{extension}"

def save_uploaded_file(file: UploadFile, path: str):
    with open(path, "wb") as buffer:
        buffer.write(file.file.read())

def delete_file(path: str):
    if os.path.exists(path):
        os.remove(path)