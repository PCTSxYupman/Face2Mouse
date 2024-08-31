import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Webcam setup
cap = cv2.VideoCapture(0)

# Constants for controlling the mouse movement
MOUSE_SPEED = 250  # Adjust the mouse movement speed
SLEEP_INTERVAL = 0.01  # Control the smoothness of the movement
CENTER_TOLERANCE = 0.1  # Tolerance for joystick deadzone

def move_mouse(x, y, center_x, center_y):
    """Move the mouse like a joystick based on face direction."""
    dx = (x - center_x) / center_x  # Normalize the movement
    dy = (y - center_y) / center_y  # Normalize the movement
    
    # Apply deadzone
    if abs(dx) < CENTER_TOLERANCE and abs(dy) < CENTER_TOLERANCE:
        return  # No movement if within deadzone
    
    # Calculate the amount of movement
    move_x = dx * MOUSE_SPEED
    move_y = dy * MOUSE_SPEED
    
    # Get current mouse position
    current_x, current_y = pyautogui.position()
    
    # Calculate new mouse position
    new_x = current_x + move_x
    new_y = current_y + move_y
    
    # Move the mouse, allowing it to move freely across monitors
    pyautogui.moveTo(new_x, new_y, duration=SLEEP_INTERVAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find the face mesh
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Center point of the face for joystick-like movement
            x_center = int(face_landmarks.landmark[1].x * w)
            y_center = int(face_landmarks.landmark[1].y * h)

            # Move mouse based on face direction
            move_mouse(x_center, y_center, w // 2, h // 2)

    cv2.imshow('Face Mouse Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
