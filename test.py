import cv2
import numpy as np
import datetime

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(1)

# Function to handle mouse events (for quit button)
def handle_mouse_event(event, x, y, flags, param):
    global quit_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if x >= quit_button_x and x <= quit_button_x + quit_button_w and y >= quit_button_y and y <= quit_button_y + quit_button_h:
            quit_clicked = True

# Create a window with a quit button
quit_button_x, quit_button_y, quit_button_w, quit_button_h = 10, 10, 100, 50
quit_button_text = "Quit"
quit_clicked = False
cv2.namedWindow('Face Mask Detection')
cv2.setMouseCallback('Face Mask Detection', handle_mouse_event)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Resize the face region for processing
        face_roi_resized = cv2.resize(face_roi, (150, 150))
        
        # Convert the resized face to HSV color space
        hsv = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2HSV)
        
        # Define the lower and upper bounds for the mask color (blue color example)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create a mask based on the color range
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Perform morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        
        # Calculate the percentage of pixels within the mask
        mask_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_percentage = (mask_pixels / total_pixels) * 100
        
        # Assuming a mask is present if the mask percentage is above a threshold
        if mask_percentage > 10:  # Adjust threshold as needed
            label = "Mask"
            color = (0, 255, 0)  # Green color for mask
        else:
            label = "No Mask"
            color = (0, 0, 255)  # Red color for no mask
        
        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Add timestamp
        cv2.putText(frame, str(datetime.datetime.now()), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw quit button on the frame
    cv2.rectangle(frame, (quit_button_x, quit_button_y), (quit_button_x + quit_button_w, quit_button_y + quit_button_h), (255, 100, 102), -1)
    cv2.putText(frame, quit_button_text, (quit_button_x + 10, quit_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Face Mask Detection', frame)
    
    # Check if quit button is clicked
    if quit_clicked:
        break
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
