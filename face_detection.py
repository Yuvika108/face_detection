#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import time
import sys

# Create folder to save detected faces
os.makedirs("faces", exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load face cascade.")

# Open webcam
video_cap = cv2.VideoCapture(0)

if not video_cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press 'A' or 'ESC' to exit.")

last_save_time = 0

while True:
    ret, frame = video_cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Display number of faces
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Face label
        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Save one image every 2 seconds
        current_time = time.time()

        if current_time - last_save_time > 2:
            face_img = frame[y:y+h, x:x+w]

            filename = f"faces/face_{int(current_time)}.jpg"
            cv2.imwrite(filename, face_img)

            print(f"Saved: {filename}")
            last_save_time = current_time

    # Show video
    cv2.imshow("Face Detection", frame)

    # Read key press
    key = cv2.waitKey(1) & 0xFF

    # Exit on A/a or ESC
    import os
    if key == ord('a') or key == ord('A'):
        video_cap.release()
        cv2.destroyAllWindows()
        os._exit(0)

    # Exit if window closed manually
    if cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
video_cap.release()
cv2.destroyAllWindows()


# In[ ]:




