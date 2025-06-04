import cv2
import numpy as np

# Kalman Filter Setup
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

cap = cv2.VideoCapture(0)

# Canvas to draw on
_, frame = cap.read()
canvas = np.zeros_like(frame)

prev_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect red color
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prediction = kalman.predict()
    pred_x, pred_y = int(prediction[0]), int(prediction[1])

    center = None

    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:
            center = (int(x), int(y))
            kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 255), -1)
    else:
        center = (pred_x, pred_y)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)
        cv2.putText(frame, "Predicted", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 1)

    # Draw on canvas
    if center and prev_center:
        cv2.line(canvas, prev_center, center, (0, 0, 255), 3)

    prev_center = center

    # Combine canvas with webcam feed
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    cv2.imshow("Draw with Red Object", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)  # Clear canvas

cap.release()
cv2.destroyAllWindows()
