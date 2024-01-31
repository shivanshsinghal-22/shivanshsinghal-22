#import cv2 
import cv2
from ultralytics import YOLO 
import numpy as np

model = YOLO('runs/detect/train20/weights/best.pt')

#Cv2 contains a VideoCapture method that captures frames from default camera(0)
camera = cv2.VideoCapture(0)

#Set resolutions to 640 X 480
camera.set(3,640)
camera.set(4,480)


while True:
    #The read function returns a boolean describing if reading was successful and the next Frame.
    [hasRead,myFrame] = camera.read()

    #The defined Yolo model returns the frame after modification. imshow shows the processed Frame.
    results = model(myFrame,stream=True)

    for detects in results:

        boxes = detects.boxes

        for b in boxes:

            x1 , y1 , x2 , y2 = map(int,b.xyxy[0])
            id = int(b.cls[0])

            color = (0,255,0)
            thickness = 2
            cv2.rectangle(myFrame,(x1,y1),(x2,y2),color,thickness)
            cv2.putText(myFrame, str(id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    cv2.imshow('Detection',myFrame)
    
    """
    The line if cv2.waitKey(1) == ord('q'): break is a common way to handle keyboard input in OpenCV applications. Let me break it down for you:
    cv2.waitKey(1): This function waits for a keyboard event for the specified duration (given in milliseconds). In this case, it waits for 1 millisecond (1 is passed as an argument). 
    If a key is pressed within this time, the function returns the ASCII value of the key.

    ord('q'): This function returns the ASCII value of the character 'q'.

    if cv2.waitKey(1) == ord('q'): break:

    This line checks if the key pressed is 'q'. If it is, the condition evaluates to True, and the break statement is executed.
    The break statement is used to exit the loop, effectively ending the program.
    So, in simple terms, this line of code checks if the 'q' key is pressed, and if it is, it breaks out of the loop, leading to the termination of the program. 
    It's a common way to provide a way to exit a video processing loop in OpenCV applications.
    """
    if cv2.waitKey(1)==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()