import cv2

# Load the Haar cascade XML files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Draw rectangles around the detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within each face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Draw rectangles around the detected eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        # Detect mouth within each face region
        mouths = mouth_cascade.detectMultiScale(roi_gray)
        
        # Draw rectangles around the detected mouths
        for (mx,my,mw,mh) in mouths:
            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
    
    # Display the resulting frame
    cv2.imshow('Face, Eye, Mouth Detection', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
