#!/usr/bin/env python
# coding: utf-8

# # 1.Installation and Dependencies

# In[49]:


get_ipython().system('pip install mediapipe opencv.python')
#here we installed media pipe and opencv dependencies


# In[50]:


import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils #this will give us the whole drawing using media pipe utilities
mp_pose = mp.solutions.pose #this will help us to detect the pose 
#this is actually importing our pose estimation model


# In[51]:


#now we are gonna actually test out a feed from our webcam now
#video feed
cap = cv2.VideoCapture(0)    #video capture device is any kind of device in which you can make video and the number in the bracket is any number which represent my webcam
while cap.isOpened(): 
    ret, frame = cap.read() #ret is the return variable in this we gonna return the values
    #frame is going to give the image from our webcam 
    cv2.imshow('Mediapipe Feed', frame) #cv2 imshow actually give us the pop on the screen which allows us to visualize a particular image
    
    if cv2.waitKey(10) & 0xFF == ord('q'): #waitkey works when whether we are out of the feed or whether we are trying to close our screen so this will break the loop 
        break  #oxff is used to find which we have actually hit on our keyboard equals then just letter q and we are breakng 
        
        
cap.release()   #it will release the output on the screen
cv2.destroyAllWindows()
    


# # 1. Making Detections

# In[124]:


cap = cv2.VideoCapture(0)  

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose: #we are setting here the accuracy of our trackor, detecting and tracking as 50% 
    while cap.isOpened(): 
        ret, frame = cap.read() 
        
        
        # Detect the stuff and render
        #recolour image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # recolouring a frame color arrays from given bgr to rgb 
        image.flags.writeable= False #whether its writeable or not
        
        #make detection
        results = pose.process(image)
        
       # bringing back the the original image
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #RENDER DETECTIONS
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, #MP_DRAWING using  DRAW LANDMARKS it is the utilities of the given mediapipe drawing library
                        mp_drawing.DrawingSpec(color = (245,117,66), thickness=2,circle_radius=2),
                        mp_drawing.DrawingSpec(color = (245,166,616), thickness=2,circle_radius=2),
                )
        cv2.imshow('Mediapipe Feed', image) 
    
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break  
        
        
    cap.release()  
    cv2.destroyAllWindows()


# mp_drawing.draw_landmarks??

# # 2.Determing Joints

# ![pose.png](attachment:pose.png)

# #search on google how to insert iamge in ipynb file

# In[125]:


cap = cv2.VideoCapture(0)  

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose: #we are setting here the accuracy of our trackor, detecting and tracking as 50% 
    while cap.isOpened(): 
        ret, frame = cap.read() 
        
        
        # Detect the stuff and render
        #recolour image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # recolouring a frame color arrays from given bgr to rgb 
        image.flags.writeable= False #whether its writeable or not
        
        #make detection
        results = pose.process(image)
        
       # bringing back the the original image or recolur back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass
        
        #RENDER DETECTIONS
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, #MP_DRAWING using  DRAW LANDMARKS it is the utilities of the given mediapipe drawing library
                        mp_drawing.DrawingSpec(color = (245,117,66), thickness=2,circle_radius=2),
                        mp_drawing.DrawingSpec(color = (245,166,616), thickness=2,circle_radius=2),
                )
        cv2.imshow('Mediapipe Feed', image) 
    
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break  
        
        
    cap.release()  
    cv2.destroyAllWindows()


# In[126]:


len(landmarks)


# In[127]:


for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk)


# In[128]:


landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]


# In[129]:


mp_pose.PoseLandmark.LEFT_SHOULDER.value


# In[130]:


landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]


# In[131]:


landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]


# # 3.Calculate Angles

# In[132]:


def calculate_angles(a,b,c):
    a = np.array(a) #first
    b = np.array(b) #mid
    c = np.array(c) #end
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]) #calculating the angle between soulder elbow and wrist
    
    angle = np.abs(radians*180.0/np.pi) #angle we are converting into degree by dividing it with 180 degree and multiplyingg by pie
    
    if angle>180.0:  #hands can't go more than 180 degree (straigght hand)
        amgle = 360-angle
        
    return angle
        


# In[133]:


shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] 


# In[134]:


shoulder,elbow,wrist


# In[135]:


calculate_angles(shoulder,elbow,wrist)


# In[147]:


cap = cv2.VideoCapture(0)  

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose: #we are setting here the accuracy of our trackor, detecting and tracking as 50% 
    while cap.isOpened(): 
        ret, frame = cap.read() 
        
        
        # Detect the stuff and render
        #recolour image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # recolouring a frame color arrays from given bgr to rgb 
        image.flags.writeable= False #whether its writeable or not
        
        #make detection
        results = pose.process(image)
        
       # bringing back the the original image or recolur back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            
            #we have copied it from above to get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] 
            
            #calculate angles
            angle = calculate_angles(shoulder,elbow,wrist)
            
            #visualise 
            
            #rended our code to actual screen, puttext visualize on screen
            cv2.putText(image,str(angle),
                        tuple(np.multiply(elbow, [640,480]).astype(int)),  #here we have multiplied our angle with 6640 and 480 which our scren size so that we can see anfle properly on scren  
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA
                             )
                       
            
            print(landmarks)   
                        
        except:
            pass
        
        #RENDER DETECTIONS
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, #MP_DRAWING using  DRAW LANDMARKS it is the utilities of the given mediapipe drawing library
                        mp_drawing.DrawingSpec(color = (245,117,66), thickness=2,circle_radius=2),
                        mp_drawing.DrawingSpec(color = (245,166,616), thickness=2,circle_radius=2),
                )
        cv2.imshow('Mediapipe Feed', image) 
    
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break  
        
        
    cap.release()  
    cv2.destroyAllWindows()


# # 4. Curl Calculator

# In[171]:


cap = cv2.VideoCapture(0)  


#curl counter variables
counter =0
stage = None

#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose: #we are setting here the accuracy of our trackor, detecting and tracking as 50% 
    while cap.isOpened(): 
        ret, frame = cap.read() 
        
        
        # Detect the stuff and render
        #recolour image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # recolouring a frame color arrays from given bgr to rgb 
        image.flags.writeable= False #whether its writeable or not
        
        #make detection
        results = pose.process(image)
        
       # bringing back the the original image or recolur back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            
            #we have copied it from above to get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] 
            
            #calculate angles
            angle = calculate_angles(shoulder,elbow,wrist)
            
            #visualise 
            
            #rended our code to actual screen, puttext visualize on screen
            cv2.putText(image,str(angle),
                        tuple(np.multiply(elbow, [640,480]).astype(int)),  #here we have multiplied our angle with 6640 and 480 which our scren size so that we can see anfle properly on scren  
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA
                             )
                       
            #curl counter logic
            if angle>180:
                stage = 'down'
            if angle <30 and stage == 'down':
                stage = 'up'
                counter +=1
                print(counter)
                        
        except:
            pass
        
        #RENDER DETECTIONS
        
        #setup status boz
        cv2.rectangle(image, (0,0), (255,73), (345,117,16),-1) #creating a box in which we are gonna print the count
        
        #rep data
        cv2.putText(image, 'REPS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (13,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        
        #stage data
        cv2.putText(image, 'STAGE', (120,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (65,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, #MP_DRAWING using  DRAW LANDMARKS it is the utilities of the given mediapipe drawing library
                        mp_drawing.DrawingSpec(color = (245,117,66), thickness=2,circle_radius=2),
                        mp_drawing.DrawingSpec(color = (245,166,616), thickness=2,circle_radius=2),
                )
        cv2.imshow('Mediapipe Feed', image) 
    
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break  
        
        
    cap.release()  
    cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




