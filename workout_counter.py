#required libraries
import mediapipe as mp
import streamlit as st
import cv2
import numpy as np
import time
import subprocess

# importing mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#angle caculator fuction
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

st.title('Workout Counter with MediaPipe')

st.sidebar.subheader('Select the Workout')
app_mode = st.sidebar.selectbox('Choose the Workout',
                                ["None",'Bicep Curls','Squats','Push Ups','High Knees','Jab'])
st.sidebar.markdown('---')
use_webcam = st.sidebar.button('Start Video')
close_cam = st.sidebar.button('Stop Video')

st.sidebar.markdown('---')
detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
st.sidebar.markdown('---')
outVid = st.sidebar.button('Show Recording')   

if app_mode == 'None':
    st.image('demo4.gif',"Credit:https://dribbble.com/shots/2928065-Fitness-App-Animation-gif")
    
# Bicep Curls
if app_mode in ['Bicep Curls','Push Ups','Jab']:
    stframe = st.empty()
    counter = 0 
    if use_webcam:
        vid = cv2.VideoCapture(0)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))
        fps = 0
        i = 0
        
        position = None

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Count**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Position**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        with mp_pose.Pose(min_detection_confidence=detection_confidence,
                        min_tracking_confidence=tracking_confidence) as pose:
            prevTime = 0
            while(True):
                i +=1
                ret, frame = vid.read()
                if not ret:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False

                results = pose.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    
                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)
                    
                    # Visualize angle
                    cv2.putText(frame, str(angle), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    # Curl counter logic
                    if angle > 160:
                        position = "Down"
                    if angle < 110 and position =='Down':
                        position="Up"
                        counter +=1
                        print(counter)
                except:
                    pass
                # Render detections
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )               
                 
                #cv2.imshow('Mediapipe Feed', frame)
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                
                #Dashboard
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{counter}</h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{position}</h1>", unsafe_allow_html=True)

                stframe.image(frame,channels = 'BGR',use_column_width=True)
                out.write(frame)
                if close_cam:
                    vid.release()
                    out. release()
                    cv2.destroyAllWindows()


if app_mode in ['Squats', 'High Knees']:    
    stframe = st.empty()
    counter = 0 
    if use_webcam:
        vid = cv2.VideoCapture(0)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))
        fps = 0
        i = 0
        
        position = None

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Count**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**position**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        with mp_pose.Pose(min_detection_confidence=detection_confidence,
                        min_tracking_confidence=tracking_confidence) as pose:
            prevTime = 0
            while(True):
                i +=1
                ret, frame = vid.read()
                if not ret:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False

                results = pose.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    # Calculate angle
                    angle = calculate_angle(hip, knee, ankle)
                    
                    # Visualize angle
                    cv2.putText(frame, str(angle), 
                                tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    # Curl counter logic
                    if angle > 160:
                        position = "Up"
                    if angle < 110 and position =='Up':
                        position="Down"
                        counter +=1
                        print(counter)
                        
                        
                except:
                    pass
                # Render detections
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )               
                 
                #cv2.imshow('Mediapipe Feed', frame)
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                
                #Dashboard
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{counter}</h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{position}</h1>", unsafe_allow_html=True)

            
                stframe.image(frame,channels = 'BGR',use_column_width=True)
                out.write(frame)
                if close_cam:
                    vid.release()
                    out. release()
                    cv2.destroyAllWindows()

    
if outVid:
    subprocess.run(["powershell","-Command","echo y | ffmpeg -i output1.mp4 -vcodec libx264 output2.mp4"])
    #st.write('Hurry!! You did {} Squats'.format(counter))
    st.balloons()
    st.text('Video Processed')
    video_file = open('output2.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    