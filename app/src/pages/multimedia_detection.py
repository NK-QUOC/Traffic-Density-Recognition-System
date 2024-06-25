import os
from pathlib import Path
import re
import tempfile
import cv2
import pandas as pd
import streamlit as st
from ..utils import Page
import settings
from ..utils import TrafficMonitor
from PIL import Image

from streamlit_drawable_canvas import st_canvas
from shapely.geometry import LineString, Point, Polygon

class MultimediaDetection(Page):
    def __init__(self, state):
        self.state = state
        self.traffic_monitor = TrafficMonitor(self.state)
        
    def write(self):
        st.subheader(
            "📺 :blue[Traffic Density Recognition for Multimedia Data]", divider="gray")

        if 'media_type' not in st.session_state:
            st.session_state['media_type'] = 'Image'
        if 'captured_frame' not in st.session_state:
            st.session_state['captured_frame'] = None
        if 'model_type' not in st.session_state:
            st.session_state['model_type'] = 'yolov9'
        if 'confidence' not in st.session_state:
            st.session_state['confidence'] = 0.25
        if 'iou' not in st.session_state:
            st.session_state['iou'] = 0.5
        if 'model_path' not in st.session_state:
            st.session_state['model_path'] = None
            
        def handle_source_change():
            st.session_state['captured_frame'] = None

        media_type = st.selectbox(
            "Select Source Type", ['Image', 'Video', 'YouTube', 'Webcam', 'RTSP'], 
            key='media_type',
            on_change=handle_source_change
        )
        
        params_col, results_col = st.columns([0.25, 0.75], gap="medium")

        with params_col:

            st.subheader("Parameters", divider="gray")
            
            # Image
            if media_type == 'Image':

                uploaded_file = st.file_uploader(
                    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

                if uploaded_file is not None:
                    try:
                        source_image = Image.open(
                            uploaded_file)
                        st.image(
                            source_image, caption="Uploaded Image", use_column_width=True)
                    except Exception as ex:
                        st.error("Error occurred while opening the image.")
                        st.error(ex)

            # Video
            elif media_type == 'Video':
                uploaded_file = st.file_uploader(
                "Choose a video file...", type=["mp4", "avi"],
                on_change=handle_source_change)
                is_stream = False
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        video_path = tmp_file.name
                    if video_path:
                        st.video(video_path)
                else:
                    video_path = st.selectbox(
                        "Choose a video...", settings.VIDEOS_DICT.keys(),
                        on_change=handle_source_change)
                    video_file_path = settings.VIDEOS_DICT.get(video_path)
                    if os.path.exists(video_file_path):
                        with open(video_file_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                    else:
                        st.error("Selected video file does not exist.")
                    if video_bytes:
                        st.video(video_bytes)
                    video_path = str(settings.VIDEOS_DICT.get(video_path))
            # YouTube
            elif media_type == 'YouTube':
                youtube_url = st.text_input("Enter YouTube URL", on_change=handle_source_change)
                video_path = None
                is_stream = False
                if youtube_url:
                    if self.traffic_monitor.is_youtube_url(youtube_url)[0]:
                        stream = st.radio("Stream", ('Yes', 'No'), index=1)
                        is_stream = True if stream == 'Yes' else False

                        if is_stream is False:
                            with st.spinner("Downloading video..."):
                                video_path = self.traffic_monitor.download_yt_video(youtube_url)
                            st.success("Video downloaded successfully.")
                        else:
                            video_path = youtube_url
                    else:
                        st.error("Not a valid YouTube URL.")
            # Webcam
            elif media_type == 'Webcam':
                video_path = settings.WEBCAM_PATH
                is_stream = False
                
            # RTSP
            elif media_type == 'RTSP':
                rtsp_url = st.text_input("Enter RTSP URL", on_change=handle_source_change)
                video_path = rtsp_url
                is_stream = False
                
            model_type = st.selectbox(
                "Select Model", ['yolov9', 'yolov8'],
                index=['yolov9', 'yolov8'].index(st.session_state['model_type']),
                key='model_type')
            
            confidence = st.slider(
                "Confidence Threshold", 0.0, 1.0, 0.25, 0.01, key='confidence')
            iou = st.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01, key='iou')
            
            if model_type == 'yolov9':
                st.session_state['model_path'] = Path(settings.YOLOv9_MODEL)
            elif model_type == 'yolov8':
                st.session_state['model_path'] = Path(settings.YOLOv8_MODEL)
                
                
            if media_type in ['Video', 'YouTube', 'Webcam', 'RTSP']:
                if video_path is not None and video_path != '':
                    vid_cap, is_stream =  self.traffic_monitor.read_video(video_path, is_stream=is_stream)
                else:
                    vid_cap = None
                st.subheader("Counting Regions")
                
                # Reset counting regions
                self.traffic_monitor.counting_regions = []

                selective_area = st.selectbox(
                    'Selective Area', 
                    ['Draw Area', 'ALL'], 
                    index=0,
                    key='selectived_draw_area',
                    )
                
                catured_frame_button = st.button("Capture Frame", key='capture_frame_button')
                
                if catured_frame_button:
                    try:
                        if is_stream:
                            read_vid_cap = vid_cap.read()
                            if read_vid_cap is not None:
                                # image = cv2.resize(vid_cap, (720, int(720*(9/16))))
                                st.session_state['captured_frame'] = cv2.cvtColor(
                                    read_vid_cap, cv2.COLOR_BGR2RGB)

                            else:
                                st.error("Failed to capture frame.")
                            vid_cap.stop()
                        else:
                            read_vid_cap = cv2.VideoCapture(video_path)
                            success, frame = read_vid_cap.read()
                            
                            if success:
                                # image = cv2.resize(frame, (720, int(720*(9/16))))
                                st.session_state['captured_frame'] = cv2.cvtColor(
                                    frame, cv2.COLOR_BGR2RGB)
                            else:
                                st.error("Failed to capture frame.")
                            read_vid_cap.release()
                    except Exception as e:
                        st.error("Error loading video: " + str(e))
                
                st.subheader("Tracker Options")
                self.traffic_monitor.display_tracker_options()                   
                    
            
            start_button = st.button("Start")

        with results_col:
            st.subheader("Results", divider="gray")
            if st.session_state['model_path'] is not None:
                self.traffic_monitor.load_model(st.session_state['model_path'])
                self.traffic_monitor.confidence = confidence
                self.traffic_monitor.iou = iou
                # Image
                if media_type == 'Image':
                    if start_button:
                        if uploaded_file is not None:
                            col1, col2 = st.columns([0.5, 0.5], gap="medium")
                            with col1:
                                st.image(source_image, caption="Default Image", use_column_width=True)
                            with col2:
                                self.traffic_monitor.process_image(
                                    source_image)
                        else:
                            st.error("Please upload an image.")
                                
                    if uploaded_file is None:
                        col1, col2 = st.columns([0.5, 0.5], gap="medium")
                        with col1:
                            default_image_path = str(settings.DEFAULT_IMAGE)
                            st.image(
                                default_image_path, caption="Default Image", use_column_width=True)
                        with col2:
                            default_detected_image_path = str(
                                settings.DEFAULT_DETECT_IMAGE)
                            st.image(default_detected_image_path,
                                    caption='Detected Image', use_column_width=True)
                            
                # Video
                elif media_type in ['Video', 'YouTube', 'Webcam', 'RTSP']:
                    points = []
                    if 'points' not in st.session_state:
                        st.session_state['points'] = []
                    if 'button_pressed' not in st.session_state:
                        st.session_state['button_pressed'] = False
                    draw_area_placeholder = st.empty()
                    with draw_area_placeholder.container():
                        if st.session_state['captured_frame'] is not None:
                            img_pil = Image.fromarray(st.session_state['captured_frame'])
                            print(img_pil.size)
                            if selective_area == 'Draw Area':
                                points = self.traffic_monitor.draw_tool(background_image=img_pil)
                                
                            elif selective_area == 'ALL':
                                points = [[(0, 0), (img_pil.width, 0), (img_pil.width,
                                                                        img_pil.height), (0, img_pil.height)]]
                            
                            if st.button('Complete Drawing', key='complete_drawing'):
                                st.session_state['button_pressed'] = True
                        else:
                            st.text("Click 'Capture Frame' to capture a frame.")
                                                                                
                        if st.session_state['button_pressed']:
                            st.session_state['points'] = points
                            print("new ", points)
                            self.traffic_monitor.create_counting_regions(st.session_state['points'])
                            
                            if self.traffic_monitor.counting_regions:
                                st.text("Successfully created counting regions.")
                                st.write("Counting Regions: ",  self.traffic_monitor.counting_regions)
                    
                    if start_button:
                        if vid_cap is not None:
                            draw_area_placeholder.empty()
                            self.traffic_monitor.process_video(vid_cap, is_stream)
                        else:
                            st.error("Please load a video.")