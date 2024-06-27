import gc
import os
import re
import uuid
import cv2
import numpy as np
import pandas as pd
import datetime
import time
from PIL import Image
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import tempfile

import asyncio
import threading
import queue

import streamlit as st
from streamlit_folium import st_folium
from folium.plugins import Draw
import folium
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import LineString, Point, Polygon
import plotly.express as px

from pytube import YouTube
import torch
from vidgear.gears import CamGear

from ultralytics.utils.plotting import Annotator, colors
from ultralytics import YOLO
from src import settings

from sqlalchemy.orm import Session
from src.database.database import get_db
from src.database.models import TrafficDensity, Vehicle

def add_custom_css():
    st.markdown(
        """
        """,
        unsafe_allow_html=True
    )

class Page(ABC):
    @abstractmethod
    def write(self):
        pass

class TrafficMonitor():
    def __init__(self, state=None):
        self.state = state
        self.model = YOLO(settings.YOLOv9_MODEL)
        self.counting_regions = []
        self.track_history = defaultdict(list)
        self.confidence = 0.25
        self.iou = 0.5
        self.is_display_tracker = True
        self.tracker_type = None
        self.stream_options = {"STREAM_RESOLUTION": "720p"}
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # line chart
        self.data_graph_df = pd.DataFrame(
            {'Time': [], 'Vehicle Count': [], 'Vehicle Type': []})
        self.placeholder = st.empty()
        self.placeholder_count = st.empty()
        self.placeholder_chart = st.empty()
        self.chart_start_time = time.time()
        self.is_display_chart = False
        
        # count
        self.counted_ids = {}
        self.counted_classes = {}
        self.disappearance_counts = {}
        self.MAX_DISAPPEARANCE_FRAMES = 30
        self.camera_locations = []
        self.vehicle_counts = {}
        
        # Density
        self.roi_area = []
        self.vehicles_area_total = []
        
    def load_model(self, model_path):
        self.model = YOLO(model_path)

    def get_class_names(self):
        return self.model.names
    
    def display_tracker_options(self):

        # display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
        # self.is_display_tracker = True if display_tracker == 'Yes' else False
        # if self.is_display_tracker:
        tracker_radio = st.radio(
            "Tracker", ('bytetrack', 'botsort'))
        self.tracker_type = settings.bytetrack if tracker_radio == 'bytetrack' else settings.botsort

    def is_youtube_url(self, url):
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
        youtube_match = re.match(youtube_regex, url)
        if youtube_match:
            return True, youtube_match.group(6)  # Trả về True và ID của video
        return False, None
    
    def convert_coordinates(self, old_coords, original_size, resized_size):
        width_ratio = original_size[0] / resized_size[0]
        height_ratio = original_size[1] / resized_size[1]

        new_coords = []
        for x, y in old_coords:
            new_x = round(x * width_ratio)
            new_y = round(y * height_ratio)
            new_coords.append((new_x, new_y))
        
        return new_coords
    
    @st.experimental_fragment
    def draw_tool(self, background_image=None, height=405, width=720):
        
        drawing_mode = st.selectbox(
            "Drawing tool:", 
            ("rect", "polygon", "transform"),
            index=1)
        if drawing_mode == "point":
            point_display_radius = st.slider(
                "Point display radius: ", 1, 25, 3)
        if drawing_mode == "polygon":
            st.caption("Left-click to add a point, right-click to close the polygon, double-click to remove the latest point")
            
        # stroke_width = st.slider("Stroke width: ", 1, 25, 3)
        # stroke_color = st.color_picker("Stroke color hex: ")
        # bg_color = st.color_picker("Background color hex: ", "#eee")
        realtime_update = st.checkbox("Update in realtime", True)

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color='#000',
            background_color='#eee',
            background_image=background_image,
            update_streamlit=realtime_update,
            # height=background_image.height if background_image else 150,
            # width=background_image.width if background_image else 300,
            height=405,
            width=720,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            display_toolbar=st.checkbox("Display toolbar", True),
            key="full_app",
        )
        # if canvas_result.image_data is not None:
        #     st.image(canvas_result.image_data, use_column_width=True)
        
        points = []
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)

            for obj in canvas_result.json_data["objects"]:
                x = obj['left']
                y = obj['top']
                w = obj['width']
                h = obj['height']
                if obj['type'] == 'line':
                    old_coords = [(int(x-w/2), int(y-h/2)),
                                  (int(x+w/2), int(y+h/2))]
                    new_coords = self.convert_coordinates(old_coords, (background_image.width, background_image.height), (width, height))
                    points.append(new_coords)
                elif obj['type'] == 'rect':
                    old_coords = [(int(x), int(y)), (int(x+w), int(y)),
                                  (int(x+w), int(y+h)), (int(x), int(y+h))]
                    new_coords = self.convert_coordinates(old_coords, (background_image.width, background_image.height), (width, height))
                    points.append(new_coords)
                elif obj['type'] == 'path':
                    old_coords = self.parse_svg_path(obj['path'])
                    new_coords = self.convert_coordinates(old_coords, (background_image.width, background_image.height), (width, height))
                    points.append(new_coords)
        else:
            points = [(0, 0), (background_image.width, 0), (background_image.width,
                                                            background_image.height), (0, background_image.height)]
        return points
    
    def draw_tool_old(self, background_image=None):
        
        drawing_mode = st.selectbox(
            "Drawing tool:", ("rect", "polygon", "transform"))
        stroke_width = st.slider("Stroke width: ", 1, 25, 3)
        if drawing_mode == "point":
            point_display_radius = st.slider(
                "Point display radius: ", 1, 25, 3)
        stroke_color = st.color_picker("Stroke color hex: ")
        bg_color = st.color_picker("Background color hex: ", "#eee")
        realtime_update = st.checkbox("Update in realtime", True)

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=background_image,
            update_streamlit=realtime_update,
            height=background_image.height if background_image else 150,
            width=background_image.width if background_image else 300,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="full_app",
        )
        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data)

        points = []
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)

            for obj in canvas_result.json_data["objects"]:
                x = obj['left']
                y = obj['top']
                w = obj['width']
                h = obj['height']
                if obj['type'] == 'line':
                    points.append([(int(x-w/2), int(y-h/2)),
                                  (int(x+w/2), int(y+h/2))])
                elif obj['type'] == 'rect':
                    points.append([(int(x), int(y)), (int(x+w), int(y)),
                                  (int(x+w), int(y+h)), (int(x), int(y+h))])
                elif obj['type'] == 'path':
                    points.append(obj['path'])
        else:
            points = [(0, 0), (background_image.width, 0), (background_image.width,
                                                            background_image.height), (0, background_image.height)]
        return points

    def selective_area_tool(self, source_vid, isCameGear=False):

        selective_area = st.selectbox(
            'Selective Area', ['Default', 'Draw Area', 'ALL'])

        points = []

        if 'catured_frame_button' not in st.session_state:
            st.session_state.catured_frame_button = None
        
        if 'captured_frame' not in st.session_state:
            st.session_state.captured_frame = None

        if st.button('Capture Frame'):
            try:
                if isCameGear:
                    vid_cap = source_vid.read()
                    if vid_cap is not None:
                        # image = cv2.resize(vid_cap, (720, int(720*(9/16))))
                        st.session_state.captured_frame = cv2.cvtColor(
                            vid_cap, cv2.COLOR_BGR2RGB)

                    else:
                        st.error("Failed to capture frame.")
                    vid_cap.stop()
                else:
                    vid_cap = cv2.VideoCapture(source_vid)
                    success, frame = vid_cap.read()
                    if success:
                        # image = cv2.resize(frame, (720, int(720*(9/16))))
                        st.session_state.captured_frame = cv2.cvtColor(
                            frame, cv2.COLOR_BGR2RGB)

                    else:
                        st.error("Failed to capture frame.")
                    vid_cap.release()
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))

        if st.session_state.captured_frame is not None:
            img_pil = Image.fromarray(st.session_state.captured_frame)
            if selective_area == 'Draw Area':
                points = self.draw_tool(background_image=img_pil)
            elif selective_area == 'ALL':
                points = [[(0, 0), (img_pil.width, 0), (img_pil.width,
                                                        img_pil.height), (0, img_pil.height)]]
            else:
                points = [[(0, 0), (img_pil.width, 0), (img_pil.width,
                                                        img_pil.height), (0, img_pil.height)]]
        else:
            st.text("Click 'Capture Frame' to capture a frame.")

        return points

    def display_selective_area(self, tracks, frame, CLASS_NAMES):
        line_thickness = 2
        region_thickness = 2
        track_thickness = 1
        
        self.vehicles_area_total = {region["id"]:0 for region in self.counting_regions}
        
        annotator = Annotator(
            frame, line_width=line_thickness, example=str(CLASS_NAMES))
        
        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            track_ids = tracks[0].boxes.id.int().tolist()
            clss = tracks[0].boxes.cls.cpu().tolist()
            confs = tracks[0].boxes.conf

            
            
            # Update counts for all tracked objects at the start of each frame
            for region_id in self.counted_ids:
                for track_id in self.counted_ids[region_id]:
                    if track_id not in track_ids:
                        # If the object is not detected in the current frame, increase the disappearance count
                        self.disappearance_counts[track_id] += 1
                        # If the object has left the region for a certain number of frames, decrease the count and remove its ID from the dictionary
                        if self.disappearance_counts[track_id] > self.MAX_DISAPPEARANCE_FRAMES:
                            for region in self.counting_regions:
                                region_id = region["id"]
                                if track_id in self.counted_ids.get(region_id, []):
                                    region["counts"][self.counted_classes[track_id]] -= 1
                                    self.counted_ids[region_id].remove(
                                        track_id)
            
            
            for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                label = f"{CLASS_NAMES[cls]}#{track_id}#{conf:.2f}"
                annotator.box_label(box, label, color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                
                track_line = self.track_history[track_id]
                track_line.append(
                    (float(bbox_center[0]), float(bbox_center[1])))
                if len(track_line) > 30:
                    track_line.pop(0)
                points = np.hstack(track_line).astype(
                    np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(
                    cls, True), thickness=track_thickness)
                
                for region in self.counting_regions:
                    region_id = region["id"]
                    
                    if region['type'] == 'polygon':

                        if region["points"].contains(Point((bbox_center[0], bbox_center[1]))):
                            
                            bbox_area = (box[2] - box[0]) * (box[3] - box[1])*0.85
                            self.vehicles_area_total[region_id] += bbox_area 
                            
                            if track_id not in self.counted_ids.get(region_id, []):
                                
                                self.counted_classes[track_id] = CLASS_NAMES[cls]
                                region["counts"][self.counted_classes[track_id]] += 1
                                self.counted_ids.setdefault(
                                    region_id, []).append(track_id)

                                # Reset the disappearance count when the object is detected in the region
                                self.disappearance_counts[track_id] = 0
                        else:
                            # If the object has left the region, decrease the count and remove its ID from the dictionary
                            if track_id in self.counted_ids.get(region_id, []):
                                region["counts"][self.counted_classes[track_id]] -= 1
                                self.counted_ids[region_id].remove(track_id)            
            
            y_offset = 10  # Khoảng cách từ đỉnh khung hình
            x_offset = 10  # Khoảng cách từ cạnh trái khung hình
            for region in self.counting_regions:
                if region["type"] == "polygon":
                    region_label = ' | '.join(
                        [f"{cls}: {count}" for cls, count in region["counts"].items() if count > 0])
                    region_color = region["region_color"]
                    region_text_color = region["text_color"]

                    polygon_coords = np.array(
                        region["points"].exterior.coords, dtype=np.int32)
                    text_size, _ = cv2.getTextSize(
                        region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)
                    text_x = x_offset
                    text_y = y_offset + text_size[1]
                    cv2.rectangle(
                        frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                    cv2.putText(frame, region_label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2)

                    # Tăng y_offset để in thông tin tiếp theo bên dưới
                    y_offset += text_size[1] + 10

                    cv2.polylines(frame, [
                                  polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)
        return frame
        
    def parse_svg_path(self, svg_path):
        "Parse SVG path to get coordinates of the polygon."
        if not isinstance(svg_path, (list, tuple)):
            raise ValueError(f"Expected a list or tuple of SVG commands, got {type(svg_path)} instead.")

        # print(svg_path)

        # Check if the svg_path contains valid commands
        if not any(isinstance(command, (list, tuple)) and command[0] in ['M', 'L', 'C', 'z'] for command in svg_path):
            return svg_path 
        
        coords = []
        for command in svg_path:
            if command[0] in ['M', 'L']:
                coords.append((command[1], command[2]))
            elif command[0] == 'C':
                coords.append((command[5], command[6]))
        return coords

    def create_counting_regions(self, points):
        region_id = 0
        for point in points:
            # Check if all elements are tuples of length 2
            # point = self.parse_svg_path(point)
            # print(point)
            if all(isinstance(coord, tuple) and len(coord) == 2 for coord in point):
                if len(point) == 2:  # It's a line
                    self.counting_regions.append(
                        {
                            "id": region_id,
                            "type": "line",
                            "points": LineString(point),
                            "counts": defaultdict(int),
                            "region_color": (0, 255, 0),
                            "text_color": (255, 255, 255),
                        }
                    )
                elif len(point) > 2:  # It's a polygon
                    self.counting_regions.append(
                        {
                            "id": region_id,
                            "type": "polygon",
                            "points": Polygon(point),
                            "counts": defaultdict(int),
                            "region_color": (255, 42, 4),
                            "text_color": (255, 255, 255),
                        }
                    )
                region_id += 1

    def download_yt_video(self, url):
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension="mp4", res="720p").first()
        return stream.url

    def read_video(self, video_path, is_stream=False):
        if is_stream:
            video_stream = CamGear(
                source=video_path, stream_mode=True, **self.stream_options).start()

            # gstreamer_pipeline = f'souphttpsrc location={video_path} ! decodebin ! videoconvert ! appsink'
            # video_stream = cv2.VideoCapture(
            #     gstreamer_pipeline, cv2.CAP_GSTREAMER)

            return video_stream, True
        else:
            video_cap = cv2.VideoCapture(video_path)
            return video_cap, False

    def save_data_to_csv(self):
        # Save data to CSV file
        self.data_graph_df.to_csv(
            settings.DATA_CSV_PATH / 'data.csv', index=False)

    def load_data_from_csv(self):
        try:
            self.data_graph_df = pd.read_csv(
                settings.DATA_CSV_PATH / 'data.csv')
        except FileNotFoundError:
            st.warning(
                "No existing data found. Starting with an empty dataset.")
            self.data_graph_df = pd.DataFrame(
                {'Time': [], 'Vehicle Count': [], 'Vehicle Type': []})

    def plot_graph(self, placeholder):
        with placeholder.container():
            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                fig = px.line(self.data_graph_df, x='Time',
                              y='Vehicle Count', color='Vehicle Type', title='Vehicle Count vs Time')
                st.plotly_chart(fig)
            with fig_col2:

                fig = px.pie(self.data_graph_df, names='Vehicle Type',
                                values='Vehicle Count', title='Vehicle Type Distribution')
                st.plotly_chart(fig)

    def create_unique_filename(self, base_name="output", extension=".mp4"):
        while True:
            # Tạo ID ngẫu nhiên
            unique_id = uuid.uuid4()
            # Tạo tên file mới với ID này
            filename = f"{base_name}_{unique_id}{extension}"
            # Kiểm tra xem file đã tồn tại chưa
            if not os.path.exists(filename):
                return filename
  
    def process_video(self, vid_cap, is_stream=False):
                
        frame_window = st.image([])

        self.data_graph_df = pd.DataFrame(
            {'Time': [], 'Vehicle Count': [], 'Vehicle Type': []})
        
        self.placeholder = st.empty()
        self.placeholder_count = st.empty()
        self.placeholder_chart = st.empty()
        
        self.counted_ids = {}
        self.counted_classes = {}
        self.vehicle_counts = {}
        start_time = time.time()
        
        if 'start_process' not in st.session_state:
            st.session_state.start_process = True
            
        if st.button("Stop Process"):
            st.session_state.start_process = False
        
        if self.counting_regions:
            self.calculate_area_roi(self.counting_regions)
        
        if is_stream:
            frame = vid_cap.read()
            frame_height, frame_width = frame.shape[:2]
        else:
            frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        output_filename = self.create_unique_filename(settings.OUTPUT_VIDEO_DIR / 'multimedia/output', '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_filename, fourcc, 30.0, (frame_width, frame_height))
        
        while st.session_state.start_process:

            prev_time = time.time()

            if is_stream:
                frame = vid_cap.read()
            else:
                success, frame = vid_cap.read()
                if not success:
                    vid_cap.release()
                    break
            
            # frame = cv2.resize(frame, (720, int(720*(9/16))))

            if self.counting_regions:

                # -------- Object Detection And Tracking -------- #
                results = self.model.track(frame, conf=self.confidence, iou=self.iou, persist=True,
                                           device=self.device, tracker=self.tracker_type, show=False)
                
                # -------- Display Selective Area And Counting -------- #
                new_frame = self.display_selective_area(
                    results, frame, self.model.names)

                # -------- Plot Graph -------- #
                if time.time() - start_time > 5:  # Update data every 5 second
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print("Time", time.time() - start_time)
                    start_time = time.time()
                    self.is_display_chart = True
                    for region in self.counting_regions:
                        for cls, count in region['counts'].items():
                            # Thêm dữ liệu mới vào DataFrame
                            new_row = {'Time': timestamp, 'Vehicle Count': count,
                                       'Vehicle Type': cls}
                            self.data_graph_df = pd.concat(
                                [self.data_graph_df, pd.DataFrame([new_row])], ignore_index=True)

                col1, col2, col3 = self.placeholder.columns(3, gap="medium")

                density_level = self.calculate_density()
                
                if self.is_display_chart:
                    self.plot_graph(placeholder=self.placeholder_chart)
                    self.is_display_chart = False
                
                # Display the FPS
                curr_time = time.time()
                fps = round(1 / (curr_time - prev_time))
                prev_time = curr_time
                print("FPS: ", fps)

                col1.markdown(f"<p style='text-align: center; color: red; font-size: 20px; font-weight: bold;'>FPS: {fps}</p>", unsafe_allow_html=True)
                col3.markdown(f"<p style='text-align: center; color: red; font-size: 20px; font-weight: bold;'>Size: {frame.shape[1]}x{frame.shape[0]}</p>", unsafe_allow_html=True)
                
                for region in self.counting_regions:
                    region_id = region["id"]
                    density_level_value = "{:.2f}".format(round(density_level[region_id], 2))
                    density_level_str = 'low' if density_level[region_id] < 1/3 else 'medium' if density_level[region_id] < 2/3 else 'high'
                    col2.markdown(f"<p style='text-align: center; color: red; font-size: 20px; font-weight: bold;'>Density Level: {density_level_value} | {density_level_str} </p>", unsafe_allow_html=True)

                    CLASS_NAMES = self.model.names.values()
                    
                    for cls in CLASS_NAMES:
                        if cls not in region["counts"]:
                            region["counts"][cls] = 0
                        else:
                            pass
                    
                    region_label = ' | '.join(
                        [f"{cls}: {count}" for cls, count in region["counts"].items()])
                    
                    total_count = sum(region["counts"].values())
                    self.placeholder_count.write(f"""\n
                                                **[Region ID]**: {region["id"]}
                                                \n
                                                - **[Counts]**: {region_label}
                                                \n
                                                - **[Total Count]**: {total_count}
                                                \n
                                                 """)
                    offset_x_density = int(frame_width/2) + 30
                    cv2.putText(new_frame, f"Density Level: {density_level_value}", (offset_x_density, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                # cv2.putText(new_frame, "FPS: {:.2f}".format(fps), (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # density_level_x_position = 10 + 200  # 10 là vị trí x của FPS, 200 là khoảng cách mong muốn              
                
                frame_window.image(new_frame,
                                   caption='Detected Video',
                                   channels="BGR",
                                   use_column_width=True)
                
                out.write(new_frame)

                del results
                del new_frame
                del frame
                gc.collect()
            else:
                st.error(
                    "No counting regions found. Please create counting regions.")
                break
        
        if is_stream:
            vid_cap.stop()
        else:
            vid_cap.release()
 
    def save_detection_data(self, db: Session, timestamp : datetime, vehicle_type: str, count: int, camera_id: str):
        new_record = Vehicle(timestamp=timestamp, vehicle_type=vehicle_type, count=count, camera_id=camera_id)
        db.add(new_record)
        db.commit()
        db.refresh(new_record)

    def calculate_area_roi(self, points : list):
        for point in points:
            self.roi_area.append(
                {
                    "region_id": point["id"],
                    "area": point["points"].area # diện tích
                }
            )

    def calculate_density(self):
        density_levels = {}
        for region in self.counting_regions:
            region_id = region["id"]
            vehicle_area_total = float(self.vehicles_area_total.get(region_id, 0))
            roi_area = next((item["area"] for item in self.roi_area if item["region_id"] == region_id), None)

            if roi_area is not None and roi_area != 0:
                density = vehicle_area_total / roi_area
            else:
                density = 0

            density_levels[region_id] = density

        return density_levels

    def update_region_density(self, db: Session, timestamp : datetime, count_total: int, camera_id: str):
        timestamp = datetime.datetime.now()
        
        for region in self.counting_regions:
            region_id = region["id"]
            density_levels = self.calculate_density()
            density_level = density_levels[region_id]
        print("Density Level: ", density_level)
        new_density = TrafficDensity(timestamp=timestamp, density_level=density_level, vehicle_count=count_total, camera_id=camera_id)
        db.add(new_density)
        db.commit()
        db.refresh(new_density)
    
    def process_frame(self, frame, route_id, fps_window, count_window, chart_window):
        new_frame = None
        if self.counting_regions:
            prev_time = time.time()
            # -------- Object Detection And Tracking -------- #
            results = self.model.track(frame, conf=self.confidence, iou=self.iou, persist=True,
                                        device=self.device, tracker=self.tracker_type, show=False)
            
            # -------- Display Selective Area And Counting -------- #
            new_frame = self.display_selective_area(
                results, frame, self.model.names)

            # -------- Plot Graph -------- #

            if time.time() - self.chart_start_time > 5:  # Update data every 5 second
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.chart_start_time = time.time()
                self.is_display_chart = True

                for region in self.counting_regions:
                    for cls, count in region['counts'].items():
                        # Thêm dữ liệu mới vào DataFrame
                        new_row = {'Time': timestamp, 'Vehicle Count': count,
                                    'Vehicle Type': cls, 'Region ID': region['id']}
                        count_total = sum(region['counts'].values())
                        self.data_graph_df = pd.concat(
                            [self.data_graph_df, pd.DataFrame([new_row])], ignore_index=True)
                        
                        with get_db() as db: 
                            self.save_detection_data(db, timestamp, cls, count, route_id)
                            self.update_region_density(db, timestamp, count_total, route_id)

                # self.plot_graph(placeholder=chart_window)
            
            # density_level = self.calculate_density()
            
            # # Display the FPS
            # curr_time = time.time()
            # fps = 1 / (curr_time - prev_time)
            # print("FPS: ", fps)
            
            # cv2.putText(new_frame, "FPS: {:.2f}".format(fps), (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # density_level_x_position = 10 + 50  # 10 là vị trí x của FPS, 200 là khoảng cách mong muốn

            # # Hiển thị Density Level ngay sau FPS
            # cv2.putText(new_frame, f"Density Level: {density_level}", (density_level_x_position, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            col1, col2, col3 = fps_window.columns(3, gap="medium")

            density_level = self.calculate_density()
            
            if self.is_display_chart:
                self.plot_graph(placeholder=chart_window)
                self.is_display_chart = False
            
            # Display the FPS
            curr_time = time.time()
            fps = round(1 / (curr_time - prev_time))
            prev_time = curr_time
            print("FPS: ", fps)
            
            col1.markdown(f"<p style='text-align: center; color: red; font-size: 20px; font-weight: bold;'>FPS: {fps}</p>", unsafe_allow_html=True)
            col3.markdown(f"<p style='text-align: center; color: red; font-size: 20px; font-weight: bold;'>Size: {frame.shape[1]}x{frame.shape[0]}</p>", unsafe_allow_html=True)
            
            
            for region in self.counting_regions:
                region_id = region["id"]
                density_level_value = "{:.2f}".format(round(density_level[region_id], 2))
                density_level_str = 'low' if density_level[region_id] < 1/3 else 'medium' if density_level[region_id] < 2/3 else 'high'
                col2.markdown(f"<p style='text-align: center; color: red; font-size: 20px; font-weight: bold;'>Density Level: {density_level_value} | {density_level_str} </p>", unsafe_allow_html=True)

                CLASS_NAMES = self.model.names.values()
                
                for cls in CLASS_NAMES:
                    if cls not in region["counts"]:
                        region["counts"][cls] = 0
                    else:
                        region["counts"][cls] = region["counts"][cls]

                region_label = ' | '.join(
                    [f"{cls}: {count}" for cls, count in region["counts"].items()])
                
                total_count = sum(region["counts"].values())
                count_window.write(f"""\n
                                    **[Region ID]**: {region["id"]}
                                    \n
                                    - **[Counts]**: {region_label}
                                    \n
                                    - **[Total Count]**: {total_count}
                                    \n
                                    """)
                offset_x_density = int(frame.shape[1]/2) + 30
                cv2.putText(new_frame, f"Density Level: {density_level_value}", (offset_x_density, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
            # frame_window.image(new_frame,
            #                    caption='Detected Video',
            #                    channels="BGR",
            #                    use_column_width=True)

            # del results
            # del new_frame
            # del frame
            # gc.collect()
            
            return new_frame
            
        else:
            st.error(
                "No counting regions found. Please create counting regions.")
    
    
    
    def process_image(self, image):
        try:
            res = self.model.predict(
                image, conf=self.confidence, iou=self.iou, device=self.device)
            boxes = res[0].boxes
            res_plotted = res[0].plot()

            st.image(res_plotted, channels="BGR",
                               caption='Detected Image', use_column_width=True)

            with st.expander("Detection Results"):

                clss = boxes.cls.cpu().tolist()
                classes_count = Counter(clss)
                
                st.text("Vehicle Counts:")
                for cls, count in classes_count.items():
                    st.write(f"- {self.get_class_names()[cls]}: {count}")
                
                st.text("Bounding Boxes:")
                for box in boxes:   
                    st.write(box.data)
        except Exception as ex:
            st.error("Error occurred during detection.")
            st.error(ex)
            
if __name__ == "__main__":
    """"""