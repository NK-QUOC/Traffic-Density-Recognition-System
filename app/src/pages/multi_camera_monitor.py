import ast
from datetime import timedelta
import gc
from pathlib import Path
from queue import Queue
import threading
from threading import Thread
from multiprocessing import Process
import threading
import cv2
import folium.features
import folium.vector_layers
import numpy as np
import shapely.wkt
import streamlit as st
import time
from vidgear.gears import CamGear
from PIL import Image

from ..utils import Page
from ..utils import TrafficMonitor
import src.settings as settings

import folium
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from streamlit_folium import st_folium, folium_static
from src.database.database import get_db
from src.database.models import Camera, TrafficDensity
from sqlalchemy import and_, func
import geopandas as gpd
import shapely

class MultiCameraMonitor(Page):
    def __init__(self, state):
        self.state = state
        self.traffic_monitor = TrafficMonitor()
        self.traffic_monitors = []
        self.routes = {}
        self.urls = []
        
        self.threads = []
        self.queues = []
        self.map_queue = Queue()
                    
    @st.experimental_dialog(title="Update Camera", width="large")
    def update_camera(self, camera_id):
        input_holder = st.empty()
        progress = st.empty()
        with get_db() as db:
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
        
        if 'update_button' not in st.session_state:
            st.session_state['update_button'] = False
        
        if camera:                
            with input_holder.form(key="update_camera_form"):
                camera_name = st.text_input("Camera Name", value=camera.name)    
                camera_url = st.text_input("Stream URL", value=camera.stream_url)
                camera_roi = st.text_area("Count ROI", value=camera.count_roi[0], disabled=True)
                
                is_youtube_url, _ = self.traffic_monitor.is_youtube_url(camera_url)
                if not is_youtube_url:
                    st.warning("Please enter a direct video stream URL")
                
                update_button = st.form_submit_button("Update")
                
                progress.progress(0)
                vid_cap = CamGear(source=camera_url, logging=True, stream_mode=True, **{"STREAM_RESOLUTION": "720p"}).start()
                frame = vid_cap.read()
                vid_cap.stop()
                progress.progress(50)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                progress.progress(100)
                points = self.traffic_monitor.draw_tool(image)
                if points == [(0, 0), (image.width, 0), (image.width,image.height), (0, image.height)]:
                    points = []
                
                if points:
                    st.write("ROI Points: ", points)
                                    
                if update_button:
                    st.session_state['update_button'] = True

                if st.session_state['update_button']:
                    
                    is_changed = False
                    if camera_name != camera.name:
                        is_changed = True
                    if camera_url != camera.stream_url:
                        is_changed = True
                    if camera_roi != str(points) and points != []:
                        camera_roi = points
                        is_changed = True
                    if is_changed:
                        st.session_state["update_camera"] = {
                            "camera_id": camera_id,
                            "camera_name": camera_name,
                            "camera_url": camera_url,
                            "camera_roi": camera_roi
                        }
                        st.rerun()
                    else:
                        for region in self.traffic_monitor.counting_regions:
                            print("Region: ", region['points'])   
                        st.info("No changes detected")
                        
    def params_config(self):
        with st.expander("Monitor Configuration", expanded=True):
            model_type = st.selectbox(
                "Select Model", ['yolov9', 'yolov8'],
                index=0,
                key='model_type')

            confidence = st.slider(
                "Confidence Threshold", 0.0, 1.0, 0.25, 0.01, key='confidence')
            iou = st.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01, key='iou')

            tracker_radio = st.radio(
                "Tracker", ('bytetrack',))
            tracker_type = settings.bytetrack if tracker_radio == 'bytetrack' else settings.botsort

            if model_type == 'yolov9':
                mode_path = Path(settings.YOLOv9_MODEL)
            elif model_type == 'yolov8':
                mode_path = Path(settings.YOLOv8_MODEL)

            num_routes = st.slider(
                "Number of Routes to Monitor", 1, 3, 1, key='num_routes')

            for num in range(num_routes):
                traffic_monitor = TrafficMonitor()
                traffic_monitor.load_model(mode_path)
                traffic_monitor.confidence = confidence
                traffic_monitor.iou = iou
                traffic_monitor.tracker_type = tracker_type
                self.traffic_monitors.append(traffic_monitor)

            with get_db() as db:
                cameras = db.query(Camera).all()
                camera_options = {camera.name: camera.id for camera in cameras}
                camera_names = list(camera_options.keys())

            selected_cameras = []

            self.routes = {}

            for i in range(num_routes):
                
                available_cameras = [name for name in camera_names if name not in selected_cameras]
                
                camera_name = st.selectbox(
                    "Select a location:",
                    options=available_cameras,
                    index=0,
                    key=f'camera_{i}'
                )
                
                if camera_name:
                    selected_cameras.append(camera_name)
                    self.routes[camera_options[camera_name]] = camera_name

                if "update_camera" not in st.session_state:
                    update_button = st.button(f"Update {camera_name}", key=f'update_{i}')
                    if update_button:
                        self.update_camera(camera_options[camera_name])
                else:
                    print(st.session_state["update_camera"]['camera_id'])
                    print(st.session_state["update_camera"]['camera_name'])
                    print(st.session_state["update_camera"]['camera_url'])
                    print(st.session_state["update_camera"]['camera_roi'])
                    
                    with get_db() as db:
                        camera = db.query(Camera).filter(Camera.id == st.session_state["update_camera"]['camera_id']).first()
                        if camera:
                            camera.name = st.session_state["update_camera"]['camera_name']
                            camera.stream_url = st.session_state["update_camera"]['camera_url']
                            camera.count_roi = st.session_state["update_camera"]['camera_roi']
                            db.commit()
                    
                    if "update_button" in st.session_state:
                        st.session_state["update_button"] = False
                    st.toast(f"Updated {st.session_state['update_camera']['camera_name']} successfully", icon="🎉")
                    time.sleep(2)
                    del st.session_state["update_camera"]
                    st.rerun()
                
    def write(self):
        st.title("Real-Time Traffic Route Monitor")
        self.params_config()
        rois = []
        if self.traffic_monitors and self.routes:
            with get_db() as db:
                for camera_id in self.routes.keys():
                    camera = db.query(Camera).filter(
                        Camera.id == camera_id).first()
                    if camera:
                        self.urls.append(camera.stream_url)
                        roi = camera.count_roi
                        rois.append(roi)
 
            for i, traffic_monitor in enumerate(self.traffic_monitors):
                points_as_tuples = [[tuple(point) for point in sublist] for sublist in rois[i]]
                traffic_monitor.create_counting_regions(points_as_tuples)
                traffic_monitor.calculate_area_roi(traffic_monitor.counting_regions)

            st.session_state['detector_running'] = False 

            if st.button("Start Monitor"):
                st.session_state['detector_running'] = True
                st.write("Monitoring Routes...")

            if st.button("Stop Monitor"):
                st.session_state['detector_running'] = False

            self.monitor()

        else:
            st.write("Please enter YouTube URLs for all routes")
            
    def get_latest_traffic_density(self, db):
        max_timestamp_subquery = db.query(
            TrafficDensity.camera_id,
            func.max(TrafficDensity.timestamp).label('max_timestamp')
        ).group_by(TrafficDensity.camera_id).subquery()

        # Query to get the latest traffic density data for each camera
        latest_traffic = db.query(TrafficDensity).join(
            max_timestamp_subquery,
            and_(
                TrafficDensity.camera_id == max_timestamp_subquery.c.camera_id,
                TrafficDensity.timestamp == max_timestamp_subquery.c.max_timestamp
            )
        ).all()

        return latest_traffic

    def get_average_traffic_density_last_minute(self, db):
        # query to get the max timestamp in the database
        max_timestamp = db.query(func.max(TrafficDensity.timestamp)).scalar()
        one_minute_ago = max_timestamp - timedelta(minutes=1)
        
        # query to get the average traffic density for each camera in the last minute
        latest_traffic = db.query(
            TrafficDensity.camera_id,
            func.avg(TrafficDensity.density_level).label('avg_density_level'),
            func.sum(TrafficDensity.vehicle_count).label('total_vehicle_count'),
            func.max(TrafficDensity.timestamp).label('max_timestamp')
        ).filter(
            TrafficDensity.timestamp >= one_minute_ago,
            TrafficDensity.timestamp <= max_timestamp
        ).group_by(TrafficDensity.camera_id).all()
        
        return latest_traffic
    
    def get_traffic_data(self):
        with get_db() as db:
            latest_traffic_data = self.get_average_traffic_density_last_minute(db)
            camera_data = db.query(Camera).all()
        return latest_traffic_data, camera_data

    def create_map(self, latest_traffic_data, camera_data):
        camera_dict = {camera.id: camera for camera in camera_data}
        traffic_density_dict = {traffic.camera_id: traffic for traffic in latest_traffic_data}
                
        density_map = folium.Map(location=[16.07415081156636, 108.2167160511017], zoom_start=20)

        def get_polygon_color(density_level):
            return 'green' if density_level == 'low' else 'orange' if density_level == 'medium' else 'red'
        
        
        gpdfs = []
        for camera in camera_data:
            folium.Marker(
                location=[camera.latitude, camera.longitude],
                popup=f"<i>{camera.name}</i>",
                tooltip=camera.name,
            ).add_to(density_map)

            traffic = traffic_density_dict.get(camera.id)
            density_level = traffic.avg_density_level if traffic else 0
            density_level_str = "low" if density_level < 1/3 else "medium" if density_level < 2/3 else "high"
            count = traffic.total_vehicle_count if traffic else 0
            color = get_polygon_color(density_level_str)

            if camera.area:
                wkt = "POLYGON ((" + \
                    ", ".join(f"{lon} {lat}" for lon, lat in camera.area) + "))"
                polygon = shapely.wkt.loads(wkt)
                gpdf = gpd.GeoDataFrame(
                    geometry=[polygon]).set_crs("EPSG:4326")
                gpdfs.append([gpdf, camera.name, color, count])

        for gpdf, name, color, count in gpdfs:
            fg = folium.FeatureGroup(name=name)
            fg.add_child(folium.GeoJson(
                gpdf,
                style_function=lambda x, col=color: {
                    "fillColor": col,
                    "color": col,
                    "fillOpacity": 0.13,
                    "weight": 2,
                },
            ))
            fg.add_child(folium.Popup(f"{name} - Vehicle Count: {count}"))
            density_map.add_child(fg)

        control = folium.LayerControl(collapsed=False)
        density_map.add_child(control)

        for traffic in latest_traffic_data:
            camera = camera_dict.get(traffic.camera_id)
            if camera:
                density_level = traffic.avg_density_level
                density_level_str = "low" if density_level < 1/3 else "medium" if density_level < 2/3 else "high"
                color = get_polygon_color(density_level_str)
                folium.Marker(
                    location=[camera.latitude, camera.longitude],
                    popup=f"<i>{camera.name} - Average Traffic Density: {traffic.avg_density_level} - Last Update : {traffic.max_timestamp} </i>",
                    tooltip=camera.name,
                    icon=folium.Icon(color=color)
                ).add_to(density_map)

        return density_map
    
    def update_map(self):
        while st.session_state['detector_running']:
            print("Updating map...")
            latest_traffic_data, camera_data = self.get_traffic_data()
            density_map = self.create_map(latest_traffic_data, camera_data)
            self.map_queue.put(density_map)
            
            del latest_traffic_data
            del camera_data
            del density_map
            gc.collect()
            time.sleep(60)       
            
    def VideoFeed(self, url, i, route_id, fps_window, count_window, chart_window):
        vid_cap = CamGear(source=url, logging=True, stream_mode=True, **{"STREAM_RESOLUTION": "720p"}).start()
        frame = vid_cap.read()
        frame_height, frame_width = frame.shape[:2]
        frame_rate = vid_cap.framerate

        # Define the codec and create VideoWriter object
        output_filename = self.traffic_monitors[i].create_unique_filename(settings.OUTPUT_VIDEO_DIR / 'multicamera/output', '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_filename, fourcc, 30.0, (frame_width, frame_height))
        
        while st.session_state['detector_running']:
            frame = vid_cap.read()
            if frame is None:
                break
            processed_frame = self.traffic_monitors[i].process_frame(frame, route_id, fps_window, count_window, chart_window)
            out.write(processed_frame)
            self.queues[i].put(item = processed_frame)
            time.sleep(1/frame_rate) 
            
            del frame
            del processed_frame
            gc.collect()  
                            
        vid_cap.stop()

    def monitor(self):
        tab1, tab2 = st.tabs(["Realtime Detection", "Map"])
            
        with tab1:
            tab1.subheader("Realtime Detection")
            placeholders = []
            fps_placeholders = []
            count_placeholders = []
            chart_placeholders = []
            expanders = []
            for i, (route_id, route_name) in enumerate(self.routes.items()):
                expander = tab1.expander(f"Route {i + 1}: {route_name}")
                expanders.append(expander)
                with expander:
                    tab_detection, tab_chart = st.tabs(["Detection", "Chart"])
                    with tab_detection:
                        placeholders.append(st.image([]))
                        fps_placeholders.append(st.empty())
                        count_placeholders.append(st.empty())
                    with tab_chart:
                        chart_placeholders.append(st.empty())
                
            if st.session_state['detector_running']:
                self.queues = [Queue() for _ in range(len(self.urls))]
                for i, (url, route_id) in enumerate(zip(self.urls, self.routes.keys())):
                    thead = threading.Thread(target=self.VideoFeed, args=(url, i, route_id, fps_placeholders[i], count_placeholders[i], chart_placeholders[i]), daemon=True)
                    ctx = get_script_run_ctx()
                    add_script_run_ctx(thead, ctx)
                    thead.start()
                    self.threads.append(thead)
                
                t_map = threading.Thread(target=self.update_map, daemon=True)
                ctx_map = get_script_run_ctx()
                add_script_run_ctx(t_map, ctx_map)
                t_map.start()
                self.threads.append(t_map)
                
                map_placeholder = tab2.empty()
                map_placeholder.subheader("Map")
                start_time = time.time()
                while st.session_state['detector_running']:
                    for i in range(len(self.routes)):
                        frame = self.queues[i].get()
                        placeholders[i].image(frame, channels="BGR", use_column_width=True)
                        
                    if time.time() - start_time > 60 and not self.map_queue.empty():
                        print("Updated map")
                        try:
                            density_map = self.map_queue.get()
                            with map_placeholder:
                                folium_static(density_map, width=800, height=400)
                            start_time = time.time()
                        except:
                            pass
                    
            else:
                st.write("Monitoring Stopped")
                for thread in self.threads:
                    thread.join()
                gc.collect()
                
        with tab2:
            if not st.session_state['detector_running']:
                st.subheader("Map")
                latest_traffic_data, camera_data = self.get_traffic_data()
                density_map = self.create_map(latest_traffic_data, camera_data)
                folium_static(density_map, width=800, height=400)

if __name__ == "__main__":
    from src.state import provide_state

    @provide_state()
    def main(state=None):
        page = MultiCameraMonitor(state)
        page.write()

    main()