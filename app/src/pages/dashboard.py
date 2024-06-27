import pandas as pd
import streamlit as st
from ..utils import Page
from ..utils import TrafficMonitor
from sqlalchemy.orm import Session
from src.database.database import get_db
from src.database.models import TrafficDensity, Vehicle, Camera
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime
import numpy as np


class Dashboard(Page):
    def __init__(self, state=None):
        self.state = state
        self.traffic_monitor = TrafficMonitor(self.state)

    def write(self):
        st.title("Dashboard")

        with get_db() as db:
            query = db.query(Vehicle).all()
            cameras = db.query(Camera).all()
            data = [{'timestamp': q.timestamp, 'vehicle_type': q.vehicle_type,
                     'count': q.count, 'camera_id': q.camera_id} for q in query]
            df = pd.DataFrame(data)
            
            vehicle_query = db.query(Vehicle).all()
            cameras = db.query(Camera).all()
            traffic_density_query = db.query(TrafficDensity).all()

            vehicle_data = [{'timestamp': q.timestamp, 'vehicle_type': q.vehicle_type,
                             'count': q.count, 'camera_id': q.camera_id} for q in vehicle_query]
            df_vehicles = pd.DataFrame(vehicle_data)

            density_data = [{'timestamp': q.timestamp, 'density_level': q.density_level,
                             'vehicle_count': q.vehicle_count, 'camera_id': q.camera_id} for q in traffic_density_query]
            df_density = pd.DataFrame(density_data)

        df_vehicles['timestamp'] = pd.to_datetime(df_vehicles['timestamp'])
        df_density['timestamp'] = pd.to_datetime(df_density['timestamp'])

        # Add date and camera selection widgets
        date_cols = st.columns(2)
        start_date = date_cols[0].date_input(
            "Start date", df_vehicles['timestamp'].min().date())
        end_date = date_cols[1].date_input(
            "End date", df_vehicles['timestamp'].max().date())
        camera_options = {camera.name: camera.id for camera in cameras}
        selected_camera = st.selectbox(
            "Select Camera", options=list(camera_options.keys()))

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date) + \
            pd.Timedelta(days=1)  # Add 1 day to include end date

        # Filter data based on time period and selected camera
        mask_vehicles = (df_vehicles['timestamp'] >= start_date) & (df_vehicles['timestamp'] < end_date) & (
            df_vehicles['camera_id'] == camera_options[selected_camera])
        df_vehicles_filtered = df_vehicles.loc[mask_vehicles].copy()

        mask_density = (df_density['timestamp'] >= start_date) & (df_density['timestamp'] < end_date) & (
            df_density['camera_id'] == camera_options[selected_camera])
        df_density_filtered = df_density.loc[mask_density].copy()

        if df_vehicles_filtered.empty:
            st.warning("No data available for selected date and camera.")
            return

        if df_density_filtered.empty:
            st.warning("No density data available for selected date and camera.")
            return
        
        df_vehicles_filtered['timestamp'] = pd.to_datetime(df_vehicles_filtered['timestamp'])
        df_vehicles_filtered.set_index('timestamp', inplace=True)
        
        df_density_filtered['timestamp'] = pd.to_datetime(df_density_filtered['timestamp'])
        df_density_filtered.set_index('timestamp', inplace=True)

        # Identify numeric columns
        numeric_cols_vehicles = df_vehicles_filtered.select_dtypes(
            include=[np.number]).columns.tolist()
        numeric_cols_density = df_density_filtered.select_dtypes(include=[np.number]).columns.tolist()

        # Group data by minute and hour
        df_vehicles_filtered_grouped_minute = df_vehicles_filtered.groupby(
            [pd.Grouper(freq='Min'), 'vehicle_type'])[numeric_cols_vehicles].sum().reset_index()
        df_vehicles_filtered_grouped_hour = df_vehicles_filtered.groupby(
            [pd.Grouper(freq='h'), 'vehicle_type'])[numeric_cols_vehicles].sum().reset_index()
        
        df_filtered_grouped_minute_density = df_density_filtered.resample('Min')[numeric_cols_density].mean().reset_index()
        df_filtered_grouped_hour_density = df_density_filtered.resample('h')[numeric_cols_density].mean().reset_index()

        
        df_vehicles_hourly = df_vehicles_filtered.resample('h')[numeric_cols_vehicles].agg(
            {'count': ['min', 'max', 'mean']}).reset_index()
        df_vehicles_hourly.columns = ['timestamp',
                             'min_count', 'max_count', 'avg_count']
        
        # Find min, max and average vehicle count
        min_index = df_vehicles_hourly['min_count'].idxmin()
        max_index = df_vehicles_hourly['max_count'].idxmax()
        min_row = df_vehicles_hourly.loc[min_index]
        max_row = df_vehicles_hourly.loc[max_index]

        # Display summary statistics
        st.subheader("Summary Statistics")
        st.write(f"Minimum Vehicle Count: **{int(min_row['min_count'])}** at **{min_row['timestamp'].strftime('%H:%M:%S %Y-%m-%d')}**")
        st.write(f"Maximum Vehicle Count: **{int(max_row['max_count'])}** at **{max_row['timestamp'].strftime('%H:%M:%S %Y-%m-%d')}**")
        st.write(f"Average Vehicle Count: **{df_vehicles_filtered['count'].mean():.2f}**")
        
        # Line chart for vehicle count per minute and hour
        fig_minute = px.line(df_vehicles_filtered_grouped_minute, x='timestamp', y='count',
                             color='vehicle_type', title='Vehicle Count per Minute', markers=True)
        fig_minute.update_xaxes(
            rangeslider_visible=True,
            rangeslider=dict(
                autorange=True
            ),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            )
        )

        fig_hour = px.line(df_vehicles_filtered_grouped_hour, x='timestamp', y='count',
                           color='vehicle_type', title='Vehicle Count per Hour', markers=True)
        fig_hour.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=1, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            )
        )

        # Line chart for traffic density per minute and hour
        fig_minute_density = px.line(df_filtered_grouped_minute_density, x='timestamp', y='density_level',
                                     title='Average Traffic Density per Minute', markers=True)
        fig_minute_density.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            )
        )
        fig_minute_density.add_shape(
            type="line",
            x0=df_filtered_grouped_minute_density['timestamp'].min(),
            x1=df_filtered_grouped_minute_density['timestamp'].max(),
            y0=1/3,
            y1=1/3,
            line=dict(color="green", dash="dash")
        )
        fig_minute_density.add_shape(
            type="line",
            x0=df_filtered_grouped_minute_density['timestamp'].min(),
            x1=df_filtered_grouped_minute_density['timestamp'].max(),
            y0=2/3,
            y1=2/3,
            line=dict(color="red", dash="dash")
        )
        
        fig_hour_density = px.line(df_filtered_grouped_hour_density, x='timestamp', y='density_level',
                                   title='Average Traffic Density per Hour', markers=True)
        fig_hour_density.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=1, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            )
        )
        fig_hour_density.add_shape(
            type="line",
            x0=df_filtered_grouped_minute_density['timestamp'].min(),
            x1=df_filtered_grouped_minute_density['timestamp'].max(),
            y0=1/3,
            y1=1/3,
            line=dict(color="green", dash="dash")
        )
        fig_hour_density.add_shape(
            type="line",
            x0=df_filtered_grouped_minute_density['timestamp'].min(),
            x1=df_filtered_grouped_minute_density['timestamp'].max(),
            y0=2/3,
            y1=2/3,
            line=dict(color="red", dash="dash")
        )
        
        # Pie chart for vehicle type distribution and heatmap for traffic density
        df_vehicles_vehicle_type = df_vehicles_filtered.groupby(
            'vehicle_type')[numeric_cols_vehicles].sum().reset_index()
        fig_pie = px.pie(df_vehicles_vehicle_type, names='vehicle_type',
                         values='count', title='Vehicle Type Distribution')

        df_vehicles_filtered['day_of_week'] = df_vehicles_filtered.index.dayofweek
        df_vehicles_filtered['hour'] = df_vehicles_filtered.index.hour
        df_vehicles_heatmap = df_vehicles_filtered.groupby(['day_of_week', 'hour'])[
            numeric_cols_vehicles].sum().reset_index()
        # Use pivot_table instead of pivot
        df_vehicles_pivot = df_vehicles_heatmap.pivot_table(
            index='day_of_week', columns='hour', values='count')
        fig_heatmap = px.imshow(df_vehicles_pivot, labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Vehicle Count'},
                                title='Traffic Density Heatmap', aspect='auto')

        linechart_cols = st.columns(2)
        linechart_cols[0].plotly_chart(fig_minute)
        linechart_cols[1].plotly_chart(fig_hour)

        linechart_density_cols = st.columns(2)
        linechart_density_cols[0].plotly_chart(fig_minute_density)
        linechart_density_cols[1].plotly_chart(fig_hour_density)

        pie_cols = st.columns(2)
        pie_cols[0].plotly_chart(fig_pie)
        pie_cols[1].plotly_chart(fig_heatmap)

        fig_combined = make_subplots(rows=1, cols=2, subplot_titles=[
                                     'Bar Chart', 'Line Chart'])

        colors = {
            'min_count': 'rgba(0, 0, 255, 0.7)',
            'max_count': 'rgba(255, 0, 0, 0.7)',
            'avg_count': 'rgba(0, 255, 0, 0.7)'
        }
        
        # Bar chart with min, max, and average values
        fig_combined.add_trace(go.Bar(x=df_vehicles_hourly['timestamp'], y=df_vehicles_hourly['min_count'],
                               name='Min Count', marker_color=colors['min_count'], showlegend=False), row=1, col=1)
        fig_combined.add_trace(go.Bar(x=df_vehicles_hourly['timestamp'], y=df_vehicles_hourly['max_count'],
                               name='Max Count', marker_color=colors['max_count'], showlegend=False), row=1, col=1)
        fig_combined.add_trace(go.Bar(x=df_vehicles_hourly['timestamp'], y=df_vehicles_hourly['avg_count'],
                               name='Average Count', marker_color=colors['avg_count'], showlegend=False), row=1, col=1)

        # Line chart with min, max, and average values
        fig_combined.add_trace(go.Scatter(x=df_vehicles_hourly['timestamp'], y=df_vehicles_hourly['avg_count'], mode='lines+markers', line=dict(
            color=colors['avg_count']), name='Average Count'), row=1, col=2)
        fig_combined.add_trace(go.Scatter(x=df_vehicles_hourly['timestamp'], y=df_vehicles_hourly['min_count'],
                               mode='lines+markers', line=dict(color=colors['min_count']), name='Min Count'), row=1, col=2)
        fig_combined.add_trace(go.Scatter(x=df_vehicles_hourly['timestamp'], y=df_vehicles_hourly['max_count'],
                               mode='lines+markers', line=dict(color=colors['max_count']), name='Max Count'), row=1, col=2)

        fig_combined.update_layout(title='Min, Max, and Average Vehicle Count per Hour',
                                   xaxis_title='Thời gian', yaxis_title='Số lượng',
                                   showlegend=True, legend=dict(x=1.0, y=1.1))

        st.plotly_chart(fig_combined)


if __name__ == "__main__":
    from src.state import provide_state

    @provide_state()
    def main(state=None):
        page = Dashboard(state)
        page.write()

    main()
