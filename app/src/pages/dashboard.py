import pandas as pd
import streamlit as st
from ..utils import Page
from ..utils import TrafficMonitor
from sqlalchemy.orm import Session
from src.database.database import get_db
from src.database.models import Vehicle, Camera
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

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Thêm các widget chọn ngày và chọn camera
        date_cols = st.columns(2)
        start_date = date_cols[0].date_input(
            "Start date", df['timestamp'].min().date())
        end_date = date_cols[1].date_input(
            "End date", df['timestamp'].max().date())
        camera_options = {camera.name: camera.id for camera in cameras}
        selected_camera = st.selectbox(
            "Select Camera", options=list(camera_options.keys()))

        # Chuyển đổi các giá trị ngày được chọn thành datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date) + \
            pd.Timedelta(days=1)  # Bao gồm ngày kết thúc

        # Lọc dữ liệu dựa trên khoảng thời gian và camera được chọn
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] < end_date) & (
            df['camera_id'] == camera_options[selected_camera])
        df_filtered = df.loc[mask].copy()

        if df_filtered.empty:
            st.warning("No data available for selected date and camera.")
            return

        df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
        df_filtered.set_index('timestamp', inplace=True)

        # Identify numeric columns
        numeric_cols = df_filtered.select_dtypes(
            include=[np.number]).columns.tolist()

        # Biểu đồ dòng cho số lượng xe theo phút và giờ
        df_filtered_grouped_minute = df_filtered.groupby(
            [pd.Grouper(freq='Min'), 'vehicle_type'])[numeric_cols].sum().reset_index()
        df_filtered_grouped_hour = df_filtered.groupby(
            [pd.Grouper(freq='h'), 'vehicle_type'])[numeric_cols].sum().reset_index()
        
        df_hourly = df_filtered.resample('h')[numeric_cols].agg(
            {'count': ['min', 'max', 'mean']}).reset_index()
        df_hourly.columns = ['timestamp',
                             'min_count', 'max_count', 'avg_count']
        
        # Tìm thông tin chi tiết cho giá trị min và max
        min_index = df_hourly['min_count'].idxmin()
        max_index = df_hourly['max_count'].idxmax()
        min_row = df_hourly.loc[min_index]
        max_row = df_hourly.loc[max_index]

        # Hiển thị các giá trị thống kê với thông tin chi tiết
        st.subheader("Summary Statistics")
        st.write(f"Minimum Vehicle Count: **{int(min_row['min_count'])}** at **{min_row['timestamp'].strftime('%H:%M:%S %Y-%m-%d')}**")
        st.write(f"Maximum Vehicle Count: **{int(max_row['max_count'])}** at **{max_row['timestamp'].strftime('%H:%M:%S %Y-%m-%d')}**")
        st.write(f"Average Vehicle Count: **{df_filtered['count'].mean():.2f}**")
        
        # Biểu đồ dòng cho số lượng xe theo phút và giờ
        
        fig_minute = px.line(df_filtered_grouped_minute, x='timestamp', y='count',
                             color='vehicle_type', title='Vehicle Count per Minute', markers=True)
        fig_minute.update_xaxes(
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

        fig_hour = px.line(df_filtered_grouped_hour, x='timestamp', y='count',
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

        # Biểu đồ tròn cho tỷ lệ loại xe
        df_vehicle_type = df_filtered.groupby(
            'vehicle_type')[numeric_cols].sum().reset_index()
        fig_pie = px.pie(df_vehicle_type, names='vehicle_type',
                         values='count', title='Vehicle Type Distribution')

        df_filtered['day_of_week'] = df_filtered.index.dayofweek
        df_filtered['hour'] = df_filtered.index.hour
        df_heatmap = df_filtered.groupby(['day_of_week', 'hour'])[
            numeric_cols].sum().reset_index()
        # Use pivot_table instead of pivot
        df_pivot = df_heatmap.pivot_table(
            index='day_of_week', columns='hour', values='count')
        fig_heatmap = px.imshow(df_pivot, labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Vehicle Count'},
                                title='Traffic Density Heatmap', aspect='auto')

        linechart_cols = st.columns(2)
        linechart_cols[0].plotly_chart(fig_minute)
        linechart_cols[1].plotly_chart(fig_hour)

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
        # Biểu đồ cột với giá trị min, max và average
        fig_combined.add_trace(go.Bar(x=df_hourly['timestamp'], y=df_hourly['min_count'],
                               name='Min Count', marker_color=colors['min_count'], showlegend=False), row=1, col=1)
        fig_combined.add_trace(go.Bar(x=df_hourly['timestamp'], y=df_hourly['max_count'],
                               name='Max Count', marker_color=colors['max_count'], showlegend=False), row=1, col=1)
        fig_combined.add_trace(go.Bar(x=df_hourly['timestamp'], y=df_hourly['avg_count'],
                               name='Average Count', marker_color=colors['avg_count'], showlegend=False), row=1, col=1)

        # Biểu đồ dòng với giá trị average và các đường biểu diễn giá trị min và max
        fig_combined.add_trace(go.Scatter(x=df_hourly['timestamp'], y=df_hourly['avg_count'], mode='lines+markers', line=dict(
            color=colors['avg_count']), name='Average Count'), row=1, col=2)
        fig_combined.add_trace(go.Scatter(x=df_hourly['timestamp'], y=df_hourly['min_count'],
                               mode='lines+markers', line=dict(color=colors['min_count']), name='Min Count'), row=1, col=2)
        fig_combined.add_trace(go.Scatter(x=df_hourly['timestamp'], y=df_hourly['max_count'],
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
