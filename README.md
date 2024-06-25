# Traffic Density Recognition System

## Introduction

This project aims to build a traffic density recognition system using YOLOv9, trained on a custom dataset. The system can detect vehicles from public cameras and YouTube livestreams. The web application, built entirely with Streamlit, offers several key features including multimedia detection and real-time traffic monitoring from multiple cameras simultaneously.

## Features

1. **Multimedia Detection**: Upload images, videos, or provide URLs (including YouTube, webcam, or RTSP) for vehicle detection and display the predictions.

2. **Multi-camera-monitor**: Detect and display real-time traffic from multiple YouTube livestreams on the same page.
3. **Dashboard**: Display vehicle count statistics in graphs, showing hourly or daily traffic density.

## Project Structure

```
app/
    main.py
    src/
        assets/             # Static assets like images, videos, etc.
        cfg/                # Configuration files
        database/           # Database models and queries
        weights/            # Pre-trained YOLOv9 weights
        pages/              # Streamlit pages
        settings.py         # Application settings
        state.py            # State management for the application
        utils.py            # Utility functions
```

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/traffic-density-recognition-system.git
cd traffic-density-recognition-system/app
```

2. **Install dependencies:**

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

3. **Setup the database:**

Install PostgreSQL and create a database, then adjust the settings in the settings.py file:

```python
POSTGRES_SERVER='' # PostgreSQL server address
POSTGRES_PORT=     # Port to connect to the PostgreSQL server
POSTGRES_USER=''   # PostgreSQL username
POSTGRES_PASSWORD='' # Password of the PostgreSQL user
POSTGRES_DB=''     # Name of the PostgreSQL database
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app/main.py
```

