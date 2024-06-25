import datetime
from sqlalchemy.orm import Session
from .database import engine
from .models import Base, Camera, Vehicle, TrafficDensity

def init_db():
    Base.metadata.create_all(bind=engine)
    
    # Check if there is any data in the Camera table
    db = Session(bind=engine)
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if db.query(Camera).count() == 0:
            # Add initial data
            initial_cameras = [
                Camera(name="Camera CT-NguyenHue", 
                       latitude=16.074145656874663, 
                       longitude=108.21628153324129, 
                       area = [[108.216019,16.074025 ], [108.216298, 16.074076], [108.216314,16.074012], [108.216032, 16.07396], [ 108.216019,  16.074025]],
                       count_roi = [[(421, 247), (800, 380), (951, 258), (692, 194), (421, 247)]],
                       stream_url="https://www.youtube.com/watch?v=Fu3nDsqC1J0"),
                Camera(name="Camera CS-BenhVienC", 
                       latitude=16.074181739713826, 
                       longitude=108.2164853811264, 
                       area = [[108.216491,16.074117], [108.21674, 16.074171  ],[108.216759, 16.074092 ],[108.216496,16.07405 ],[108.216491, 16.074117]],
                       count_roi = [[(626, 245), (869, 231), (1184, 388), (791, 560), (626, 245)]],
                       stream_url="https://www.youtube.com/watch?v=fW5e8xsLnBc"),
                Camera(name="Camera ViewCT-BenhVienC", 
                       latitude=16.07404256301241, 
                       longitude=108.2157987356186, 
                       area = [[108.215694,16.073968], [108.215962, 16.074017 ],[ 108.215986, 16.073934], [108.21571, 16.073888],[108.215694,16.073968 ]],
                       count_roi = [[(357, 636), (597, 535), (677, 574), (457, 706), (357, 636)]],
                       stream_url="https://www.youtube.com/watch?v=b6fkug3AmH4"),
            ]
            db.add_all(initial_cameras)
            
        if db.query(Vehicle).count() == 0:
            initial_vehicles = [
                Vehicle(vehicle_type="car", count=2, timestamp=timestamp, camera_id=initial_cameras[0].id),
                Vehicle(vehicle_type="motorcycle", count=2, timestamp=timestamp, camera_id=initial_cameras[1].id),
                Vehicle(vehicle_type="motorcycle", count=2, timestamp=timestamp, camera_id=initial_cameras[2].id),
            ]
            db.add_all(initial_vehicles)
            
        if db.query(TrafficDensity).count() == 0:
            initial_traffic_density = [
                TrafficDensity(timestamp=timestamp, density_level=0.1, vehicle_count=2, camera_id=initial_cameras[0].id),
                TrafficDensity(timestamp=timestamp, density_level=0.1, vehicle_count=2, camera_id=initial_cameras[1].id),
                TrafficDensity(timestamp=timestamp, density_level=0.1, vehicle_count=2, camera_id=initial_cameras[2].id),
            ]
            db.add_all(initial_traffic_density)
        
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()    
    
