from ultralytics import YOLO

# Load a model
model = YOLO("yolov8-tdrs-for-convert.pt")  # load a custom trained model

# Export the model
model.export(format="engine", 
             int8 = True, 
             simplify = True, 
             dynamic = True, 
             workspace = 2)