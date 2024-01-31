from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
mask_trained  = model.train(data=r'C:\Users\singh\Desktop\Project\config.yaml',epochs=100)

