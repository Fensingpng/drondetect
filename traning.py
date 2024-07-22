from ultralytics import YOLO

# Загрузите предобученную модель YOLOv8
model = YOLO("yolov8n.pt")

# Обучите модель на вашем датасете
model.train(data="C:/MyProject/traningyolo/dataset/data.yaml", epochs=50)
