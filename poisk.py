import cv2
from ultralytics import YOLO

# Загрузка модели с наилучшими весами, сохраненными после тренировки
model = YOLO('runs/detect/train3/weights/best.pt')

# Открываем видео файл
video_path = r'C:\Users\ovsya\Downloads\video2d.mp4'  # Использование raw строки

cap = cv2.VideoCapture(video_path)

# Определение желаемого размера окна
window_width = 1024
window_height = 768

# Идентификатор класса для дронов
drone_class_id = 0  # Измените на ваш идентификатор класса дронов, если это не 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Обнаружение объектов на кадре
    results = model(frame)
    
    # Обработка результатов и отрисовка рамок на кадре
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())  # Преобразование тензора в numpy массив и затем в список
            class_id = int(box.cls.cpu().numpy().flatten()[0])  # Получение идентификатора класса
            confidence = box.conf.cpu().numpy().flatten()[0]  # Получение доверительного значения
            
            # Определение метки в зависимости от класса
            if class_id == drone_class_id:
                label = f'DRON {confidence:.2f}'
            else:
                label = f'Класс {class_id} {confidence:.2f}'
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Масштабирование кадра до заданного размера
    frame = cv2.resize(frame, (window_width, window_height))
    
    # Показываем кадр с рамками
    cv2.imshow('Drone Detection', frame)
    
    # Нажмите 'q' для выхода из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
