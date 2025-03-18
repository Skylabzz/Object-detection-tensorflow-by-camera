import threading
import time
import cv2
import numpy as np
from datetime import datetime
import requests
import tensorflow as tf

model = tf.keras.models.load_model(r"./app\best_model_high_accuracy.keras")

labels = ["Fall Down", "Lying Down", "Sit Down", "Sitting", "Stand up", "Standing", "Walking"]

def send_message_to_line(line_token, message):
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_token}"}
    payload = {'message': message}
    response = requests.post(url, headers=headers, data=payload)
    return response.status_code

def send_image_to_line(frame, line_token, message):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_token}"}
    payload = {'message': message}
    files = {'imageFile': ('image.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(url, headers=headers, data=payload, files=files)
    return response.status_code

def predict_with_model(frame_stack, camera_name):
    frame_array = np.array(frame_stack)  
    frame_array = np.expand_dims(frame_array, axis=0)
    prediction = model.predict(frame_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]  
    return labels[predicted_class], confidence

from collections import deque
def start_rtsp_stream(rtsp_url, camera_name, line_token, camera_message, room_name, is_notification, camera_flags, camera_threads):
    camera_flags[camera_name] = False
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_stack = deque(maxlen=30)

    sent_initial_notification = False

    while True:
        if camera_flags[camera_name]:
            break

        ret, frame = cap.read()
        if not ret:
            send_message_to_line(line_token, f"❌ ข้อผิดพลาด: กล้อง {camera_name} ไม่สามารถส่งเฟรมได้ กรุณาตรวจสอบการเชื่อมต่อกล้อง.")
            break

        if not sent_initial_notification:
            success_message = f"✅ กล้อง {camera_name} (ห้อง {room_name}) เชื่อมต่อสำเร็จ"
            send_message_to_line(line_token, success_message)
            sent_initial_notification = True
    
        frame_resized = cv2.resize(frame, (224, 224))
        frame_stack.append(frame_resized)

        if len(frame_stack) == 30:
            frame_array = np.array(frame_stack)
            frame_array = np.expand_dims(frame_array, axis=0)
            prediction = model.predict(frame_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][predicted_class]

            if predicted_class == labels.index("Fall Down") and confidence >= 0.50:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                full_message = f"""
                *📅 เวลา*: {timestamp}
                *📷 กล้อง*: {camera_name}
                *🏠 ห้อง*: {room_name}
                {camera_message}
                """

                if is_notification:
                    frame_noti = cv2.resize(frame, (360, 240))
                    send_image_to_line(frame_noti, line_token, full_message)
                else:
                    send_message_to_line(line_token, full_message)

            frames_to_keep = int(fps)
            frame_stack = deque(list(frame_stack)[-frames_to_keep:], maxlen=30)

        time.sleep(1 / fps)
    
    cap.release()