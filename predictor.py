import cv2
import numpy as np
import mediapipe as mp
from model import model
from utils import draw_axis, map_to_pi
import traceback

mp_face_mesh = mp.solutions.face_mesh

def preprocess(face, width=450, height=450):
    x_val = [lm.x * width for lm in face.landmark]
    y_val = [lm.y * height for lm in face.landmark]
    x_val = np.array(x_val) - np.mean(x_val[1])
    y_val = np.array(y_val) - np.mean(y_val[1])
    x_val = x_val / x_val.max() if x_val.max() != 0 else x_val
    y_val = y_val / y_val.max() if y_val.max() != 0 else y_val
    return np.concatenate([x_val, y_val])

def predict_pose(image_path: str):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        marks = preprocess(face, image.shape[1], image.shape[0])
        pred_angles = model.predict(marks.reshape(1, -1))[0]
        pred_pitch, pred_yaw, pred_roll = map_to_pi(pred_angles[0]), map_to_pi(pred_angles[1]), map_to_pi(pred_angles[2])

        center = face.landmark[1]
        tdx, tdy = int(center.x * image.shape[1]), int(center.y * image.shape[0])
        return draw_axis(image, pred_pitch, pred_yaw, pred_roll, tdx, tdy)
    finally:
        face_mesh.close()

def process_video(video_path: str, output_path: str, frame_skip=2):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not load video from {video_path}")
            return None

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"Error: Could not create output video at {output_path}")
            cap.release()
            return None

        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]
                    marks = preprocess(face, frame.shape[1], frame.shape[0])
                    pred_angles = model.predict(marks.reshape(1, -1))[0]
                    pred_pitch, pred_yaw, pred_roll = map_to_pi(pred_angles[0]), map_to_pi(pred_angles[1]), map_to_pi(pred_angles[2])

                    center = face.landmark[1]
                    tdx, tdy = int(center.x * frame.shape[1]), int(center.y * frame.shape[0])
                    frame = draw_axis(frame, pred_pitch, pred_yaw, pred_roll, tdx, tdy)

                out.write(frame)
        finally:
            face_mesh.close()

        cap.release()
        out.release()
        print(f"Video saved as {output_path}")
        return output_path
    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        traceback.print_exc()
        return None