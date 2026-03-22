import cv2
import mediapipe as mp
import pickle
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections as mp_hands
import os

def preprocess_single_hand(landmarks):
    """
    Subtrai o pulso (x0, y0, z0) de um conjunto de 21 landmarks.
    """
    wrist_x, wrist_y, wrist_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    flat_data = []
    for lm in landmarks:
        flat_data.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
    return np.array(flat_data).reshape(1, -1)

def main():
    model_file = 'gesture_model.pkl'
    encoder_file = 'label_encoder.pkl'

    if not os.path.exists(model_file) or not os.path.exists(encoder_file):
        print("Erro: Modelo ou Encoder não encontrados. Execute 'train_gesture_model.py' primeiro.")
        return

    # Carregar modelo treinado
    with open(model_file, 'rb') as f: model = pickle.load(f)
    with open(encoder_file, 'rb') as f: le = pickle.load(f)

    # Configurar MediaPipe para detecção de marcas
    model_path = "gesture_recognizer.task" # Usamos o mesmo detector
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2
    )

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        cap = cv2.VideoCapture(0)
        print("Testando Modelo Personalizado. Pressione 'Q' para sair.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            result = recognizer.recognize_for_video(mp_image, ms)
            
            annotated_image = frame.copy()

            if result.hand_landmarks:
                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    # 1. Desenha esqueleto
                    mp_drawing.draw_landmarks(
                        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # 2. PREDIZER usando o modelo treinado
                    try:
                        processed = preprocess_single_hand(hand_landmarks)
                        pred_idx = model.predict(processed)[0]
                        pred_label = le.inverse_transform([pred_idx])[0]
                        conf = np.max(model.predict_proba(processed))
                        
                        # Exibe predição personalizada
                        x_min = int(min([lm.x for lm in hand_landmarks]) * frame.shape[1])
                        y_min = int(min([lm.y for lm in hand_landmarks]) * frame.shape[0])
                        cv2.putText(annotated_image, f"IA: {pred_label} ({conf:.2f})", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except Exception as e:
                        pass

            cv2.imshow('Custom Gesture Inference', annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
