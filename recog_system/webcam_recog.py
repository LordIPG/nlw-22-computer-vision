import cv2
import mediapipe as mp
import pickle
import os
import numpy as np
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections as mp_hands

# --- CONFIGURAÇÃO DA IA PERSONALIZADA ---
MODEL_FILE = 'gesture_model.pkl'
ENCODER_FILE = 'label_encoder.pkl'

def load_custom_model():
    """Carrega o modelo e o encoder se existirem."""
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f: model = pickle.load(f)
            with open(ENCODER_FILE, 'rb') as f: le = pickle.load(f)
            return model, le
        except Exception as e:
            print(f"Erro ao carregar modelo personalizado: {e}")
    return None, None

def predict_custom_gesture(model, le, hand_landmarks):
    """Realiza a predição usando o modelo .pkl personalizado."""
    # Preprocessamento (Invariância de transição - relativo ao pulso)
    wrist_x, wrist_y, wrist_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
    flat_data = []
    for lm in hand_landmarks:
        flat_data.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
    
    processed = np.array(flat_data).reshape(1, -1)
    
    # Predição
    pred_idx = model.predict(processed)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    
    # Tentativa de obter confiança se o modelo suportar predict_proba
    try:
        conf = np.max(model.predict_proba(processed))
    except:
        conf = 1.0 # Fallback se não tiver proba
    
    return pred_label, conf

# --- LÓGICA DE DESENHO E WEBCAM ---

def draw_landmarks_and_gestures(bgr_image, detection_result, custom_ai=None):
    if not detection_result or not detection_result.hand_landmarks:
        return bgr_image
    
    annotated_image = bgr_image.copy()
    h, w, _ = annotated_image.shape
    model, le = custom_ai if custom_ai else (None, None)

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        # 1. Desenho Oficial das Marcas (Landmarks)
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # 2. Informações de Legenda
        # Lateralidade (Esquerda/Direita)
        hand_label = detection_result.handedness[idx][0].category_name if idx < len(detection_result.handedness) else "?"
        
        # Reconhecimento Nativo do MediaPipe
        native_gesture = detection_result.gestures[idx][0].category_name if idx < len(detection_result.gestures) and detection_result.gestures[idx] else "Nenhum"
        
        # 3. Predição da IA PERSONALIZADA (se disponível)
        custom_text = ""
        if model and le:
            try:
                c_label, c_conf = predict_custom_gesture(model, le, hand_landmarks)
                custom_text = f" | IA: {c_label} ({c_conf:.2f})"
            except: 
                pass

        # Exibição do Texto na tela
        # Encontra o ponto mais alto da mão para colocar o texto
        x_min = int(min([lm.x for lm in hand_landmarks]) * w)
        y_min = int(min([lm.y for lm in hand_landmarks]) * h)
        
        display_text = f"{hand_label}: {native_gesture}{custom_text}"
        
        # Sombra do texto para legibilidade
        cv2.putText(annotated_image, display_text, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        # Texto principal (Verde se for IA personalizada, senão branco/verde)
        cv2.putText(annotated_image, display_text, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    return annotated_image

def main():
    # Inicializa modelos personalizados
    custom_ai = load_custom_model()
    if custom_ai[0]: 
        print(f"Sucesso: Modelo '{MODEL_FILE}' carregado!")
    else: 
        print("Aviso: Modelo IA Personalizado não encontrado. Usando apenas reconhecimento nativo.")

    # MediaPipe Setup (Gesture Recognizer)
    model_path = "gesture_recognizer.task"
    if not os.path.exists(model_path):
        print("Baixando modelo base do MediaPipe...")
        model_url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        try:
            with open(model_path, "wb") as f: 
                f.write(requests.get(model_url, timeout=10).content)
            print("Download concluído.")
        except Exception as e:
            print(f"Erro ao baixar modelo MediaPipe: {e}")
            return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.VIDEO, 
        num_hands=2
    )

    try:
        with vision.GestureRecognizer.create_from_options(options) as recognizer:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Erro: Não foi possível abrir a webcam.")
                return

            print("Webcam pronta. Pressione 'Q' para sair.")
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                
                # Inverte a imagem para efeito espelho
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Converte para imagem do MediaPipe
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Timestamp necessário para o modo VIDEO
                ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                
                # Executa o reconhecimento (Landmarks + Gestos Nativos)
                result = recognizer.recognize_for_video(mp_image, ms)
                
                # Processa e desenha resultados (incluindo IA personalizada)
                output_frame = draw_landmarks_and_gestures(frame, result, custom_ai)
                
                cv2.imshow('Gesture Recognition (Hybrid)', output_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            
    except Exception as e: 
        print(f"Erro na execução: {e}")

if __name__ == "__main__":
    main()
