import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections as mp_hands

import requests
import os
import numpy as np
import csv
import argparse

def save_to_csv(csv_path, target_label, handedness, landmarks):
    """
    Salva uma linha no CSV com o rótulo, lateralidade e as 63 coordenadas (21x xyz).
    """
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Cria cabeçalho se o arquivo for novo
        if not file_exists:
            header = ['target', 'handedness']
            for i in range(21):
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            writer.writerow(header)
        
        # Prepara a linha de dados
        row = [target_label, handedness]
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])
        
        writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Gravador de landmarks para dataset de gestos.')
    parser.add_argument('--target', type=str, default='desconhecido', help='Rótulo do gesto sendo gravado.')
    parser.add_argument('--file', type=str, default='gestures_dataset.csv', help='Nome do arquivo CSV.')
    args = parser.parse_args()

    model_path = "gesture_recognizer.task"
    if not os.path.exists(model_path):
        model_url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        with open(model_path, "wb") as f: f.write(requests.get(model_url).content)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1 # Gravamos uma mão por vez para simplificar o dataset
    )

    try:
        with vision.GestureRecognizer.create_from_options(options) as recognizer:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened(): return
            
            print(f"Gravador Iniciado. Alvo: '{args.target}'")
            print("Comandos: [BARRA DE ESPAÇO] ou [S] para Gravar | [Q] para Sair")
            
            last_saved_frame = 0 # Contador visual de frames salvos

            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                recognition_result = recognizer.recognize_for_video(mp_image, ms)
                
                annotated_image = frame.copy()
                h, w, _ = annotated_image.shape

                landmarks_to_save = None
                hand_label = ""

                if recognition_result.hand_landmarks:
                    # Como num_hands=1, pegamos a primeira detectada
                    landmarks_to_save = recognition_result.hand_landmarks[0]
                    hand_label = recognition_result.handedness[0][0].category_name
                    
                    # Desenha landmarks
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        landmarks_to_save,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
                # Feedback visual de gravação
                overlay_text = f"Alvo: {args.target} | Salvos: {last_saved_frame}"
                cv2.putText(annotated_image, overlay_text, (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Gravador de Gestos', annotated_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif (key == ord(' ') or key == ord('s')) and landmarks_to_save:
                    save_to_csv(args.file, args.target, hand_label, landmarks_to_save)
                    last_saved_frame += 1
                    print(f"Frame #{last_saved_frame} salvo com label '{args.target}'")
                    # Pisca a tela em verde para feedback
                    cv2.rectangle(annotated_image, (0,0), (w,h), (0, 255, 0), 20)
                    cv2.imshow('Gravador de Gestos', annotated_image)
                    cv2.waitKey(50)

            cap.release()
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
