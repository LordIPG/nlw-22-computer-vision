import cv2
import mediapipe as mp
import os
import numpy as np
import csv
import argparse
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections as mp_hands

# --- CONFIGURAÇÃO ---
CSV_FILE = 'gestures_dataset.csv'

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
    parser.add_argument('--file', type=str, default=CSV_FILE, help='Nome do arquivo CSV.')
    args = parser.parse_args()

    # MediaPipe Setup
    model_path = "gesture_recognizer.task"
    if not os.path.exists(model_path):
        print("Baixando modelo do MediaPipe...")
        model_url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        try:
            response = requests.get(model_url, timeout=10)
            with open(model_path, "wb") as f: 
                f.write(response.content)
            print("Download concluído.")
        except Exception as e:
            print(f"Erro ao baixar modelo: {e}")
            return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=2
    )

    try:
        with vision.GestureRecognizer.create_from_options(options) as recognizer:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Erro: Não foi possível abrir a webcam.")
                return

            print("Webcam pronta.")
            print(f"Alvo Atual: '{args.target}' | Arquivo: '{args.file}'")
            print("-" * 30)
            print("COMANDOS:")
            print("[S] - Salvar frame único")
            print("[C] - Alternar gravação CONTÍNUA")
            print("[Q] - Sair")
            print("-" * 30)
            
            recording_continuous = False
            saved_count = 0
            current_target = args.target

            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Timestamp para o modo VIDEO em milissegundos
                ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                result = recognizer.recognize_for_video(mp_image, ms)
                
                annotated_image = frame.copy()
                h, w, _ = annotated_image.shape
                
                # Processamento de detecções
                if result.hand_landmarks:
                    for idx, hand_landmarks in enumerate(result.hand_landmarks):
                        # Desenha landmarks
                        mp_drawing.draw_landmarks(
                            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        hand_label = result.handedness[idx][0].category_name
                        
                        # Gravação contínua se ativa
                        if recording_continuous:
                            save_to_csv(args.file, current_target, hand_label, hand_landmarks)
                            saved_count += 1
                
                # Interface Visual (Overlay)
                status_color = (0, 0, 255) if recording_continuous else (255, 255, 0)
                status_text = "GRAVANDO..." if recording_continuous else "AGUARDANDO"
                
                # Desenha barra de status no topo
                cv2.rectangle(annotated_image, (0, 0), (w, 110), (50, 50, 50), -1)
                cv2.putText(annotated_image, f"ALVO: {current_target}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_image, f"STATUS: {status_text}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
                cv2.putText(annotated_image, f"SALVOS: {saved_count}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Gravador de Landmarks - Continuo', annotated_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if result.hand_landmarks:
                        for idx, hand_landmarks in enumerate(result.hand_landmarks):
                            h_label = result.handedness[idx][0].category_name
                            save_to_csv(args.file, current_target, h_label, hand_landmarks)
                            saved_count += 1
                        print(f"Frame salvo! Total: {saved_count}")
                        # Feedback visual: flash verde
                        flash = annotated_image.copy()
                        cv2.rectangle(flash, (0,0), (w,h), (0, 255, 0), 10)
                        cv2.imshow('Gravador de Landmarks - Continuo', flash)
                        cv2.waitKey(50)
                elif key == ord('c'):
                    recording_continuous = not recording_continuous
                    state = "INICIADA" if recording_continuous else "PARADA"
                    print(f"Gravação contínua {state}")

            cap.release()
            cv2.destroyAllWindows()
            print(f"\nGravação finalizada. Total de frames salvos: {saved_count}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()
