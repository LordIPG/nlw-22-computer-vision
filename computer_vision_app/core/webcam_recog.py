import cv2
from gesture_processor import get_processor

def main():
    """
    Script principal para reconhecimento de gestos via webcam (Modularizado).
    """
    try:
        # Inicializa o processador (gerencia MediaPipe e IA personalizada)
        processor = get_processor()
    except Exception as e:
        print(f"Erro ao inicializar o processador: {e}")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nIniciando reconhecimento modular... Pressione 'q' para sair.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Inverte para efeito espelho
            frame = cv2.flip(frame, 1)
            
            # --- FUNÇÃO SOLICITADA ---
            # Recebe uma imagem (BGR) e retorna uma imagem processada (BGR)
            output_frame = processor.process_image(frame)
            
            cv2.imshow('Custom Gesture Recognition (Modular)', output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        processor.close()

if __name__ == "__main__":
    main()
