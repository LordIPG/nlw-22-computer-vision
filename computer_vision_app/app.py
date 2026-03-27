import json
import os
import time
from fasthtml.common import *
from core.processor import GestureProcessor
from core.utils import decode_image, encode_image

app, rt = fast_app(pico=False, hdrs=(
    Link(rel='stylesheet', href='/assets/style.css?v=5'),
    Script(src='/assets/script.js'),
))
processor = GestureProcessor()

# To track FPS per connection, we'd normally use a session or context, 
# but for simplicity in this single-user-focused app, let's use a class to track it.
class FPSTracker:
    def __init__(self):
        self.prev_time = time.time()
    def update(self):
        curr = time.time()
        fps = 1 / (curr - self.prev_time) if curr > self.prev_time else 0
        self.prev_time = curr
        return int(fps)

fps_tracker = FPSTracker()

@rt("/")
def get():
    return Title("GestureFlow"), Main(
        Header(
            H1("GestureFlow"),
            P("Real-Time Hand Recognition", cls="subtitle")
        ),
        Div(
            Div(
                Video(id="video", autoplay=True, playsinline=True, style="display:none"),
                Canvas(id="canvas"),
                Div("FPS: 0", id="fps-counter", cls="fps-badge"),
                cls="vision-card"
            ),
            Div(
                Div(
                    H3("Image Quality Control"),
                    Div(
                        Input(type="range", id="quality-slider", min="0.1", max="1.0", step="0.05", value="0.6"),
                        Span("60%", id="quality-value"),
                        cls="quality-control"
                    ),
                    H3("Settings"),
                    Div(
                        Label(Input(type="checkbox", id="draw-landmarks-cb", checked=True), " Draw Hand Landmarks"),
                        cls="settings-control"
                    ),
                    H3("Live Feed Data"),
                    Div(id="gesture-container"),
                    cls="info-card"
                ),
                Div(
                    H3("Detected Gesture"),
                    Div(Img(id="gesture-image"), cls="gesture-preview-box"),
                    cls="info-card"
                ),
                cls="sidebar-info"
            ),
            cls="main-content"
        ),
        Footer(
            Button(
                NotStr('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'),
                onclick="document.getElementById('info_modal').showModal()", 
                cls="info-btn",
                title="Informações do Projeto"
            ),
            Div(
                A("Ivan Galdino", href="https://github.com/LordIPG", target="_blank", cls="footer-link"),
                Span("|", cls="footer-separator"),
                A("Rocketseat", href="https://app.rocketseat.com.br/", target="_blank", cls="footer-link"),
                cls="footer-links"
            ),
            cls="app-footer"
        ),
        Dialog(
            Div(
                H3("Sobre o Projeto GestureFlow"),
                P("Este projeto foi desenvolvido como parte do meu aprendizado em ", B("Visão Computacional e Inteligência Artificial"), ", durante a trilha da NLW (Rocketseat)."),
                P("A ideia principal foi sair da teoria e construir uma aplicação real, capaz de ", B("reconhecer gestos das mãos em tempo real"), ", utilizando a webcam."),
                P("Durante o desenvolvimento, eu consegui entender na prática como funciona um sistema completo de visão computacional — desde a captura da imagem até a predição feita por um modelo de Machine Learning."),
                P("O sistema que desenvolvi, chamado ", B("Rockit Vision"), ", realiza:"),
                Ul(
                    Li("Captura de vídeo em tempo real pelo navegador"),
                    Li("Envio dos frames para o backend via WebSocket"),
                    Li("Processamento das imagens com MediaPipe"),
                    Li("Classificação dos gestos utilizando um modelo treinado por mim"),
                    Li("Retorno do resultado instantaneamente para a interface"),
                    style="margin-left: 2rem; color: #cbd5e1; list-style-type: disc; margin-bottom: 0.5rem;"
                ),
                P("Além disso, implementei controles interativos para ajustar a qualidade da imagem e visualizar os pontos (landmarks) das mãos, o que me ajudou muito a entender como o modelo interpreta os dados."),
                P("Esse projeto foi muito importante para consolidar conceitos como:"),
                Ul(
                    Li("Processamento de imagem"),
                    Li("Detecção de padrões"),
                    Li("Comunicação em tempo real (WebSockets)"),
                    Li("Integração entre frontend e backend"),
                    Li("Uso de modelos de Machine Learning em aplicações reais"),
                    style="margin-left: 2rem; color: #cbd5e1; list-style-type: disc; margin-bottom: 0.5rem;"
                ),
                P("Mais do que apenas funcionar, o foco foi entender ", B("como tudo acontece por trás dos bastidores"), "."),
                P("Hoje consigo enxergar claramente como a visão computacional pode ser aplicada em sistemas reais, como interfaces por gestos, automação e interação humano-computador."),
                Button("Fechar", onclick="document.getElementById('info_modal').close()", cls="close-btn"),
                cls="modal-content",
                style="text-align: left;"
            ),
            id="info_modal"
        ),
        cls="app-container"
    )

@app.ws("/ws")
async def ws(image: str, draw_landmarks: bool, send):
    img = decode_image(image)
    if img is not None:
        processed_img, labels, gesture_image = processor.process_frame(img, draw_landmarks)
        fps = fps_tracker.update()
        await send(json.dumps({
            "image": encode_image(processed_img),
            "labels": labels,
            "gesture_image": gesture_image,
            "fps": fps
        }))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}...")
    try:
        serve(app, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"CRITICAL ERROR during server startup: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
