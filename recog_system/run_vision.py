import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
import time

def run_vision_analysis():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERRO] GOOGLE_API_KEY nao encontrada.")
        return
    
    genai.configure(api_key=api_key)
    
    IMAGE_DIR = Path(r"C:\Users\ivanu\Desktop\recog_system\imagens")
    REPORT_FILE = Path("analysis_report.md")
    
    if not IMAGE_DIR.exists():
        print(f"[ERRO] Diretorio {IMAGE_DIR} nao encontrado.")
        return
        
    images = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    
    if not images:
        print("[ERRO] Nenhuma imagem encontrada.")
        return

    # Tentando gemini-flash-latest
    model = genai.GenerativeModel('gemini-flash-latest')
    
    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("# Relatório de Análise de Imagens (Gemini Vision)\n\n")
        
        for i, img_path in enumerate(images):
            print(f"[INFO] Analisando ({i+1}/{len(images)}): {img_path.name}")
            img = Image.open(img_path)
            
            success = False
            retries = 0
            while not success and retries < 2:
                try:
                    response = model.generate_content([
                        "Analise esta imagem e descreva seus elementos principais em portugues. O que esta acontecendo nela?", 
                        img
                    ])
                    
                    report.write(f"## Imagem: {img_path.name}\n\n")
                    report.write(f"{response.text}\n\n")
                    report.write("---\n\n")
                    print(f"[OK] {img_path.name} processada.")
                    success = True
                    
                except Exception as e:
                    if "429" in str(e):
                        print(f"[AVISO] Rate limit atingido. Aguardando 60 segundos... (Tentativa {retries + 1})")
                        time.sleep(60)
                        retries += 1
                    else:
                        print(f"[ERRO] Falha ao processar {img_path.name}: {e}")
                        report.write(f"## Imagem: {img_path.name}\n\n[ERRO] {e}\n\n---\n\n")
                        break
            
            # Pequeno delay entre imagens bem-sucedidas para evitar bater no limite de novo rapidamente
            if success and i < len(images) - 1:
                print("[INFO] Aguardando 30 segundos antes da próxima imagem...")
                time.sleep(30)

    print(f"\n[INFO] Analise concluida! Resultados salvos em: {REPORT_FILE.absolute()}")

if __name__ == "__main__":
    # Pequena espera para limpar possiveis rate limits temporarios
    time.sleep(2)
    run_vision_analysis()
