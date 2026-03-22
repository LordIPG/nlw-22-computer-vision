# 🌌 Gemini Vision with Python

Este projeto demonstra como usar a **API do Gemini (Google AI)** para analisar imagens diretamente de um Jupyter Notebook.

## 🚀 Como Executar

1.  **Chave de API**: Obtenha sua chave em [Google AI Studio](https://aistudio.google.com/).
2.  **Configuração**: Abra o arquivo `.env` e substitua `YOUR_API_KEY_HERE` pela sua chave real.
3.  **Ambiente**:
    *   Este projeto utiliza o `uv` para gerenciamento de pacotes.
    *   Para rodar o notebook, você pode usar:
        ```bash
        uv run jupyter notebook
        ```
    *   Ou simplesmente abra o arquivo `gemini_vision.ipynb` em seu VS Code ou editor favorito que suporte notebooks e selecione o kernel do ambiente `uv` (geralmente em `.venv`).

## 🖼️ Imagens
O notebook está configurado para buscar imagens na pasta `C:\Users\ivanu\Desktop\recog_system\imagens`.

## 📦 Dependências
- `google-generativeai`
- `python-dotenv`
- `pillow`
- `matplotlib`
- `ipykernel`
