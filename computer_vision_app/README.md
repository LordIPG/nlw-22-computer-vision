# 🤖 GestureFlow — Reconhecimento de Gestos com IA em Tempo Real

> Projeto desenvolvido durante meus estudos em Visão Computacional (NLW - Rocketseat)

---

## 📌 Sobre o Projeto

Este projeto foi desenvolvido como parte do meu aprendizado em **Visão Computacional e Inteligência Artificial**, com o objetivo de construir uma aplicação real capaz de reconhecer gestos das mãos em tempo real utilizando a webcam.

Durante o desenvolvimento, saí da teoria e implementei um sistema completo, entendendo na prática como funciona um pipeline de visão computacional — desde a captura da imagem até a predição feita por um modelo de Machine Learning.

O sistema, chamado **GestureFlow**, utiliza tecnologias modernas para processar imagens em tempo real e retornar os resultados instantaneamente na interface web.

---

## 🎯 Objetivo

Desenvolver um sistema capaz de:

- Reconhecer gestos das mãos em tempo real  
- Processar imagens com baixa latência  
- Utilizar modelos de Machine Learning treinados  
- Integrar backend e frontend em tempo real  

---

## 🛠️ Tecnologias Utilizadas

- Python  
- FastHTML  
- OpenCV  
- MediaPipe  
- Scikit-Learn  
- WebSockets  
- uv (gerenciamento de dependências)  

---

## 🚀 Funcionalidades do Projeto

- Reconhecimento de gestos em tempo real  
- Comunicação em tempo real via WebSockets  
- Monitoramento de FPS (performance)  
- Controle de qualidade da imagem  
- Visualização dos landmarks da mão  
- Feedback visual instantâneo no navegador  

---

## 📁 Estrutura do Projeto

computer_vision_app/  
├── app.py  
├── core/  
│   ├── processor.py  
│   ├── models.py  
│   └── utils.py  
├── models/  
│   ├── gesture_model.joblib  
│   └── label_encoder.joblib  
├── assets/  
│   ├── script.js  
│   ├── style.css  
│   └── images/  
├── pyproject.toml  
└── README.md  

---

## ⚙️ Como Executar

### 🧩 Pré-requisitos

- Python 3.11+  
- uv (recomendado)  

👉 https://astral.sh/uv  

---

### 📦 Instalação

cd computer_vision_app  
uv sync  

---

### ▶️ Executando a Aplicação

uv run app.py  

Abra no navegador:

http://localhost:5001  

---

## 📥 Download dos Modelos

Para o sistema funcionar corretamente, é necessário adicionar o modelo do MediaPipe:

- `gesture_recognizer.task`

Coloque o arquivo em:

computer_vision_app/models/  

---

## ⚙️ Como Funciona

O sistema funciona em um fluxo contínuo:

1. O navegador captura imagens da webcam  
2. Os frames são comprimidos e enviados via WebSocket  
3. O backend processa as imagens com MediaPipe  
4. O modelo de Machine Learning classifica o gesto  
5. O resultado é enviado de volta para o frontend  
6. A interface atualiza em tempo real  

---

## 📚 Aprendizados

Durante esse projeto eu consegui aprender e consolidar diversos conceitos importantes:

- Detecção de mãos e landmarks com MediaPipe  
- Treinamento e uso de modelos de Machine Learning  
- Comunicação em tempo real com WebSockets  
- Processamento de imagens com OpenCV  
- Integração entre frontend e backend  
- Construção de aplicações interativas em tempo real  

---

## 🧾 Considerações Finais

Este projeto foi um dos mais importantes do meu aprendizado até agora, pois me permitiu desenvolver uma aplicação completa, unindo visão computacional, backend e frontend.

Mais do que fazer funcionar, o foco foi entender cada etapa do processo e como essas tecnologias se conectam em um sistema real.

---