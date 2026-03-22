# 🖐️ Sistema de Reconhecimento de Gestos e Visão Computacional

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00C7B7?style=for-the-badge&logo=google&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 Sobre o Projeto

Este projeto foi desenvolvido como parte do meu aprendizado em Visão Computacional, explorando na prática diferentes áreas como classificação, detecção, segmentação e reconhecimento de gestos.

Durante o desenvolvimento, trabalhei com modelos pré-treinados e também com a criação de um modelo próprio de Machine Learning, construindo um pipeline completo que vai desde a coleta de dados até a inferência em tempo real.

O principal objetivo foi entender como aplicar visão computacional em cenários reais utilizando Python.

---

## 🎯 Objetivo

Construir um sistema capaz de:

- Reconhecer gestos em tempo real utilizando webcam  
- Treinar um modelo customizado com dados coletados manualmente  
- Explorar diferentes técnicas modernas de visão computacional  

---

## 🛠️ Tecnologias Utilizadas

- Python  
- OpenCV (captura de vídeo)  
- MediaPipe (detecção de mãos e landmarks)  
- Scikit-Learn (treinamento do modelo)  
- Pandas / NumPy (manipulação de dados)  
- Modelos pré-treinados (MobileNet, YOLO, CLIPSeg)  
- Google Gemini API (análise de imagens)  
- uv (gerenciamento de dependências)  

---

## 🔄 Pipeline de Reconhecimento de Gestos

O sistema principal foi dividido em três etapas:

### 📸 Coleta de Dados

Captura das coordenadas (x, y, z) dos 21 pontos da mão utilizando MediaPipe.

- Criação de dataset próprio  
- Registro manual de gestos  
- Salvamento em arquivo `.csv`  

---

### 🧠 Treinamento do Modelo

Treinamento de um modelo de Machine Learning utilizando Scikit-Learn:

- Algoritmo: Random Forest  
- Processamento dos dados coletados  
- Geração dos arquivos:
  - `gesture_model.pkl`  
  - `label_encoder.pkl`  

---

### 🎥 Reconhecimento em Tempo Real

Uso da webcam para:

- Detectar mãos  
- Extrair landmarks  
- Classificar gestos em tempo real  

---

## 🔬 Explorações em Visão Computacional

Além do sistema de gestos, também explorei outras áreas importantes:

- Classificação de imagens com MobileNet  
- Detecção de objetos com YOLO  
- Segmentação de imagens com CLIPSeg  
- Análise de imagens com a API do Google Gemini  

---

## 📁 Estrutura do Projeto

recog_system/  
├── record_hand_landmarks.py  
├── train_gesture_model.py  
├── webcam_recog.py  
├── mobilenet_classification.ipynb  
├── yolos_detection.ipynb  
├── segmentation_clipseg.ipynb  
├── gemini_vision.ipynb  
├── pyproject.toml  
└── README.md  

---

## ⚙️ Como Executar

### 🧩 Pré-requisitos

Este projeto utiliza o gerenciador de pacotes uv:

https://astral.sh/uv  

---

### 📦 Instalação

cd recog_system  
uv sync  

---

### ▶️ Execução

Coletar dados:

python record_hand_landmarks.py  

Treinar o modelo:

python train_gesture_model.py  

Executar reconhecimento em tempo real:

python webcam_recog.py  

---

## 🚀 Funcionalidades do Projeto

- Criação de dataset personalizado de gestos  
- Treinamento de modelo próprio  
- Reconhecimento em tempo real via webcam  
- Testes com modelos modernos de visão computacional  
- Integração com API de IA (Gemini)  

---

## 📚 Aprendizados

Durante esse projeto eu aprendi:

- Como funciona a detecção de mãos com MediaPipe  
- Como estruturar um pipeline de Machine Learning  
- Como coletar e preparar dados reais  
- Como treinar e avaliar modelos  
- Como aplicar visão computacional em tempo real  
- Como utilizar modelos pré-treinados  

---

## 🧾 Considerações Finais

Este projeto foi essencial para consolidar meu entendimento em visão computacional, permitindo aplicar conceitos teóricos em aplicações reais.

---

## 👨‍💻 Autor

Desenvolvido por mim durante meus estudos em Inteligência Artificial e Visão Computacional.