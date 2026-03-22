# 🧠 LeNet-5 — Classificação de Dígitos MNIST

> Projeto desenvolvido durante meus estudos em Visão Computacional (NLW - Rocketseat)

---

## 📌 Sobre o Projeto

Este projeto foi desenvolvido como parte do meu aprendizado em **Deep Learning e Visão Computacional**, onde implementei do zero a arquitetura clássica **LeNet-5** utilizando **PyTorch**.

A proposta foi entender na prática como uma Rede Neural Convolucional funciona, desde o processamento dos dados até a avaliação final do modelo.

Durante o desenvolvimento, apliquei melhorias modernas na arquitetura original para melhorar desempenho e entendimento.

---

## 🎯 Objetivo

Classificar dígitos manuscritos (0–9) utilizando o dataset **MNIST**.

---

## 🧠 Arquitetura Utilizada

A arquitetura segue a base da LeNet-5 com algumas adaptações modernas:

- Camadas Convolucionais  
- Função de ativação **ReLU** (substituindo Sigmoid)  
- **Max Pooling (2x2)**  
- Camadas totalmente conectadas (Fully Connected)  

### 🔽 Fluxo da rede:

Input (1x28x28)  
↓  
Conv1 → ReLU → MaxPool  
↓  
Conv2 → ReLU → MaxPool  
↓  
Flatten  
↓  
FC1 → ReLU  
↓  
FC2 → ReLU  
↓  
FC3 → Output (10 classes)

---

## 📊 O que foi explorado no projeto

Durante o desenvolvimento eu consegui visualizar e entender melhor o comportamento da rede:

- Visualização dos filtros da primeira camada  
- Feature Maps (mapas de ativação)  
- Análise de erros do modelo  
- Entendimento de como a rede identifica padrões (bordas, formas, etc.)  

---

## 🏆 Resultados Obtidos

Após o treinamento:

- **Acurácia final no teste: 98.91%**  
- Modelo salvo com sucesso (`lenet5_mnist.pth`)  

### 🔍 Teste real:

Predição do modelo carregado: 7  
Rótulo real: 7  

---

## 📁 Estrutura do Projeto

lenet/  
├── data/                # Dataset MNIST (download automático)  
├── weights/             # Pesos do modelo (.pth)  
├── lenet5.ipynb         # Notebook principal  
├── pyproject.toml       # Dependências (uv)  
└── README.md  

---

## ⚙️ Como Executar

### 1️⃣ Pré-requisitos

Este projeto utiliza o gerenciador de pacotes **uv**:

👉 https://astral.sh/uv

---

### 2️⃣ Instalação

cd lenet  
uv sync  

---

### 3️⃣ Executando o projeto

Abra o arquivo:

lenet5.ipynb  

E execute as células na ordem.

---

## 🔍 O que o notebook faz

- Carrega e visualiza o dataset MNIST  
- Define a arquitetura da rede  
- Treina o modelo  
- Avalia a acurácia  
- Salva os pesos do modelo  
- Permite testar previsões  

---

## 🚀 Aprendizados

Durante esse projeto eu aprendi:

- Como funciona uma CNN na prática  
- Como estruturar um pipeline de Deep Learning  
- Como treinar e avaliar modelos  
- Como interpretar resultados e erros  
- Como salvar e reutilizar modelos treinados  

---

## 💡 Considerações

Esse foi meu primeiro contato mais profundo com redes neurais convolucionais, e foi essencial para entender como modelos de visão computacional realmente funcionam por trás dos panos.

---

## 👨‍💻 Autor

Desenvolvido por mim durante meus estudos em Inteligência Artificial e Visão Computacional 🚀