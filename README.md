# 🧠 LeNet-5 — Classificação de Dígitos MNIST

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-green)
![Status](https://img.shields.io/badge/Status-Learning-yellow)
![Field](https://img.shields.io/badge/Field-Computer%20Vision-purple)

Este projeto foi desenvolvido como parte do meu processo de aprendizado em **Visão Computacional e Deep Learning**.

Aqui implementei a arquitetura clássica **LeNet-5 (Yann LeCun, 1998)** utilizando **PyTorch**, com o objetivo de entender na prática como **Redes Neurais Convolucionais (CNN)** funcionam para classificação de imagens.

O modelo foi treinado para reconhecer dígitos manuscritos utilizando o dataset **MNIST**.

---

# 📚 Contexto de Aprendizado

Este projeto foi desenvolvido durante a trilha de **Computer Vision da NLW (Next Level Week)** da Rocketseat.

Durante o desenvolvimento explorei na prática conceitos como:

- funcionamento de **Redes Neurais Convolucionais**
- processamento de imagens com **PyTorch**
- treinamento de modelos de Deep Learning
- análise de desempenho do modelo

O objetivo principal foi **compreender o funcionamento interno de uma CNN clássica**.

---

# 🛠️ Arquitetura LeNet-5

A implementação segue a arquitetura clássica **LeNet-5**, com pequenas adaptações modernas comuns em implementações atuais.

### Camadas Convolucionais

- Conv1 → 6 filtros (5×5)
- Conv2 → 16 filtros (5×5)

### Função de Ativação

- ReLU (substituindo Sigmoid)

### Pooling

- Max Pooling (2×2)

### Camadas Fully Connected

- FC1 → 120 neurônios
- FC2 → 84 neurônios
- Output → 10 classes

Fluxo da rede:

```text
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
```

---

# 📊 Dataset

O dataset utilizado foi o **MNIST**, um dos conjuntos de dados mais tradicionais para aprendizado de visão computacional.

Características:

- 60.000 imagens de treino
- 10.000 imagens de teste
- imagens em escala de cinza
- resolução de **28×28 pixels**

Classes:

```text
0 1 2 3 4 5 6 7 8 9
```

---

# 🧪 Resultados

Após o treinamento por algumas épocas, o modelo alcançou aproximadamente:

**Accuracy ≈ 98%**

Esse resultado demonstra como mesmo arquiteturas clássicas como a **LeNet-5** ainda são bastante eficazes para esse tipo de problema.

---

# 🔍 O que explorei neste projeto

Durante o desenvolvimento procurei analisar o comportamento da rede neural, incluindo:

- visualização dos **filtros da primeira camada**
- análise de **feature maps**
- comparação entre **predições corretas e incorretas**

Essas análises ajudam a entender **como a rede aprende padrões nas imagens**.

---

# ⚙️ Como executar

## Instalar dependências

O projeto utiliza o gerenciador de dependências **uv**.

```bash
uv sync
```

ou manualmente:

```bash
pip install torch torchvision matplotlib
```

---

## Executar o notebook

Abra o arquivo:

```text
LeNet5.ipynb
```

Execute as células sequencialmente para:

1. carregar o dataset MNIST  
2. visualizar exemplos de imagens  
3. treinar a rede neural  
4. avaliar o modelo  

---

# 📁 Estrutura do Projeto

```text
lenet/
│
├── data/                # Dataset MNIST (baixado automaticamente)
├── LeNet5.ipynb         # Notebook principal
├── main.py              # Script auxiliar
├── pyproject.toml       # Dependências do projeto
├── uv.lock              # Lock de dependências
├── README.md
└── .gitignore
```

---

# 🎯 Objetivo do Projeto

Este projeto foi desenvolvido com o objetivo de aprender e explorar conceitos fundamentais de:

- Redes Neurais Convolucionais (CNN)
- Classificação de imagens
- Treinamento de modelos de Deep Learning
- Visualização de representações internas da rede

---

# 🚀 Próximos estudos

Este projeto faz parte do meu processo de evolução em **Machine Learning e Computer Vision**.

Próximos tópicos que pretendo explorar:

- classificação de imagens mais complexas (CIFAR-10)
- visualização avançada de feature maps
- detecção de objetos
- arquiteturas modernas de CNN

---

# 👨‍💻 Autor

Projeto desenvolvido como parte do meu aprendizado em **Computer Vision com PyTorch**.
