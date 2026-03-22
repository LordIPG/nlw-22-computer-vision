import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_landmarks(df):
    """
    Subtrai as coordenadas do pulso (x0, y0, z0) de todos os marcos
    para tornar o modelo invariante à transição (posição na tela).
    """
    # Copia as colunas de landmarks
    landmark_cols = [c for c in df.columns if c not in ['target', 'handedness']]
    processed_data = df[landmark_cols].copy()
    
    # Para cada mão (linha), subtraímos o pulso (x0, y0, z0)
    for i in range(21):
        processed_data[f'x{i}'] = processed_data[f'x{i}'] - df['x0']
        processed_data[f'y{i}'] = processed_data[f'y{i}'] - df['y0']
        processed_data[f'z{i}'] = processed_data[f'z{i}'] - df['z0']
        
    return processed_data

def train_model(csv_path, model_output='gesture_model.pkl', encoder_output='label_encoder.pkl'):
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo {csv_path} não encontrado!")
        return

    # 1. Carregar dados
    df = pd.read_csv(csv_path)
    
    # Verifica se há classes suficientes
    classes = df['target'].unique()
    if len(classes) < 2:
        print(f"Erro: Você precisa de pelo menos 2 tipos de gestos para treinar. Encontrado: {classes}")
        return

    print(f"Carregando {len(df)} amostras de {len(classes)} gestos: {classes}")

    # 2. Pré-processamento
    X = preprocess_landmarks(df)
    y = df['target']

    # Codificar labels (texto para números)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3. Divisão Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 4. Treinar Modelo (Random Forest)
    print("Treinando o modelo Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Avaliação
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia do modelo: {acc * 100:.2f}%")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 6. Salvar Modelo e Encoder
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)
    with open(encoder_output, 'wb') as f:
        pickle.dump(le, f)
    
    print(f"\nSucesso! Modelo salvo em '{model_output}' e encoder em '{encoder_output}'.")

if __name__ == "__main__":
    # O arquivo gerado pelo script anterior é 'gestures_dataset.csv'
    train_model('gestures_dataset.csv')
