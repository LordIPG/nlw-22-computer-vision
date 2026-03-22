import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Configurações
DATASET_PATH = 'gestures_dataset.csv'
MODEL_PATH = 'gesture_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'

def preprocess_landmarks(df):
    """
    Transforma as coordenadas absolutas em coordenadas relativas ao pulso (landmark 0).
    Isso torna o modelo invariante à posição da mão na tela.
    """
    # Identifica as colunas de coordenadas (x0, y0, z0, ..., x20, y20, z20)
    # Ignoramos 'target' e 'handedness'
    landmark_cols = [c for c in df.columns if c not in ['target', 'handedness']]
    processed_data = df[landmark_cols].copy()
    
    # Subtrai o pulso (x0, y0, z0) de cada ponto correspondente
    for i in range(21):
        processed_data[f'x{i}'] = processed_data[f'x{i}'] - df['x0']
        processed_data[f'y{i}'] = processed_data[f'y{i}'] - df['y0']
        processed_data[f'z{i}'] = processed_data[f'z{i}'] - df['z0']
        
    return processed_data

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Erro: Arquivo {DATASET_PATH} não encontrado.")
        print("Certifique-se de gravar alguns dados primeiro usando o script de captura.")
        return

    print(f"Lendo dataset: {DATASET_PATH}...")
    # Lemos o CSV. O record_hand_landmarks.py cria um cabeçalho.
    # Usamos encoding='latin-1' para lidar com caracteres especiais comuns no Windows.
    try:
        df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(DATASET_PATH, encoding='latin-1')

    if 'target' not in df.columns:
        print("Erro: O dataset não possui a coluna 'target'. Verifique o formato do CSV.")
        return

    classes = df['target'].unique()
    print(f"Classes encontradas: {classes}")
    
    if len(classes) < 2:
        print("Aviso: Você precisa de pelo menos 2 tipos de gestos para um treinamento útil.")
        # prossegue mesmo assim se o usuário quiser testar o fluxo

    # 1. Pré-processamento
    print("Pré-processando marcos (landmarks)...")
    X = preprocess_landmarks(df)
    y = df['target']

    # 2. Codificação de Rótulos
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 3. Divisão Treino/Teste
    print("Dividindo dados em treino e teste (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded if len(classes) > 1 else None
    )

    # 4. Treinamento
    print("Treinando modelo Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Avaliação
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print(f"ACURÁCIA: {accuracy * 100:.2f}%")
    print("="*30)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 6. Salvando
    print(f"\nSalvando modelo em: {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Salvando encoder em: {ENCODER_PATH}")
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)

    print("\nSucesso! O modelo está pronto para ser usado no script de reconhecimento em tempo real.")

if __name__ == "__main__":
    main()
