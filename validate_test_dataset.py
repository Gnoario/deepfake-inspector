"""
🧠 Deepfake Detector - Avaliação em Lote (todas as imagens de Test)
Autor: Tadeu Sobral Jenuario
Data: 2025-10-17

Descrição:
    - Carrega o modelo treinado e o threshold ótimo.
    - Percorre todas as imagens da pasta Test (Fake e Real).
    - Exibe e salva o resultado da classificação (Deepfake ou Real).
"""

import os
import json
import numpy as np
import pandas as pd
from tensorflow import keras
from PIL import Image
from tqdm import tqdm

MODEL_PATH = "/workspace/models/patience-10/deepfake_detector_model.keras"
THRESHOLD_PATH = "/workspace/models/patience-10/best_threshold.json"
TEST_DIR = "/workspace/datasets/Dataset/Test"
IMAGE_SIZE = (256, 256)
OUTPUT_CSV = "/workspace/test_results.csv"

print("🔹 Carregando modelo...")
model = keras.models.load_model(MODEL_PATH)

if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "r") as f:
        best_threshold = json.load(f)["best_threshold"]
else:
    best_threshold = 0.5
print(f"🔹 Threshold ótimo carregado: {best_threshold:.4f}\n")

def preprocess_image(image_path, target_size=(256, 256)):
    """Lê e prepara uma imagem para o modelo."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"⚠️ Erro ao processar {image_path}: {e}")
        return None

results = []

print("🔹 Avaliando todas as imagens da pasta Test...\n")

for root, _, files in os.walk(TEST_DIR):
    for file in tqdm(files, desc=f"Processando {root}", ncols=100):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(root, file)
        img = preprocess_image(img_path, IMAGE_SIZE)
        if img is None:
            continue

        pred = model.predict(img, verbose=0)
        score = float(pred[0][0])
        label = "Deepfake" if score < best_threshold else "Real"
        confidence = (1 - score) * 100 if label == "Deepfake" else score * 100

        expected_class = os.path.basename(root) 

        results.append({
            "image": img_path,
            "expected": expected_class,
            "predicted": label,
            "score": round(score, 4),
            "confidence": round(confidence, 2),
        })

df = pd.DataFrame(results)
print("\n✅ Avaliação concluída!")
print(f"📊 Total de imagens: {len(df)}")
print(f"💾 Resultados salvos em: {OUTPUT_CSV}")

df.to_csv(OUTPUT_CSV, index=False)

total = len(df)
acertos = (df["expected"] == df["predicted"]).sum()
precisao = acertos / total * 100 if total > 0 else 0

print(f"🎯 Precisão total: {precisao:.2f}%")

print("\n📋 Amostras:")
print(df.head(10).to_string(index=False))
