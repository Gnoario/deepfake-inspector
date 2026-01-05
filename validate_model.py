"""
🧠 Calibração Robusta do Threshold do Modelo Deepfake Detector
Autor: Tadeu Sobral Jenuario
Data: 2025-10-17
Descrição:
    - Lê o dataset de validação completo (ou parcial).
    - Faz predições em blocos tolerantes a OOM.
    - Ajusta automaticamente o batch size se o container ficar sem memória.
    - Gera métricas e gráficos de calibração de threshold.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import json
import gc

MODEL_PATH = "/workspace/models/patience-10/deepfake_detector_model.keras"
VAL_DIR = "/workspace/datasets/Dataset/Validation"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 128 if tf.config.list_physical_devices('GPU') else 32
MAX_BATCHES = None  
SAVE_PARTS = True   
SAVE_PATH = os.path.join(os.path.dirname(MODEL_PATH), "best_threshold.json")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

print("🔹 Carregando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

print("🔹 Carregando dataset de validação...")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = val_ds.class_names
print(f"Classes detectadas: {class_names}")

val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).prefetch(buffer_size=tf.data.AUTOTUNE)
if MAX_BATCHES:
    val_ds = val_ds.take(MAX_BATCHES)
    print(f"⚠️ Usando apenas {MAX_BATCHES * BATCH_SIZE} imagens (amostragem rápida)")

print("\n🔹 Gerando previsões de forma robusta...")
y_true, y_scores = [], []
part = 0

for i, (images, labels) in enumerate(val_ds):
    try:
        preds = model.predict(images, verbose=0).flatten()
        y_scores.extend(preds)
        y_true.extend(labels.numpy())

        if SAVE_PARTS and (i + 1) % 50 == 0:
            np.save(f"preds_part_{part}.npy", np.array(y_scores))
            np.save(f"labels_part_{part}.npy", np.array(y_true))
            print(f"💾 Progresso salvo (parte {part}) - {len(y_true)} imagens processadas.")
            part += 1
            y_true, y_scores = [], []  

    except tf.errors.ResourceExhaustedError:
        BATCH_SIZE = max(8, BATCH_SIZE // 2)
        print(f"⚠️ OOM detectado! Reduzindo batch size para {BATCH_SIZE} e continuando...")
        gc.collect()
        tf.keras.backend.clear_session()
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            VAL_DIR,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=False
        ).map(lambda x, y: (x / 255.0, y)).prefetch(buffer_size=tf.data.AUTOTUNE)
        continue

if SAVE_PARTS:
    print("\n🔹 Consolidando partes salvas...")
    preds_files = sorted([f for f in os.listdir() if f.startswith("preds_part_")])
    labels_files = sorted([f for f in os.listdir() if f.startswith("labels_part_")])
    for p, l in zip(preds_files, labels_files):
        y_scores.extend(np.load(p).tolist())
        y_true.extend(np.load(l).tolist())

y_true = np.array(y_true).astype(int)
y_scores = np.array(y_scores)

y_true = np.array(y_true, dtype=int).ravel()
y_scores = np.array(y_scores, dtype=float).ravel()

print(f"✅ Total de amostras avaliadas: {len(y_true)}")

print("\n🔹 Calculando threshold ótimo...")
prec, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
f1_scores = np.nan_to_num(f1_scores)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5

print(f"\n✅ Melhor threshold encontrado: {best_threshold:.4f}")
print(f"Precisão: {prec[best_idx]:.4f}")
print(f"Recall: {recall[best_idx]:.4f}")
print(f"F1-score: {f1_scores[best_idx]:.4f}")

y_pred = (y_scores > best_threshold).astype(int)
print("\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))
print("Matriz de confusão:")
print(confusion_matrix(y_true, y_pred))

with open(SAVE_PATH, "w") as f:
    json.dump({"best_threshold": float(best_threshold)}, f)

plt.figure(figsize=(8, 6))
plt.plot(recall, prec, label="Curva Precisão-Recall", color="blue")
plt.scatter(recall[best_idx], prec[best_idx], color="red", label=f"Threshold ótimo = {best_threshold:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precisão")
plt.title("Curva de Precisão-Recall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(y_scores[y_true == 0], bins=50, alpha=0.6, label="Fake (label=0)")
plt.hist(y_scores[y_true == 1], bins=50, alpha=0.6, label="Real (label=1)")
plt.axvline(best_threshold, color="red", linestyle="--", label=f"Threshold ótimo = {best_threshold:.3f}")
plt.xlabel("Score (sigmoid)")
plt.ylabel("Frequência")
plt.title("Distribuição dos scores")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("score_distribution.png")
plt.close()

print(f"\n💾 Threshold salvo em: {SAVE_PATH}")
print("📊 Gráficos salvos: precision_recall_curve.png e score_distribution.png")
print("\n✅ Calibração robusta concluída com sucesso.")
