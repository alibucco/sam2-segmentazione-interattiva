import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import cv2
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Carico l'immagine
image = np.array(
    Image.open("images/IMG_6624.jpeg").convert("RGB")
)

# Carico SAM2
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    "checkpoints/sam2.1_hiera_tiny.pt",
    device=device,
    apply_postprocessing=False
)

predictor = SAM2ImagePredictor(model)

print("Elaborazione dell'immagine...")

with torch.inference_mode():
    predictor.set_image(image)

print("SAM 2 pronto")

# Selezione interattiva
selected_points = []
selected_labels = []

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x,y))
        selected_labels.append(1)
        print("Punto positivo:", (x,y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        selected_points.append((x,y))
        selected_labels.append(0)
        print("Punto negativo:", (x,y))

# Visualizzo l'immagine
display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.namedWindow("Seleziona oggetto")
cv2.setMouseCallback("Seleziona oggetto", mouse_callback)

print("Clicca sull'oggetto da segmentare")
print("Premi INVIO dopo aver selezionato il punto")

while True:
    preview = display_image.copy()

    for (x,y), label in zip(selected_points, selected_labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(preview, (x,y), 6, color, -1)

    cv2.imshow("Seleziona oggetto", display_image)

    key = cv2.waitKey(20) & 0xFF

    if key in (10, 13) and selected_points:
        break

    if key == 27:
        cv2.destroyAllWindows()
        raise SystemExit("Operazione annullata")

cv2.destroyAllWindows()

# Segmentazione
input_point = np.array(selected_points, dtype=np.float32)
input_label = np.array(selected_labels, dtype=np.int32)

with torch.inference_mode():
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

best_mask = masks[np.argmax(scores)] > 0.5

# Salvo la maschera
os.makedirs("results", exist_ok=True)

mask_image = (best_mask.astype(np.uint8) * 255)

cv2.imwrite("results/maschera.png", mask_image)

# Creazione overlay
overlay = image.copy()

color = np.array([0, 180, 255])

overlay[best_mask] = (
    0.5 * overlay[best_mask] + 0.5 * color
).astype(np.uint8)

Image.fromarray(overlay).save(
    "results/segmentazione.png"
)

print("Segmentazione completata!")
print("Risultati salvati nella cartella results/")
