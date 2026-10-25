"""Segmentazione interattiva di immagini con SAM 2 e click positivi/negativi."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


ROOT = Path(__file__).resolve().parent
SAM2_DIR = ROOT / "sam2"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta un oggetto con click positivi e negativi usando SAM 2."
    )
    parser.add_argument("image", type=Path, help="Percorso dell'immagine da segmentare")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SAM2_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt",
        help="Checkpoint SAM 2.1 tiny (predefinito: sam2/checkpoints/...)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Cartella in cui salvare maschera e overlay",
    )
    return parser.parse_args()


def choose_points(image_bgr):
    import cv2

    points, labels = [], []
    window_name = "SAM 2 — click: sinistro positivo, destro negativo"

    def mouse_callback(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(1)
            print(f"Punto positivo: {(x, y)}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y))
            labels.append(0)
            print(f"Punto negativo: {(x, y)}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    print("Click sinistro = oggetto; click destro = sfondo.")
    print("Premi Invio per confermare, R per azzerare, Esc per annullare.")

    while True:
        preview = image_bgr.copy()
        for (x, y), label in zip(points, labels):
            color = (0, 255, 0) if label else (0, 0, 255)
            cv2.circle(preview, (x, y), 6, color, -1)
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and points:
            break
        if key in (ord("r"), ord("R")):
            points.clear()
            labels.clear()
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Operazione annullata.")

    cv2.destroyAllWindows()
    return points, labels


def main() -> None:
    args = parse_args()
    if not SAM2_DIR.is_dir():
        raise FileNotFoundError(
            "Cartella 'sam2' non trovata. Segui la sezione Installazione del README."
        )
    if not args.image.is_file():
        raise FileNotFoundError(f"Immagine non trovata: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {args.checkpoint}")

    # Consente di usare il checkout locale senza includerlo nella repository.
    sys.path.insert(0, str(SAM2_DIR))

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = np.array(Image.open(args.image).convert("RGB"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Caricamento SAM 2 su: {device}")
    model = build_sam2(
        DEFAULT_CONFIG,
        str(args.checkpoint),
        device=device,
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(model)

    with torch.inference_mode():
        predictor.set_image(image)
    print("SAM 2 pronto.")

    points, labels = choose_points(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
        )

    best_mask = masks[np.argmax(scores)] > 0.5
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "maschera.png"), best_mask.astype(np.uint8) * 255)

    overlay = image.copy()
    color = np.array([0, 180, 255])
    overlay[best_mask] = (0.5 * overlay[best_mask] + 0.5 * color).astype(np.uint8)
    Image.fromarray(overlay).save(args.output_dir / "segmentazione.png")
    print(f"Segmentazione completata. Risultati: {args.output_dir}")


if __name__ == "__main__":
    main()
