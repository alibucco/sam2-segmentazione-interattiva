# Segmentazione interattiva con SAM 2

Progetto per la segmentazione interattiva di oggetti in immagini con [SAM 2](https://github.com/facebookresearch/sam2). Il file `segmentazione.py` è lo script originale del progetto: usa click positivi e negativi per selezionare l'oggetto e salva maschera e overlay.

La repository contiene soltanto il codice del progetto e le istruzioni. Non include SAM 2, immagini personali, dataset, checkpoint o risultati.

## Installazione

Installa SAM 2 seguendo le istruzioni ufficiali; nella struttura locale il suo codice deve trovarsi nella cartella `sam2/`. Su macOS con Apple Silicon:

```bash
git clone https://github.com/facebookresearch/sam2.git sam2
cd sam2
python3 -m venv ../.venv
source ../.venv/bin/activate
SAM2_BUILD_CUDA=0 python -m pip install -e .
python -m pip install -r ../requirements.txt
```

Scarica il checkpoint **SAM 2.1 Tiny** e salvalo in `sam2/checkpoints/sam2.1_hiera_tiny.pt`. Copia poi `segmentazione.py` nella cartella `sam2/`, insieme al codice del modello.

## Uso

Dentro `sam2/`, modifica il percorso dell'immagine nella prima parte di `segmentazione.py` e avvia:

```bash
python segmentazione.py
```

- Click sinistro: punto dell'oggetto da selezionare.
- Click destro: punto da escludere.
- `Invio`: conferma e genera i risultati.
- `Esc`: annulla.

I risultati vengono creati in `results/`, che resta locale.

## Crediti

SAM 2 è un progetto di Meta AI, distribuito con licenza Apache 2.0.
