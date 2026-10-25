# Segmentazione interattiva con SAM 2

Progetto per la segmentazione interattiva di oggetti in immagini usando [SAM 2](https://github.com/facebookresearch/sam2). L'utente indica l'oggetto con click positivi e può correggere le aree indesiderate con click negativi; il programma salva una maschera binaria e un'immagine con overlay.

Il repository contiene solo il codice necessario per riprodurre l'esperimento: non include fotografie personali, dataset, pesi del modello o risultati generati.

## Requisiti

- Python 3.10 o superiore
- macOS con Apple Silicon, oppure un computer con CPU o GPU compatibile con PyTorch
- Git

## Installazione

Clona questo repository e crea un ambiente virtuale:

```bash
git clone https://github.com/USERNAME/sam2-segmentazione-interattiva.git
cd sam2-segmentazione-interattiva
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Scarica anche SAM 2 nella cartella prevista dal programma:

```bash
git clone https://github.com/facebookresearch/sam2.git sam2
SAM2_BUILD_CUDA=0 python -m pip install -e ./sam2
```

Infine scarica il checkpoint **SAM 2.1 Tiny** dalla documentazione ufficiale di SAM 2 e salvalo come:

```text
sam2/checkpoints/sam2.1_hiera_tiny.pt
```

Il checkpoint non viene incluso nella repository perché è un file di grandi dimensioni.

## Uso

Metti un'immagine in una cartella locale a scelta e avvia:

```bash
python segmentazione_interattiva.py percorso/della/immagine.jpg
```

Nella finestra visualizzata:

- click sinistro: aggiunge un punto dell'oggetto da mantenere;
- click destro: aggiunge un punto dello sfondo da escludere;
- `R`: cancella i punti e ricomincia;
- `Invio`: esegue la segmentazione;
- `Esc`: annulla.

I file `maschera.png` e `segmentazione.png` vengono creati in `results/`, cartella che resta locale e non viene pubblicata.

Per scegliere una cartella diversa per i risultati:

```bash
python segmentazione_interattiva.py percorso/della/immagine.jpg --output-dir risultati/prova-01
```

## Crediti

Il modello è [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2) di Meta AI, distribuito con licenza Apache 2.0. Questo repository contiene lo script di interazione e la documentazione del progetto.
