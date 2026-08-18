# AI-Based Audio Classification & Edge Optimization

A full ML pipeline that classifies short audio clips into 5 sound categories — two alarm sounds (fire alarm, ambulance siren) and three non-alarm sounds (car horn, dog bark, background noise) — then optimises the trained model for edge deployment through post-training quantisation and TFLite export.

## Pipeline

1. **Feature extraction** — audio clips are loaded, padded/trimmed to a fixed duration, and converted to MFCC (Mel-frequency cepstral coefficient) spectrograms.
2. **CNN training** — a compact 3-block Conv2D network is trained on the MFCC feature maps, with class weighting to handle any class imbalance.
3. **Evaluation** — accuracy, per-class precision/recall/F1, and a confusion matrix on a held-out test set.
4. **Post-training quantisation** — two strategies:
   - Dynamic-range quantisation (weights only)
   - Full integer (int8) quantisation, calibrated on a representative subset of the training data
5. **TFLite export & edge-side evaluation** — the quantised models are run directly through the TFLite interpreter (not the original Keras model) to measure real accuracy on-device, and compared against the float32 baseline.
6. **Model size comparison** — float32 vs. quantised model sizes, since size and latency are the actual point of edge optimisation.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Point `DATA_DIR` in the notebook at a folder organised like:

```
data/
  fire_alarm/
    clip_001.wav
    ...
  ambulance_siren/
    ...
  car_horn/
    ...
  dog_bark/
    ...
  background_noise/
    ...
```

## Usage

Open `AI_Audio_Classification_Edge_Optimization.ipynb` and run cells top to bottom. Update `DATA_DIR` and `CLASSES` in the configuration cell to point at your dataset before running the training cells.
