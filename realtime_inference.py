"""
Real-time audio classification.

Listens through the microphone in short windows, runs each window through
the trained (quantised) model, and prints an alert whenever one of the
configured "alarming" classes is detected.

Requires the model already trained and exported by the notebook:
    audio_cnn_int8_quant.tflite

Usage:
    pip install sounddevice
    python realtime_inference.py
"""

import queue
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf

# --- Must match training configuration exactly ---
SAMPLE_RATE = 22050
DURATION = 4.0          # seconds per window analysed
N_MFCC = 40
MAX_FRAMES = int(np.ceil(SAMPLE_RATE * DURATION / 512))

CLASSES = [
    "fire_alarm", "ambulance_siren", "car_horn", "dog_bark", "background_noise",
]  # must match the CLASSES order used during training, exactly

ALARM_CLASSES = {"fire_alarm", "ambulance_siren"}

CONFIDENCE_THRESHOLD = 0.70   # only report a detection above this confidence
MODEL_PATH = "audio_cnn_int8_quant.tflite"


def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_frames=MAX_FRAMES):
    """Same feature extraction as training — must stay identical or predictions won't make sense."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]
    return mfcc.astype(np.float32)


class RealtimeClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.is_int8 = self.input_details["dtype"] == np.int8

    def predict(self, audio_window):
        features = extract_mfcc(audio_window)
        features = features[np.newaxis, ..., np.newaxis]  # (1, n_mfcc, frames, 1)

        if self.is_int8:
            scale, zero_point = self.input_details["quantization"]
            features = (features / scale + zero_point).astype(np.int8)
        else:
            features = features.astype(np.float32)

        self.interpreter.set_tensor(self.input_details["index"], features)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details["index"])[0]

        if self.is_int8:
            out_scale, out_zero_point = self.output_details["quantization"]
            output = (output.astype(np.float32) - out_zero_point) * out_scale

        pred_idx = int(np.argmax(output))
        confidence = float(output[pred_idx])
        return CLASSES[pred_idx], confidence


def listen_loop():
    classifier = RealtimeClassifier()
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_q.put(indata[:, 0].copy())

    window_samples = int(SAMPLE_RATE * DURATION)
    buffer = np.zeros(0, dtype=np.float32)

    print("Listening... (Ctrl+C to stop)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while True:
            buffer = np.concatenate([buffer, audio_q.get()])
            while len(buffer) >= window_samples:
                window = buffer[:window_samples]
                buffer = buffer[window_samples:]

                label, confidence = classifier.predict(window)

                if confidence >= CONFIDENCE_THRESHOLD:
                    if label in ALARM_CLASSES:
                        print(f"ALARM DETECTED: {label}  (confidence: {confidence:.2f})")
                    else:
                        print(f"Detected: {label}  (confidence: {confidence:.2f})")
                else:
                    print(f"No confident detection (best guess: {label}, {confidence:.2f})")


if __name__ == "__main__":
    listen_loop()
