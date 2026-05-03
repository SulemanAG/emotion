"""
Federated Learning Server for EmotionDetector — Lightweight Edition
====================================================================
Uses a minimal numpy-based neural network instead of full TensorFlow,
so it works on Render's free tier (512MB RAM).

Training pipeline:
  1. Collect feedback (tokenized text + correct label) from Android app
  2. Train a small MLP using pure numpy (no TF dependency at training time)
  3. Convert trained weights to TFLite using tf.lite (lazy import, only at export)
  4. Serve updated .tflite model to Android app

Endpoints:
  POST /submit_feedback    — receive tokens + correct label
  GET  /check_model_update — check if newer model exists
  GET  /download_model     — download updated .tflite
  POST /trigger_training   — manually trigger training
  GET  /status             — health check
"""

import os
import json
import time
import threading
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# ─── Configuration ────────────────────────────────────────────────────
FEEDBACK_DIR = "fl_data"
MODEL_DIR = "fl_models"
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback_samples.json")
MODEL_VERSION_FILE = os.path.join(MODEL_DIR, "version.json")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "text_model_updated.tflite")

SEQ_LEN = 100
NUM_CLASSES = 4
VOCAB_SIZE = 5000
EMBED_DIM = 16
HIDDEN_DIM = 16
EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]
MIN_SAMPLES_FOR_TRAINING = 5
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 20

# ─── State ────────────────────────────────────────────────────────────
model_version = 0
feedback_samples = []
training_lock = threading.Lock()
is_training = False
server_start_time = time.time()
training_history = []
last_training_error = None

os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Persistence ──────────────────────────────────────────────────────

def load_state():
    global model_version, feedback_samples
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                feedback_samples = json.load(f)
            print(f"[INIT] Loaded {len(feedback_samples)} feedback samples")
        except Exception as e:
            print(f"[INIT] Failed to load feedback: {e}")
            feedback_samples = []
    if os.path.exists(MODEL_VERSION_FILE):
        try:
            with open(MODEL_VERSION_FILE, 'r') as f:
                model_version = json.load(f).get("version", 0)
            print(f"[INIT] Model version: {model_version}")
        except:
            model_version = 0


def save_feedback():
    try:
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(feedback_samples, f)
    except Exception as e:
        print(f"[ERROR] Save feedback failed: {e}")


def save_model_version():
    try:
        with open(MODEL_VERSION_FILE, 'w') as f:
            json.dump({"version": model_version, "timestamp": time.time()}, f)
    except Exception as e:
        print(f"[ERROR] Save version failed: {e}")


# ─── Lightweight Training (numpy) then TFLite export ─────────────────

def train_and_export():
    """Train a small model with numpy, then export to TFLite using TF (lazy import)."""
    global model_version, is_training, last_training_error

    with training_lock:
        if is_training:
            return False
        is_training = True

    last_training_error = None

    try:
        print(f"\n{'='*60}")
        print(f"[TRAIN] Starting training on {len(feedback_samples)} samples...")
        print(f"{'='*60}")

        # 1. Prepare data
        X_raw = []
        y_raw = []
        for s in feedback_samples:
            tokens = s["tokens"]
            if len(tokens) < SEQ_LEN:
                tokens = [0] * (SEQ_LEN - len(tokens)) + tokens
            elif len(tokens) > SEQ_LEN:
                tokens = tokens[-SEQ_LEN:]
            X_raw.append(tokens)
            y_raw.append(s["correct_label"])

        X = np.array(X_raw, dtype=np.int32)
        y = np.array(y_raw, dtype=np.int32)

        # Augment small datasets
        if len(X) < 10:
            repeat = max(2, 10 // len(X) + 1)
            X = np.tile(X, (repeat, 1))
            y = np.tile(y, repeat)
            print(f"[TRAIN] Augmented to {len(X)} samples")

        print(f"[TRAIN] X shape: {X.shape}, y shape: {y.shape}")
        print(f"[TRAIN] Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # 2. Build and train using TF Keras (lightweight model)
        print("[TRAIN] Importing TensorFlow (this may take a moment)...")

        # Minimize TF memory usage
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        print("[TRAIN] TensorFlow imported successfully. Building model...")

        # Clear any previous session
        tf.keras.backend.clear_session()

        # Very lightweight model — fits in 512MB RAM
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=SEQ_LEN),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(HIDDEN_DIM, activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary(print_fn=lambda x: print(f"[TRAIN] {x}"))

        # Convert X to float32 for Keras
        X_float = X.astype(np.float32)

        print(f"[TRAIN] Training for {TRAINING_EPOCHS} epochs...")
        history = model.fit(
            X_float, y,
            epochs=TRAINING_EPOCHS,
            batch_size=min(8, len(X_float)),
            verbose=0
        )

        final_loss = float(history.history['loss'][-1])
        final_acc = float(history.history['accuracy'][-1])
        print(f"[TRAIN] Training complete! Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")

        # 3. Convert to TFLite
        print("[TRAIN] Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)

        model_version += 1
        save_model_version()

        result = {
            "version": model_version,
            "timestamp": time.time(),
            "num_samples": len(feedback_samples),
            "final_loss": final_loss,
            "final_accuracy": final_acc,
            "tflite_size_bytes": len(tflite_model)
        }
        training_history.append(result)

        print(f"[TRAIN] ✅ Model v{model_version} saved!")
        print(f"[TRAIN] TFLite size: {len(tflite_model)} bytes")
        print(f"{'='*60}\n")

        # Cleanup
        del model, tflite_model, X_float
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

        return True

    except Exception as e:
        last_training_error = str(e)
        print(f"[TRAIN] ❌ Training FAILED: {e}")
        traceback.print_exc()
        return False

    finally:
        with training_lock:
            is_training = False


# ─── API Endpoints ────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "EmotionDetector FL Server",
        "version": "3.0.0-lightweight",
        "model_version": model_version,
        "total_feedback": len(feedback_samples),
        "is_training": is_training,
        "last_error": last_training_error
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "model_version": model_version,
        "total_feedback": len(feedback_samples),
        "min_samples_for_training": MIN_SAMPLES_FOR_TRAINING,
        "is_training": is_training,
        "has_trained_model": os.path.exists(TFLITE_MODEL_PATH),
        "last_error": last_training_error,
        "training_history": training_history[-5:],
        "uptime_seconds": round(time.time() - server_start_time, 1)
    })


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    try:
        data = request.get_json(force=True)

        if "tokens" not in data or "correct_label" not in data:
            return jsonify({"error": "Missing 'tokens' or 'correct_label'"}), 400

        tokens = data["tokens"]
        correct_label = int(data["correct_label"])
        raw_text = data.get("raw_text", "")

        if correct_label < 0 or correct_label >= NUM_CLASSES:
            return jsonify({"error": f"Invalid label {correct_label}"}), 400

        sample = {
            "tokens": tokens[:SEQ_LEN],
            "correct_label": correct_label,
            "raw_text": raw_text,
            "timestamp": time.time()
        }
        feedback_samples.append(sample)
        save_feedback()

        emotion_name = EMOTION_LABELS[correct_label]
        print(f"[FB] #{len(feedback_samples)}: '{raw_text[:40]}' → {emotion_name}")

        # Auto-train check
        should_train = (
            len(feedback_samples) >= MIN_SAMPLES_FOR_TRAINING
            and len(feedback_samples) % MIN_SAMPLES_FOR_TRAINING == 0
            and not is_training
        )

        if should_train:
            print(f"[FB] 🚀 Auto-training triggered! ({len(feedback_samples)} samples)")
            t = threading.Thread(target=train_and_export)
            t.daemon = True
            t.start()

        return jsonify({
            "status": "accepted",
            "message": f"Feedback '{emotion_name}' accepted"
                       + (" — training started!" if should_train else ""),
            "total_samples": len(feedback_samples),
            "training_triggered": should_train,
            "model_version": model_version
        })

    except Exception as e:
        print(f"[FB] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/check_model_update", methods=["GET"])
def check_model_update():
    client_version = int(request.args.get("version", 0))
    has_update = model_version > client_version and os.path.exists(TFLITE_MODEL_PATH)
    return jsonify({
        "has_update": has_update,
        "latest_version": model_version,
        "client_version": client_version,
        "total_feedback": len(feedback_samples),
        "is_training": is_training
    })


@app.route("/download_model", methods=["GET"])
def download_model():
    if not os.path.exists(TFLITE_MODEL_PATH):
        return jsonify({"error": "No trained model available yet."}), 404

    print(f"[DL] Serving model v{model_version} ({os.path.getsize(TFLITE_MODEL_PATH)} bytes)")
    return send_file(
        TFLITE_MODEL_PATH,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=f'text_model_v{model_version}.tflite'
    )


@app.route("/trigger_training", methods=["POST"])
def trigger_training():
    if len(feedback_samples) == 0:
        return jsonify({"error": "No feedback samples."}), 400
    if is_training:
        return jsonify({"error": "Training already in progress."}), 409

    t = threading.Thread(target=train_and_export)
    t.daemon = True
    t.start()

    return jsonify({
        "status": "training_started",
        "num_samples": len(feedback_samples)
    })


@app.route("/training_history", methods=["GET"])
def get_training_history():
    return jsonify({
        "history": training_history,
        "current_version": model_version,
        "last_error": last_training_error
    })


@app.route("/reset", methods=["POST"])
def reset():
    global feedback_samples, model_version, training_history, last_training_error
    feedback_samples = []
    model_version = 0
    training_history = []
    last_training_error = None
    save_feedback()
    save_model_version()
    if os.path.exists(TFLITE_MODEL_PATH):
        os.remove(TFLITE_MODEL_PATH)
    return jsonify({"status": "reset"})


# ─── Entry ────────────────────────────────────────────────────────────

load_state()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 EmotionDetector FL Server v3.0 (lightweight)")
    print(f"   Port: {port} | Model v{model_version} | {len(feedback_samples)} samples")
    print(f"   Auto-train after {MIN_SAMPLES_FOR_TRAINING} samples")
    app.run(host="0.0.0.0", port=port, debug=False)
