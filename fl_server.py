"""
Federated Learning Server for EmotionDetector
==============================================
Real FL pipeline:
  1. POST /submit_feedback  — receives tokenized text + correct emotion label from the app
  2. Auto-training          — when enough feedback samples accumulate, retrains the text model
  3. GET /check_model_update — app checks if a newer model version exists
  4. GET /download_model     — app downloads the updated .tflite model
  5. GET /status             — health check with training stats

Model Architecture (matches the Android app's text_model.tflite):
  Input:  float[1][100]   — padded token sequences
  Output: float[1][4]     — emotion probabilities [angry, happy, neutral, sad]

  Keras model:
    Embedding(vocab_size, 64, input_length=100)
    → LSTM(64)
    → Dense(32, relu)
    → Dropout(0.3)
    → Dense(4, softmax)
"""

import os
import json
import time
import threading
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
EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]
VOCAB_SIZE = 50000  # Should match the tokenizer word_index size
MIN_SAMPLES_FOR_TRAINING = 5  # Auto-train after this many new feedback samples
TRAINING_EPOCHS = 10
BATCH_SIZE = 8

# ─── State ────────────────────────────────────────────────────────────
model_version = 0
feedback_samples = []
training_lock = threading.Lock()
is_training = False
server_start_time = time.time()
training_history = []

# ─── Initialize directories ──────────────────────────────────────────
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_state():
    """Load persisted feedback samples and model version on startup."""
    global model_version, feedback_samples

    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                feedback_samples = json.load(f)
            app.logger.info(f"Loaded {len(feedback_samples)} feedback samples from disk.")
        except Exception as e:
            app.logger.error(f"Failed to load feedback: {e}")
            feedback_samples = []

    if os.path.exists(MODEL_VERSION_FILE):
        try:
            with open(MODEL_VERSION_FILE, 'r') as f:
                info = json.load(f)
                model_version = info.get("version", 0)
            app.logger.info(f"Current model version: {model_version}")
        except Exception as e:
            app.logger.error(f"Failed to load version: {e}")
            model_version = 0


def save_feedback():
    """Persist feedback samples to disk."""
    try:
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(feedback_samples, f)
    except Exception as e:
        app.logger.error(f"Failed to save feedback: {e}")


def save_model_version():
    """Persist model version to disk."""
    try:
        with open(MODEL_VERSION_FILE, 'w') as f:
            json.dump({"version": model_version, "timestamp": time.time()}, f)
    except Exception as e:
        app.logger.error(f"Failed to save version: {e}")


# ─── Model Training ──────────────────────────────────────────────────

def build_model():
    """Build the Keras text classification model matching the TFLite architecture."""
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=SEQ_LEN),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model():
    """Train the model on accumulated feedback data and export to TFLite."""
    global model_version, is_training

    with training_lock:
        if is_training:
            app.logger.info("Training already in progress, skipping.")
            return False
        is_training = True

    try:
        import tensorflow as tf

        app.logger.info(f"Starting training on {len(feedback_samples)} samples...")

        # Prepare training data
        X = []
        y = []
        for sample in feedback_samples:
            tokens = sample["tokens"]
            label = sample["correct_label"]
            # Ensure tokens are the right length
            if len(tokens) < SEQ_LEN:
                tokens = [0] * (SEQ_LEN - len(tokens)) + tokens
            elif len(tokens) > SEQ_LEN:
                tokens = tokens[-SEQ_LEN:]
            X.append(tokens)
            y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        app.logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        app.logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Build and train model
        model = build_model()

        # Data augmentation: duplicate small datasets to have enough for training
        if len(X) < BATCH_SIZE:
            repeat = max(2, BATCH_SIZE // len(X) + 1)
            X = np.tile(X, (repeat, 1))
            y = np.tile(y, repeat)
            app.logger.info(f"Augmented data to {len(X)} samples for stable training.")

        history = model.fit(
            X, y,
            epochs=TRAINING_EPOCHS,
            batch_size=min(BATCH_SIZE, len(X)),
            validation_split=0.1 if len(X) >= 10 else 0.0,
            verbose=1
        )

        # Record training history
        train_result = {
            "version": model_version + 1,
            "timestamp": time.time(),
            "num_samples": len(feedback_samples),
            "final_loss": float(history.history['loss'][-1]),
            "final_accuracy": float(history.history['accuracy'][-1]),
        }
        training_history.append(train_result)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        # Save TFLite model
        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)

        model_version += 1
        save_model_version()

        app.logger.info(
            f"✅ Training complete! Model v{model_version} saved. "
            f"Loss: {train_result['final_loss']:.4f}, "
            f"Accuracy: {train_result['final_accuracy']:.4f}, "
            f"TFLite size: {len(tflite_model)} bytes"
        )

        return True

    except Exception as e:
        app.logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        with training_lock:
            is_training = False


# ─── API Endpoints ────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "EmotionDetector Federated Learning Server",
        "version": "2.0.0",
        "status": "running",
        "model_version": model_version,
        "total_feedback": len(feedback_samples),
        "is_training": is_training,
        "uptime_seconds": round(time.time() - server_start_time, 1)
    })


@app.route("/status", methods=["GET"])
def status():
    """Health check and FL status."""
    return jsonify({
        "status": "ok",
        "model_version": model_version,
        "total_feedback": len(feedback_samples),
        "min_samples_for_training": MIN_SAMPLES_FOR_TRAINING,
        "is_training": is_training,
        "has_trained_model": os.path.exists(TFLITE_MODEL_PATH),
        "training_history": training_history[-5:],  # Last 5 training runs
        "uptime_seconds": round(time.time() - server_start_time, 1)
    })


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    """
    Receive tokenized text + correct emotion label from the Android app.

    Expected JSON body:
    {
        "tokens": [0, 0, ..., 42, 15, 3],   // int[100] padded token sequence
        "correct_label": 2,                    // 0=angry, 1=happy, 2=neutral, 3=sad
        "raw_text": "I feel okay today",       // original text (for logging)
        "seq_len": 100                         // sequence length
    }
    """
    try:
        data = request.get_json(force=True)

        if "tokens" not in data or "correct_label" not in data:
            return jsonify({"error": "Missing 'tokens' or 'correct_label'"}), 400

        tokens = data["tokens"]
        correct_label = int(data["correct_label"])
        raw_text = data.get("raw_text", "")
        seq_len = data.get("seq_len", SEQ_LEN)

        # Validate
        if correct_label < 0 or correct_label >= NUM_CLASSES:
            return jsonify({"error": f"Invalid label {correct_label}. Must be 0-{NUM_CLASSES-1}"}), 400

        if not isinstance(tokens, list) or len(tokens) == 0:
            return jsonify({"error": "Tokens must be a non-empty list"}), 400

        # Store feedback sample
        sample = {
            "tokens": tokens[:SEQ_LEN],  # Ensure max SEQ_LEN
            "correct_label": correct_label,
            "raw_text": raw_text,
            "timestamp": time.time()
        }
        feedback_samples.append(sample)
        save_feedback()

        emotion_name = EMOTION_LABELS[correct_label]
        app.logger.info(
            f"📥 Feedback received: '{raw_text[:50]}...' → {emotion_name} "
            f"({len(tokens)} tokens). Total: {len(feedback_samples)} samples."
        )

        # Auto-train if we have enough samples
        should_train = (
            len(feedback_samples) >= MIN_SAMPLES_FOR_TRAINING
            and len(feedback_samples) % MIN_SAMPLES_FOR_TRAINING == 0
            and not is_training
        )

        if should_train:
            app.logger.info(f"🚀 Auto-training triggered ({len(feedback_samples)} samples)...")
            training_thread = threading.Thread(target=train_model)
            training_thread.daemon = True
            training_thread.start()

        return jsonify({
            "status": "accepted",
            "message": f"Feedback '{emotion_name}' accepted"
                       + (" — training started!" if should_train else ""),
            "total_samples": len(feedback_samples),
            "training_triggered": should_train,
            "model_version": model_version
        })

    except Exception as e:
        app.logger.error(f"Feedback error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/check_model_update", methods=["GET"])
def check_model_update():
    """
    Check if a newer model version is available.
    Query parameter: ?version=<current_version>
    """
    try:
        client_version = int(request.args.get("version", 0))
        has_update = (
            model_version > client_version
            and os.path.exists(TFLITE_MODEL_PATH)
        )

        return jsonify({
            "has_update": has_update,
            "latest_version": model_version,
            "client_version": client_version,
            "total_feedback": len(feedback_samples),
            "is_training": is_training
        })

    except Exception as e:
        app.logger.error(f"Update check error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/download_model", methods=["GET"])
def download_model():
    """
    Download the latest trained TFLite model.
    Returns the .tflite binary file.
    """
    if not os.path.exists(TFLITE_MODEL_PATH):
        return jsonify({
            "error": "No trained model available yet. Need more feedback samples."
        }), 404

    app.logger.info(f"📤 Serving model v{model_version} ({os.path.getsize(TFLITE_MODEL_PATH)} bytes)")
    return send_file(
        TFLITE_MODEL_PATH,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=f'text_model_v{model_version}.tflite'
    )


@app.route("/trigger_training", methods=["POST"])
def trigger_training():
    """Manually trigger model training on accumulated feedback."""
    if len(feedback_samples) == 0:
        return jsonify({"error": "No feedback samples to train on."}), 400

    if is_training:
        return jsonify({"error": "Training already in progress."}), 409

    training_thread = threading.Thread(target=train_model)
    training_thread.daemon = True
    training_thread.start()

    return jsonify({
        "status": "training_started",
        "num_samples": len(feedback_samples),
        "current_version": model_version
    })


@app.route("/training_history", methods=["GET"])
def get_training_history():
    """Get the history of all training runs."""
    return jsonify({
        "history": training_history,
        "total_runs": len(training_history),
        "current_version": model_version
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Reset all feedback data and model (for testing)."""
    global feedback_samples, model_version, training_history

    feedback_samples = []
    model_version = 0
    training_history = []

    save_feedback()
    save_model_version()

    if os.path.exists(TFLITE_MODEL_PATH):
        os.remove(TFLITE_MODEL_PATH)

    return jsonify({"status": "reset", "message": "All data and models cleared."})


# ─── Entry Point ──────────────────────────────────────────────────────

# Load persisted state on startup
load_state()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 EmotionDetector FL Server v2.0 starting on port {port}")
    print(f"   Model version: {model_version}")
    print(f"   Feedback samples: {len(feedback_samples)}")
    print(f"   Auto-train threshold: {MIN_SAMPLES_FOR_TRAINING} samples")
    print(f"   Endpoints:")
    print(f"     POST /submit_feedback    — Submit feedback (tokens + label)")
    print(f"     GET  /check_model_update — Check for model updates")
    print(f"     GET  /download_model     — Download updated .tflite model")
    print(f"     POST /trigger_training   — Manually trigger training")
    print(f"     GET  /status             — Server status")
    app.run(host="0.0.0.0", port=port, debug=False)
