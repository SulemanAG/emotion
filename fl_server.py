"""
Federated Learning Aggregation Server for EmotionDetector
=========================================================
This server handles:
  1. Receiving local model weight updates from Android clients
  2. Federated Averaging (FedAvg) aggregation
  3. Serving the latest global model weights back to clients

Endpoints:
  POST /upload_weights   - Client uploads local weights after training
  GET  /global_weights    - Client downloads the latest aggregated global weights
  GET  /status            - Health check / server status
  POST /start_round       - Trigger a new federated round (aggregate collected weights)
"""

import os
import json
import time
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─── In-memory storage ────────────────────────────────────────────────
# Stores weights uploaded by clients in the current round
client_updates = []

# The current global (aggregated) model weights
global_weights = None

# Metadata
current_round = 0
min_clients_per_round = 2  # Minimum clients needed before aggregation
server_start_time = time.time()


# ─── Utility Functions ────────────────────────────────────────────────

def federated_average(updates):
    """
    Perform Federated Averaging (FedAvg) on a list of weight updates.
    Each update is a list of numpy arrays (one per layer).
    """
    if not updates:
        return None

    num_clients = len(updates)
    # Initialize averaged weights with zeros matching the shape of the first update
    avg_weights = []
    for layer_idx in range(len(updates[0])):
        layer_sum = np.zeros_like(np.array(updates[0][layer_idx], dtype=np.float32))
        for client_update in updates:
            layer_sum += np.array(client_update[layer_idx], dtype=np.float32)
        avg_weights.append((layer_sum / num_clients).tolist())

    return avg_weights


# ─── API Endpoints ────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "EmotionDetector Federated Learning Server",
        "version": "1.0.0",
        "status": "running",
        "current_round": current_round,
        "clients_this_round": len(client_updates),
        "uptime_seconds": round(time.time() - server_start_time, 1)
    })


@app.route("/status", methods=["GET"])
def status():
    """Health check and server status."""
    return jsonify({
        "status": "ok",
        "current_round": current_round,
        "clients_this_round": len(client_updates),
        "min_clients_per_round": min_clients_per_round,
        "has_global_weights": global_weights is not None,
        "uptime_seconds": round(time.time() - server_start_time, 1)
    })


@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    """
    Receive local model weights from an Android client.
    Expected JSON body:
    {
        "client_id": "device_xyz",
        "model_type": "text" | "audio",
        "weights": [[layer1_weights], [layer2_weights], ...],
        "num_samples": 100
    }
    """
    global client_updates

    try:
        data = request.get_json(force=True)

        if "weights" not in data:
            return jsonify({"error": "Missing 'weights' field"}), 400

        client_id = data.get("client_id", "anonymous")
        model_type = data.get("model_type", "unknown")
        num_samples = data.get("num_samples", 0)
        weights = data["weights"]

        client_updates.append({
            "client_id": client_id,
            "model_type": model_type,
            "num_samples": num_samples,
            "weights": weights,
            "timestamp": time.time()
        })

        app.logger.info(
            f"Received weights from client '{client_id}' "
            f"(model: {model_type}, samples: {num_samples}). "
            f"Total clients this round: {len(client_updates)}"
        )

        # Auto-aggregate if we have enough clients
        if len(client_updates) >= min_clients_per_round:
            _aggregate()

        return jsonify({
            "status": "accepted",
            "client_id": client_id,
            "clients_this_round": len(client_updates),
            "round": current_round
        })

    except Exception as e:
        app.logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/global_weights", methods=["GET"])
def get_global_weights():
    """
    Return the latest aggregated global model weights.
    Clients call this to update their local models.
    """
    if global_weights is None:
        return jsonify({
            "status": "no_weights",
            "message": "No global weights available yet. Waiting for enough client updates.",
            "round": current_round
        }), 404

    return jsonify({
        "status": "ok",
        "round": current_round,
        "weights": global_weights
    })


@app.route("/start_round", methods=["POST"])
def start_round():
    """Manually trigger aggregation of current round's updates."""
    if len(client_updates) == 0:
        return jsonify({
            "status": "no_updates",
            "message": "No client updates to aggregate."
        }), 400

    _aggregate()

    return jsonify({
        "status": "aggregated",
        "round": current_round,
        "num_clients_aggregated": len(client_updates)
    })


def _aggregate():
    """Internal: run FedAvg and advance the round."""
    global global_weights, current_round, client_updates

    weight_lists = [u["weights"] for u in client_updates]
    global_weights = federated_average(weight_lists)
    current_round += 1

    app.logger.info(
        f"✅ Round {current_round} aggregation complete. "
        f"Aggregated {len(weight_lists)} client updates."
    )

    # Clear updates for next round
    client_updates = []


# ─── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 EmotionDetector FL Server starting on port {port}")
    print(f"   Endpoints:")
    print(f"     GET  /status          - Server status")
    print(f"     POST /upload_weights  - Upload local weights")
    print(f"     GET  /global_weights  - Download global weights")
    print(f"     POST /start_round     - Trigger aggregation")
    app.run(host="0.0.0.0", port=port, debug=False)
