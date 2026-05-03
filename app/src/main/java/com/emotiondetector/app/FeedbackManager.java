package com.emotiondetector.app;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * FeedbackManager — Orchestrates the federated learning feedback loop.
 *
 * Responsibilities:
 *   1. Tokenize user text using TextProcessor's tokenizer
 *   2. Send feedback (tokens + correct label) to the FL server
 *   3. Trigger server-side training explicitly (Sync button)
 *   4. Poll for model update, download and hot-swap interpreter
 *   5. Track model version via SharedPreferences
 */
public class FeedbackManager {

    private static final String TAG = "FeedbackManager";
    private static final String PREFS_NAME = "fl_prefs";
    private static final String KEY_MODEL_VERSION = "model_version";
    private static final String UPDATED_MODEL_FILENAME = "text_model_updated.tflite";

    // How long to wait for training to complete before polling (ms)
    private static final int TRAINING_WAIT_MS = 30_000; // 30s initial wait
    private static final int POLL_INTERVAL_MS = 10_000; // poll every 10s
    private static final int MAX_POLL_ATTEMPTS = 12;     // wait up to 2 minutes total

    private final Context context;
    private final FLNetworkClient networkClient;
    private final SharedPreferences prefs;

    public interface ModelUpdateCallback {
        void onModelUpdated(int newVersion);
        void onModelUpdateFailed(String error);
        // Called during polling to show progress
        default void onStatusUpdate(String message) {}
    }

    public interface FeedbackCallback {
        void onFeedbackSent(boolean success, String message, int totalSamples);
    }

    public FeedbackManager(Context context) {
        this.context = context;
        this.networkClient = new FLNetworkClient();
        this.prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    public int getCurrentModelVersion() {
        return prefs.getInt(KEY_MODEL_VERSION, 0);
    }

    /**
     * Submit user feedback to the FL server.
     */
    public void submitFeedback(TextProcessor textProcessor, String rawText,
                               int correctLabel, FeedbackCallback callback) {
        new Thread(() -> {
            try {
                int[] tokens = textProcessor.tokenize(rawText);
                int[] padded = textProcessor.padSequence(tokens, 100);
                Log.d(TAG, "Submitting feedback: " + padded.length + " tokens, label=" + correctLabel);

                FLNetworkClient.FeedbackResponse response =
                        networkClient.submitFeedback(padded, correctLabel, rawText);

                if (response != null) {
                    if (callback != null) {
                        callback.onFeedbackSent(response.success, response.message, response.totalSamples);
                    }
                } else {
                    if (callback != null) {
                        callback.onFeedbackSent(false, "No response from server", 0);
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "Feedback submission error", e);
                if (callback != null) {
                    callback.onFeedbackSent(false, "Error: " + e.getMessage(), 0);
                }
            }
        }).start();
    }

    /**
     * Full Sync flow (called from Sync button):
     *   1. Call /trigger_training to force the server to retrain
     *   2. Wait and poll /check_model_update until a new version appears
     *   3. Download and apply the new model
     */
    public void syncWithServer(ModelUpdateCallback callback) {
        new Thread(() -> {
            try {
                int currentVersion = getCurrentModelVersion();
                Log.d(TAG, "=== Sync started. Current model v" + currentVersion + " ===");

                // Step 1: Check server status and feedback count
                if (callback != null) callback.onStatusUpdate("Connecting to server...");
                FLNetworkClient.ServerStatus status = networkClient.getServerStatus();

                if (!status.online) {
                    if (callback != null) callback.onModelUpdateFailed("Server offline: " + status.message);
                    return;
                }

                Log.d(TAG, "Server online. Feedback on server: " + status.feedbackCount
                        + ", Server model v" + status.modelVersion);

                // Step 2: If there's already a newer model, skip training and download
                if (status.modelVersion > currentVersion) {
                    if (callback != null) callback.onStatusUpdate("New model found! Downloading...");
                    downloadAndApply(currentVersion, callback);
                    return;
                }

                // Step 3: Check if server has enough feedback to train
                if (status.feedbackCount < 1) {
                    if (callback != null)
                        callback.onModelUpdateFailed("No feedback samples on server yet. Send feedback first!");
                    return;
                }

                // Step 4: Trigger training
                if (callback != null)
                    callback.onStatusUpdate("Triggering training on " + status.feedbackCount + " samples...");

                boolean triggered = networkClient.triggerTraining();
                if (!triggered) {
                    // Training might already be running or failed to trigger
                    // Still poll to see if a new version appears
                    Log.w(TAG, "Training trigger returned false — may already be training.");
                }
                Log.d(TAG, "Training triggered. Polling for new model version...");

                // Step 5: Poll for new model (with increasing wait)
                int latestVersion = currentVersion;
                for (int attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {

                    int waitMs = (attempt == 0) ? TRAINING_WAIT_MS : POLL_INTERVAL_MS;
                    int remainingSec = (MAX_POLL_ATTEMPTS - attempt) * (POLL_INTERVAL_MS / 1000);

                    if (callback != null) {
                        callback.onStatusUpdate("Training in progress... (~" + remainingSec + "s remaining)");
                    }

                    Thread.sleep(waitMs);

                    FLNetworkClient.ModelUpdateInfo updateInfo = networkClient.checkModelUpdate(currentVersion);
                    if (updateInfo == null) {
                        Log.w(TAG, "Poll attempt " + attempt + ": server unreachable");
                        continue;
                    }

                    Log.d(TAG, "Poll attempt " + attempt + ": server v"
                            + updateInfo.latestVersion + " > client v" + currentVersion
                            + ", isTraining=" + updateInfo.isTraining);

                    if (updateInfo.hasUpdate) {
                        latestVersion = updateInfo.latestVersion;
                        if (callback != null) callback.onStatusUpdate("New model v" + latestVersion + " ready! Downloading...");
                        downloadAndApply(currentVersion, callback);
                        return;
                    }
                }

                // Timed out polling
                if (callback != null) {
                    callback.onModelUpdateFailed(
                            "Training is taking longer than expected. Try syncing again in a minute.");
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                if (callback != null) callback.onModelUpdateFailed("Sync interrupted");
            } catch (Exception e) {
                Log.e(TAG, "Sync error", e);
                if (callback != null) callback.onModelUpdateFailed("Sync error: " + e.getMessage());
            }
        }).start();
    }

    /**
     * Just check for a newer model and download if available (no trigger).
     */
    public void checkAndApplyModelUpdate(ModelUpdateCallback callback) {
        new Thread(() -> {
            try {
                int currentVersion = getCurrentModelVersion();
                FLNetworkClient.ModelUpdateInfo updateInfo = networkClient.checkModelUpdate(currentVersion);

                if (updateInfo == null) {
                    if (callback != null) callback.onModelUpdateFailed("Could not reach FL server");
                    return;
                }

                if (!updateInfo.hasUpdate) {
                    if (callback != null)
                        callback.onModelUpdateFailed("Already up to date (v" + currentVersion + ")");
                    return;
                }

                downloadAndApply(currentVersion, callback);

            } catch (Exception e) {
                Log.e(TAG, "Update check error", e);
                if (callback != null) callback.onModelUpdateFailed("Error: " + e.getMessage());
            }
        }).start();
    }

    private void downloadAndApply(int currentVersion, ModelUpdateCallback callback) {
        try {
            // Re-fetch latest version info
            FLNetworkClient.ModelUpdateInfo updateInfo = networkClient.checkModelUpdate(currentVersion);
            if (updateInfo == null || !updateInfo.hasUpdate) {
                if (callback != null) callback.onModelUpdateFailed("No update available");
                return;
            }

            File modelFile = new File(context.getFilesDir(), UPDATED_MODEL_FILENAME);
            boolean downloaded = networkClient.downloadModel(modelFile);

            if (!downloaded || !modelFile.exists() || modelFile.length() < 100) {
                if (callback != null) callback.onModelUpdateFailed("Download failed or file invalid");
                return;
            }

            prefs.edit().putInt(KEY_MODEL_VERSION, updateInfo.latestVersion).apply();
            Log.d(TAG, "✅ Model saved: v" + updateInfo.latestVersion + " (" + modelFile.length() + " bytes)");

            if (callback != null) callback.onModelUpdated(updateInfo.latestVersion);

        } catch (Exception e) {
            Log.e(TAG, "Download error", e);
            if (callback != null) callback.onModelUpdateFailed("Download error: " + e.getMessage());
        }
    }

    public Interpreter loadUpdatedModelInterpreter() {
        File modelFile = new File(context.getFilesDir(), UPDATED_MODEL_FILENAME);
        if (!modelFile.exists()) {
            Log.d(TAG, "No updated model file found. Using bundled model.");
            return null;
        }
        try {
            MappedByteBuffer modelBuffer = loadModelFromFile(modelFile);
            Interpreter interpreter = new Interpreter(modelBuffer);
            Log.d(TAG, "Loaded updated model: " + modelFile.length() + " bytes");
            return interpreter;
        } catch (Exception e) {
            Log.e(TAG, "Failed to load updated model", e);
            return null;
        }
    }

    private MappedByteBuffer loadModelFromFile(File file) throws IOException {
        FileInputStream fis = new FileInputStream(file);
        FileChannel fileChannel = fis.getChannel();
        MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
        fis.close();
        return buffer;
    }

    public void checkServerStatus(ServerStatusCallback callback) {
        new Thread(() -> {
            FLNetworkClient.ServerStatus status = networkClient.getServerStatus();
            if (callback != null) callback.onStatusReceived(status);
        }).start();
    }

    public interface ServerStatusCallback {
        void onStatusReceived(FLNetworkClient.ServerStatus status);
    }
}
