package com.emotiondetector.app;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

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
 *   3. Check for model updates after submission
 *   4. Download and apply updated models (hot-swap TFLite interpreter)
 *   5. Track model version via SharedPreferences
 */
public class FeedbackManager {

    private static final String TAG = "FeedbackManager";
    private static final String PREFS_NAME = "fl_prefs";
    private static final String KEY_MODEL_VERSION = "model_version";
    private static final String UPDATED_MODEL_FILENAME = "text_model_updated.tflite";

    private final Context context;
    private final FLNetworkClient networkClient;
    private final SharedPreferences prefs;

    // Callback for when model is updated
    public interface ModelUpdateCallback {
        void onModelUpdated(int newVersion);
        void onModelUpdateFailed(String error);
    }

    // Callback for feedback submission
    public interface FeedbackCallback {
        void onFeedbackSent(boolean success, String message, int totalSamples);
    }

    public FeedbackManager(Context context) {
        this.context = context;
        this.networkClient = new FLNetworkClient();
        this.prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    /**
     * Get the current model version stored locally.
     * Version 0 = original bundled model, 1+ = server-updated models.
     */
    public int getCurrentModelVersion() {
        return prefs.getInt(KEY_MODEL_VERSION, 0);
    }

    /**
     * Submit user feedback to the FL server.
     *
     * @param textProcessor  the TextProcessor instance (to access tokenizer)
     * @param rawText        the original text the user typed
     * @param correctLabel   index of the correct emotion (0=angry, 1=happy, 2=neutral, 3=sad)
     * @param callback       callback for the result
     */
    public void submitFeedback(TextProcessor textProcessor, String rawText,
                               int correctLabel, FeedbackCallback callback) {
        new Thread(() -> {
            try {
                // 1. Tokenize the text using TextProcessor's public tokenizer
                int[] tokens = textProcessor.tokenize(rawText);
                int[] padded = textProcessor.padSequence(tokens, 100);

                Log.d(TAG, "Tokenized feedback: " + padded.length + " tokens, label=" + correctLabel);

                // 2. Send to server
                FLNetworkClient.FeedbackResponse response =
                        networkClient.submitFeedback(padded, correctLabel, rawText);

                if (response != null) {
                    Log.d(TAG, "Feedback response: " + response.message);
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
     * Check for model updates and download if available.
     *
     * @param callback callback for the result
     */
    public void checkAndApplyModelUpdate(ModelUpdateCallback callback) {
        new Thread(() -> {
            try {
                int currentVersion = getCurrentModelVersion();
                Log.d(TAG, "Checking for model updates. Current version: " + currentVersion);

                // 1. Check if server has a newer model
                FLNetworkClient.ModelUpdateInfo updateInfo =
                        networkClient.checkModelUpdate(currentVersion);

                if (updateInfo == null) {
                    Log.d(TAG, "Could not reach server for update check.");
                    if (callback != null) {
                        callback.onModelUpdateFailed("Could not reach FL server");
                    }
                    return;
                }

                if (!updateInfo.hasUpdate) {
                    Log.d(TAG, "No model update available. Server version: " + updateInfo.latestVersion);
                    if (callback != null) {
                        callback.onModelUpdateFailed("Already up to date (v" + currentVersion + ")");
                    }
                    return;
                }

                // 2. Download the new model
                File modelFile = new File(context.getFilesDir(), UPDATED_MODEL_FILENAME);
                boolean downloaded = networkClient.downloadModel(modelFile);

                if (!downloaded) {
                    Log.e(TAG, "Model download failed.");
                    if (callback != null) {
                        callback.onModelUpdateFailed("Download failed");
                    }
                    return;
                }

                // 3. Validate the downloaded file
                if (!modelFile.exists() || modelFile.length() < 1000) {
                    Log.e(TAG, "Downloaded model file is invalid. Size: " +
                            (modelFile.exists() ? modelFile.length() : "missing"));
                    if (callback != null) {
                        callback.onModelUpdateFailed("Downloaded model is invalid");
                    }
                    return;
                }

                // 4. Save the new version number
                prefs.edit().putInt(KEY_MODEL_VERSION, updateInfo.latestVersion).apply();
                Log.d(TAG, "✅ Model updated to version " + updateInfo.latestVersion);

                if (callback != null) {
                    callback.onModelUpdated(updateInfo.latestVersion);
                }

            } catch (Exception e) {
                Log.e(TAG, "Model update error", e);
                if (callback != null) {
                    callback.onModelUpdateFailed("Error: " + e.getMessage());
                }
            }
        }).start();
    }

    /**
     * Load the updated model file from internal storage and create a new TFLite Interpreter.
     * Returns null if no updated model exists (app should use the bundled model).
     */
    public Interpreter loadUpdatedModelInterpreter() {
        File modelFile = new File(context.getFilesDir(), UPDATED_MODEL_FILENAME);
        if (!modelFile.exists()) {
            Log.d(TAG, "No updated model file found. Using bundled model.");
            return null;
        }

        try {
            MappedByteBuffer modelBuffer = loadModelFromFile(modelFile);
            Interpreter interpreter = new Interpreter(modelBuffer);
            Log.d(TAG, "Loaded updated model from " + modelFile.getAbsolutePath());
            return interpreter;
        } catch (Exception e) {
            Log.e(TAG, "Failed to load updated model", e);
            return null;
        }
    }

    private MappedByteBuffer loadModelFromFile(File file) throws IOException {
        FileInputStream fis = new FileInputStream(file);
        FileChannel fileChannel = fis.getChannel();
        long fileSize = fileChannel.size();
        MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
        fis.close();
        return buffer;
    }

    /**
     * Get server status (online check, model version, feedback count).
     */
    public void checkServerStatus(ServerStatusCallback callback) {
        new Thread(() -> {
            FLNetworkClient.ServerStatus status = networkClient.getServerStatus();
            if (callback != null) {
                callback.onStatusReceived(status);
            }
        }).start();
    }

    public interface ServerStatusCallback {
        void onStatusReceived(FLNetworkClient.ServerStatus status);
    }
}
