package com.emotiondetector.app;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * FLNetworkClient — Handles all HTTP communication with the Federated Learning server.
 *
 * Endpoints used:
 *   POST /submit_feedback   — send tokenized text + correct label
 *   GET  /check_model_update — check if a newer model version exists
 *   GET  /download_model     — download the updated .tflite binary
 *   GET  /status             — server health check
 */
public class FLNetworkClient {

    private static final String TAG = "FLNetworkClient";
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    // ── Server URL ──────────────────────────────────────────────────────
    // Update this after deploying to Render
    private static final String BASE_URL = "https://emotion-fl-server.onrender.com";

    private final OkHttpClient client;

    public FLNetworkClient() {
        client = new OkHttpClient.Builder()
                .connectTimeout(90, TimeUnit.SECONDS)   // Render free tier cold start = up to 50s
                .readTimeout(120, TimeUnit.SECONDS)      // Training response can take a while
                .writeTimeout(60, TimeUnit.SECONDS)
                .retryOnConnectionFailure(true)
                .build();
    }

    // ── Data classes ────────────────────────────────────────────────────

    public static class ModelUpdateInfo {
        public final boolean hasUpdate;
        public final int latestVersion;
        public final int totalFeedback;
        public final boolean isTraining;

        public ModelUpdateInfo(boolean hasUpdate, int latestVersion, int totalFeedback, boolean isTraining) {
            this.hasUpdate = hasUpdate;
            this.latestVersion = latestVersion;
            this.totalFeedback = totalFeedback;
            this.isTraining = isTraining;
        }
    }

    public static class FeedbackResponse {
        public final boolean success;
        public final String message;
        public final int totalSamples;

        public FeedbackResponse(boolean success, String message, int totalSamples) {
            this.success = success;
            this.message = message;
            this.totalSamples = totalSamples;
        }
    }

    public static class ServerStatus {
        public final boolean online;
        public final int modelVersion;
        public final int feedbackCount;
        public final boolean isTraining;
        public final String message;

        public ServerStatus(boolean online, int modelVersion, int feedbackCount, boolean isTraining, String message) {
            this.online = online;
            this.modelVersion = modelVersion;
            this.feedbackCount = feedbackCount;
            this.isTraining = isTraining;
            this.message = message;
        }
    }

    // ── API Methods ─────────────────────────────────────────────────────

    /**
     * Submit user feedback (tokenized text + correct emotion label) to the FL server.
     *
     * @param tokens       padded token sequence (int[100])
     * @param correctLabel index of the correct emotion (0=angry, 1=happy, 2=neutral, 3=sad)
     * @param rawText      the original text input (for server-side logging)
     * @return FeedbackResponse or null on failure
     */
    public FeedbackResponse submitFeedback(int[] tokens, int correctLabel, String rawText) {
        try {
            JSONObject json = new JSONObject();
            JSONArray tokenArray = new JSONArray();
            for (int t : tokens) {
                tokenArray.put(t);
            }
            json.put("tokens", tokenArray);
            json.put("correct_label", correctLabel);
            json.put("raw_text", rawText);
            json.put("seq_len", tokens.length);

            RequestBody body = RequestBody.create(json.toString(), JSON);
            Request request = new Request.Builder()
                    .url(BASE_URL + "/submit_feedback")
                    .post(body)
                    .build();

            try (Response response = client.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    JSONObject resp = new JSONObject(response.body().string());
                    Log.d(TAG, "Feedback submitted successfully: " + resp.toString());
                    return new FeedbackResponse(
                            true,
                            resp.optString("message", "Feedback accepted"),
                            resp.optInt("total_samples", 0)
                    );
                } else {
                    String errorBody = response.body() != null ? response.body().string() : "No body";
                    Log.e(TAG, "Feedback submission failed: " + response.code() + " - " + errorBody);
                    return new FeedbackResponse(false, "Server error: " + response.code(), 0);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Feedback submission error", e);
            return new FeedbackResponse(false, "Network error: " + e.getMessage(), 0);
        }
    }

    /**
     * Check if a newer model version exists on the server.
     *
     * @param currentVersion the app's current model version (0 = original bundled model)
     * @return ModelUpdateInfo or null on failure
     */
    public ModelUpdateInfo checkModelUpdate(int currentVersion) {
        try {
            Request request = new Request.Builder()
                    .url(BASE_URL + "/check_model_update?version=" + currentVersion)
                    .get()
                    .build();

            try (Response response = client.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    JSONObject resp = new JSONObject(response.body().string());
                    boolean hasUpdate = resp.optBoolean("has_update", false);
                    int latestVersion = resp.optInt("latest_version", currentVersion);
                    int totalFeedback = resp.optInt("total_feedback", 0);
                    boolean isTraining = resp.optBoolean("is_training", false);
                    Log.d(TAG, "Update check: hasUpdate=" + hasUpdate
                            + ", v" + latestVersion + ", training=" + isTraining);
                    return new ModelUpdateInfo(hasUpdate, latestVersion, totalFeedback, isTraining);
                } else {
                    Log.e(TAG, "Update check failed: " + response.code());
                    return null;
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Update check error", e);
            return null;
        }
    }

    /**
     * Explicitly trigger model training on the server.
     * Returns true if training was started successfully.
     */
    public boolean triggerTraining() {
        try {
            RequestBody body = RequestBody.create("{}", JSON);
            Request request = new Request.Builder()
                    .url(BASE_URL + "/trigger_training")
                    .post(body)
                    .build();

            try (Response response = client.newCall(request).execute()) {
                String bodyStr = response.body() != null ? response.body().string() : "";
                Log.d(TAG, "Trigger training response: " + response.code() + " " + bodyStr);
                // 200 = started, 409 = already running (both are OK)
                return response.code() == 200 || response.code() == 409;
            }
        } catch (Exception e) {
            Log.e(TAG, "Trigger training error", e);
            return false;
        }
    }

    /**
     * Download the latest updated model from the server and save to local file.
     *
     * @param savePath the local file path to save the downloaded .tflite model
     * @return true if download succeeded
     */
    public boolean downloadModel(File savePath) {
        try {
            Request request = new Request.Builder()
                    .url(BASE_URL + "/download_model")
                    .get()
                    .build();

            try (Response response = client.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    InputStream inputStream = response.body().byteStream();
                    FileOutputStream fos = new FileOutputStream(savePath);
                    byte[] buffer = new byte[8192];
                    int bytesRead;
                    long totalBytes = 0;
                    while ((bytesRead = inputStream.read(buffer)) != -1) {
                        fos.write(buffer, 0, bytesRead);
                        totalBytes += bytesRead;
                    }
                    fos.flush();
                    fos.close();
                    inputStream.close();
                    Log.d(TAG, "Model downloaded: " + totalBytes + " bytes → " + savePath.getAbsolutePath());
                    return true;
                } else {
                    Log.e(TAG, "Model download failed: " + response.code());
                    return false;
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Model download error", e);
            return false;
        }
    }

    /**
     * Get server status / health check.
     */
    public ServerStatus getServerStatus() {
        try {
            Request request = new Request.Builder()
                    .url(BASE_URL + "/status")
                    .get()
                    .build();

            try (Response response = client.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    JSONObject resp = new JSONObject(response.body().string());
                    return new ServerStatus(
                            true,
                            resp.optInt("model_version", 0),
                            resp.optInt("total_feedback", 0),
                            resp.optBoolean("is_training", false),
                            resp.optString("status", "ok")
                    );
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Server status check failed", e);
        }
        return new ServerStatus(false, 0, 0, false, "Server unreachable");
    }
}
