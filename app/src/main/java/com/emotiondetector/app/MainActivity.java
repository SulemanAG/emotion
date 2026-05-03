package com.emotiondetector.app;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "EmotionDetector";
    private static final int PERMISSION_REQUEST_CODE = 100;
    private static final int MAX_RECORDING_SECONDS = 10;

    // ── Existing UI ─────────────────────────────────────────────────────
    private EditText etTextInput;
    private Button btnAnalyzeText;
    private Button btnStartRecording;
    private Button btnStopRecording;
    private Button btnAnalyzeAudio;
    private Button btnAnalyzeAll;
    private ProgressBar progressBar;
    private TextView tvStatus;

    private CardView cardTextResult;
    private CardView cardAudioResult;
    private CardView cardFusedResult;
    private TextView tvTextEmotion;
    private TextView tvTextConfidence;
    private TextView tvAudioEmotion;
    private TextView tvAudioConfidence;
    private TextView tvFusedEmotion;
    private TextView tvFusedInfo;
    private LinearLayout layoutResults;

    // ── FL Feedback UI ──────────────────────────────────────────────────
    private CardView cardFeedback;
    private LinearLayout layoutFeedbackButtons;
    private LinearLayout layoutEmotionPicker;
    private Button btnCorrect;
    private Button btnWrong;
    private Button btnEmoAngry;
    private Button btnEmoHappy;
    private Button btnEmoNeutral;
    private Button btnEmoSad;
    private TextView tvFeedbackStatus;

    // ── FL Status UI ────────────────────────────────────────────────────
    private CardView cardFLStatus;
    private Button btnSyncModel;
    private TextView tvModelVersion;
    private TextView tvServerStatus;

    // ── Processors ──────────────────────────────────────────────────────
    private TextProcessor textProcessor;
    private AudioProcessor audioProcessor;
    private FusionEngine fusionEngine;
    private FeedbackManager feedbackManager;

    // ── State ───────────────────────────────────────────────────────────
    private MediaRecorder mediaRecorder;
    private String audioFilePath;
    private boolean isRecording = false;

    // Track the last prediction text for feedback
    private String lastPredictedText = "";
    private int lastPredictedLabelIndex = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Verify assets
        try {
            String[] assets = getAssets().list("");
            Log.d("ASSETS", "Asset files: " + Arrays.toString(assets));
        } catch (IOException e) {
            Log.e("ASSETS", "Failed to list assets", e);
        }

        initUI();
        initProcessors();
        setupListeners();
        setupFeedbackListeners();
        setupFLStatusListeners();

        // Check for model updates on startup
        checkForModelUpdates();
        refreshFLStatus();
    }

    // ═══════════════════════════════════════════════════════════════════
    // UI INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════

    private void initUI() {
        // Existing UI
        etTextInput      = findViewById(R.id.etTextInput);
        btnAnalyzeText   = findViewById(R.id.btnAnalyzeText);
        btnStartRecording = findViewById(R.id.btnStartRecording);
        btnStopRecording  = findViewById(R.id.btnStopRecording);
        btnAnalyzeAudio   = findViewById(R.id.btnAnalyzeAudio);
        btnAnalyzeAll     = findViewById(R.id.btnAnalyzeAll);
        progressBar      = findViewById(R.id.progressBar);
        tvStatus         = findViewById(R.id.tvStatus);
        layoutResults    = findViewById(R.id.layoutResults);

        cardTextResult  = findViewById(R.id.cardTextResult);
        cardAudioResult = findViewById(R.id.cardAudioResult);
        cardFusedResult = findViewById(R.id.cardFusedResult);

        tvTextEmotion   = findViewById(R.id.tvTextEmotion);
        tvTextConfidence = findViewById(R.id.tvTextConfidence);
        tvAudioEmotion  = findViewById(R.id.tvAudioEmotion);
        tvAudioConfidence = findViewById(R.id.tvAudioConfidence);
        tvFusedEmotion  = findViewById(R.id.tvFusedEmotion);
        tvFusedInfo     = findViewById(R.id.tvFusedInfo);

        // FL Feedback UI
        cardFeedback         = findViewById(R.id.cardFeedback);
        layoutFeedbackButtons = findViewById(R.id.layoutFeedbackButtons);
        layoutEmotionPicker  = findViewById(R.id.layoutEmotionPicker);
        btnCorrect           = findViewById(R.id.btnCorrect);
        btnWrong             = findViewById(R.id.btnWrong);
        btnEmoAngry          = findViewById(R.id.btnEmoAngry);
        btnEmoHappy          = findViewById(R.id.btnEmoHappy);
        btnEmoNeutral        = findViewById(R.id.btnEmoNeutral);
        btnEmoSad            = findViewById(R.id.btnEmoSad);
        tvFeedbackStatus     = findViewById(R.id.tvFeedbackStatus);

        // FL Status UI
        cardFLStatus    = findViewById(R.id.cardFLStatus);
        btnSyncModel    = findViewById(R.id.btnSyncModel);
        tvModelVersion  = findViewById(R.id.tvModelVersion);
        tvServerStatus  = findViewById(R.id.tvServerStatus);

        // Initial state
        btnStopRecording.setEnabled(false);
        layoutResults.setVisibility(View.GONE);
        progressBar.setVisibility(View.GONE);

        audioFilePath = new File(getExternalFilesDir(null), "recorded_audio.wav").getAbsolutePath();
    }

    private void initProcessors() {
        try {
            MappedByteBuffer textModelBuffer = loadModelFile(this, "text_model.tflite");
            Interpreter textInterpreter = new Interpreter(textModelBuffer);
            textProcessor = new TextProcessor(this, textInterpreter);
            Log.d(TAG, "Text model loaded successfully.");
        } catch (Exception e) {
            Log.e(TAG, "Text model load failed", e);
            setStatus("⚠ Text model not loaded: " + e.getMessage());
        }

        try {
            MappedByteBuffer audioModelBuffer = loadModelFile(this, "audio_model.tflite");
            Interpreter audioInterpreter = new Interpreter(audioModelBuffer);
            audioProcessor = new AudioProcessor(this, audioInterpreter);
            Log.d(TAG, "Audio model loaded successfully.");
        } catch (Exception e) {
            Log.e(TAG, "Audio model load failed", e);
            setStatus("⚠ Audio model not loaded: " + e.getMessage());
        }

        fusionEngine = new FusionEngine();
        feedbackManager = new FeedbackManager(this);

        // If there's an updated model from a previous FL round, load it
        Interpreter updatedInterpreter = feedbackManager.loadUpdatedModelInterpreter();
        if (updatedInterpreter != null && textProcessor != null) {
            textProcessor.reloadInterpreter(updatedInterpreter);
            int version = feedbackManager.getCurrentModelVersion();
            Log.d(TAG, "Loaded FL-updated text model v" + version);
            setStatus("Using FL-updated model v" + version);
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String modelName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // ═══════════════════════════════════════════════════════════════════
    // LISTENERS
    // ═══════════════════════════════════════════════════════════════════

    private void setupListeners() {
        btnAnalyzeText.setOnClickListener(v -> analyzeTextOnly());
        btnStartRecording.setOnClickListener(v -> startRecording());
        btnStopRecording.setOnClickListener(v -> stopRecording());
        btnAnalyzeAudio.setOnClickListener(v -> analyzeAudioOnly());
        btnAnalyzeAll.setOnClickListener(v -> analyzeAll());
    }

    private void setupFeedbackListeners() {
        // ✅ Correct — the predicted emotion was right
        btnCorrect.setOnClickListener(v -> {
            if (lastPredictedLabelIndex < 0 || lastPredictedText.isEmpty()) {
                Toast.makeText(this, "No prediction to confirm", Toast.LENGTH_SHORT).show();
                return;
            }
            sendFeedback(lastPredictedLabelIndex);
        });

        // ❌ Wrong — show emotion picker
        btnWrong.setOnClickListener(v -> {
            layoutEmotionPicker.setVisibility(View.VISIBLE);
            layoutFeedbackButtons.setVisibility(View.GONE);
        });

        // Emotion picker buttons
        btnEmoAngry.setOnClickListener(v -> sendFeedback(0));   // angry
        btnEmoHappy.setOnClickListener(v -> sendFeedback(1));   // happy
        btnEmoNeutral.setOnClickListener(v -> sendFeedback(2)); // neutral
        btnEmoSad.setOnClickListener(v -> sendFeedback(3));     // sad
    }

    private void setupFLStatusListeners() {
        btnSyncModel.setOnClickListener(v -> {
            btnSyncModel.setEnabled(false);
            tvServerStatus.setText("Server: connecting...");
            syncModelWithServer();
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    // FEDERATED LEARNING — FEEDBACK
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Send user feedback (correct emotion label) to the FL server.
     * The text is tokenized using TextProcessor and sent along with the correct label.
     */
    private void sendFeedback(int correctLabelIndex) {
        if (textProcessor == null || lastPredictedText.isEmpty()) {
            Toast.makeText(this, "Cannot send feedback", Toast.LENGTH_SHORT).show();
            return;
        }

        String emotionName = EmotionLabels.LABELS[correctLabelIndex];
        tvFeedbackStatus.setVisibility(View.VISIBLE);
        tvFeedbackStatus.setText("📤 Sending feedback (" + emotionName + ")...");
        tvFeedbackStatus.setTextColor(0xFFAAAADD);

        feedbackManager.submitFeedback(textProcessor, lastPredictedText, correctLabelIndex,
                (success, message, totalSamples) -> runOnUiThread(() -> {
                    if (success) {
                        tvFeedbackStatus.setText("✅ Feedback sent! (" + emotionName
                                + ") — " + totalSamples + " samples collected on server");
                        tvFeedbackStatus.setTextColor(0xFF90EE90);

                        // Hide the feedback buttons after successful submission
                        layoutFeedbackButtons.setVisibility(View.GONE);
                        layoutEmotionPicker.setVisibility(View.GONE);

                        // Check if server retrained the model
                        checkForModelUpdates();
                    } else {
                        tvFeedbackStatus.setText("⚠ Feedback failed: " + message);
                        tvFeedbackStatus.setTextColor(0xFFFF6B6B);

                        // Re-show buttons so user can retry
                        layoutFeedbackButtons.setVisibility(View.VISIBLE);
                        layoutEmotionPicker.setVisibility(View.GONE);
                    }
                }));
    }

    /**
     * Show the feedback card after a text prediction is made.
     */
    private void showFeedbackCard(String text, int predictedLabelIndex) {
        lastPredictedText = text;
        lastPredictedLabelIndex = predictedLabelIndex;

        cardFeedback.setVisibility(View.VISIBLE);
        layoutFeedbackButtons.setVisibility(View.VISIBLE);
        layoutEmotionPicker.setVisibility(View.GONE);
        tvFeedbackStatus.setVisibility(View.GONE);
    }

    // ═══════════════════════════════════════════════════════════════════
    // FEDERATED LEARNING — MODEL UPDATES
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Full sync: trigger server training → poll for new model → download → hot-swap.
     * Called from the Sync button.
     */
    private void syncModelWithServer() {
        feedbackManager.syncWithServer(new FeedbackManager.ModelUpdateCallback() {
            @Override
            public void onStatusUpdate(String message) {
                runOnUiThread(() -> tvServerStatus.setText("⏳ " + message));
            }

            @Override
            public void onModelUpdated(int newVersion) {
                runOnUiThread(() -> {
                    Interpreter newInterpreter = feedbackManager.loadUpdatedModelInterpreter();
                    if (newInterpreter != null && textProcessor != null) {
                        textProcessor.reloadInterpreter(newInterpreter);
                        setStatus("🎉 Model updated to v" + newVersion + " from FL server!");
                        tvModelVersion.setText("Model: v" + newVersion + " (FL-updated)");
                        tvServerStatus.setText("Server: 🟢 model v" + newVersion + " applied!");
                        Toast.makeText(MainActivity.this,
                                "✅ FL Model v" + newVersion + " loaded!", Toast.LENGTH_LONG).show();
                    }
                    btnSyncModel.setEnabled(true);
                });
            }

            @Override
            public void onModelUpdateFailed(String error) {
                runOnUiThread(() -> {
                    Log.d(TAG, "Sync result: " + error);
                    int version = feedbackManager.getCurrentModelVersion();
                    tvModelVersion.setText("Model: v" + version
                            + (version == 0 ? " (bundled)" : " (FL-updated)"));
                    tvServerStatus.setText("Server: ℹ " + error);
                    btnSyncModel.setEnabled(true);
                });
            }
        });
    }

    /**
     * Passive check on startup — only download if a newer version exists, no training trigger.
     */
    private void checkForModelUpdates() {
        feedbackManager.checkAndApplyModelUpdate(new FeedbackManager.ModelUpdateCallback() {
            @Override
            public void onModelUpdated(int newVersion) {
                runOnUiThread(() -> {
                    Interpreter newInterpreter = feedbackManager.loadUpdatedModelInterpreter();
                    if (newInterpreter != null && textProcessor != null) {
                        textProcessor.reloadInterpreter(newInterpreter);
                        setStatus("🔄 Model updated to v" + newVersion + " from FL server!");
                        tvModelVersion.setText("Model: v" + newVersion + " (FL-updated)");
                    }
                    btnSyncModel.setEnabled(true);
                });
            }

            @Override
            public void onModelUpdateFailed(String error) {
                runOnUiThread(() -> {
                    int version = feedbackManager.getCurrentModelVersion();
                    tvModelVersion.setText("Model: v" + version
                            + (version == 0 ? " (bundled)" : " (FL-updated)"));
                });
            }
        });
    }

    /**
     * Refresh the FL status display (server connectivity, model version, feedback count).
     */
    private void refreshFLStatus() {
        int version = feedbackManager.getCurrentModelVersion();
        tvModelVersion.setText("Model: v" + version
                + (version == 0 ? " (bundled)" : " (FL-updated)"));

        feedbackManager.checkServerStatus(status -> runOnUiThread(() -> {
            if (status.online) {
                tvServerStatus.setText("Server: 🟢 online | v" + status.modelVersion
                        + " | " + status.feedbackCount + " feedback samples");
            } else {
                tvServerStatus.setText("Server: 🔴 offline — " + status.message);
            }
        }));
    }

    // ═══════════════════════════════════════════════════════════════════
    // ANALYSIS METHODS (existing, with feedback integration)
    // ═══════════════════════════════════════════════════════════════════

    private void analyzeTextOnly() {
        // Read input inside click
        String textInput = etTextInput.getText().toString().trim();
        if (textInput.isEmpty()) {
            Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show();
            return;
        }
        if (textProcessor == null) {
            setStatus("Text model not loaded.");
            return;
        }

        showProgress(true);
        setStatus("Analyzing text...");
        cardAudioResult.setVisibility(View.GONE);
        cardFusedResult.setVisibility(View.GONE);
        cardFeedback.setVisibility(View.GONE);

        new Thread(() -> {
            try {
                float[] probs = textProcessor.predict(textInput);
                int idx = argmax(probs);
                String label = EmotionLabels.LABELS[idx];
                float conf = probs[idx] * 100f;

                runOnUiThread(() -> {
                    tvTextEmotion.setText("Emotion: " + label);
                    tvTextConfidence.setText(String.format("Confidence: %.1f%%", conf));
                    cardTextResult.setVisibility(View.VISIBLE);
                    layoutResults.setVisibility(View.VISIBLE);
                    showProgress(false);
                    setStatus("Text analysis complete.");

                    // Show feedback card for FL
                    showFeedbackCard(textInput, idx);
                });
            } catch (Exception e) {
                Log.e(TAG, "Text prediction error", e);
                runOnUiThread(() -> {
                    showProgress(false);
                    setStatus("Text analysis failed: " + e.getMessage());
                });
            }
        }).start();
    }

    private void analyzeAudioOnly() {
        if (audioProcessor == null) {
            setStatus("Audio model not loaded.");
            return;
        }

        File audioFile = new File(audioFilePath);
        if (!audioFile.exists()) {
            setStatus("No audio recorded yet. Please record audio first.");
            return;
        }

        showProgress(true);
        setStatus("Analyzing audio...");
        cardTextResult.setVisibility(View.GONE);
        cardFusedResult.setVisibility(View.GONE);
        cardFeedback.setVisibility(View.GONE);

        new Thread(() -> {
            try {
                float[] probs = audioProcessor.predict(audioFilePath);
                int idx = argmax(probs);
                String label = EmotionLabels.LABELS[idx];
                float conf = probs[idx] * 100f;

                runOnUiThread(() -> {
                    tvAudioEmotion.setText("Emotion: " + label);
                    tvAudioConfidence.setText(String.format("Confidence: %.1f%%", conf));
                    cardAudioResult.setVisibility(View.VISIBLE);
                    layoutResults.setVisibility(View.VISIBLE);
                    showProgress(false);
                    setStatus("Audio analysis complete.");
                });
            } catch (Exception e) {
                Log.e(TAG, "Audio prediction error", e);
                runOnUiThread(() -> {
                    showProgress(false);
                    setStatus("Audio analysis failed: " + e.getMessage());
                });
            }
        }).start();
    }

    private void analyzeAll() {
        String textInput = etTextInput.getText().toString().trim();
        File audioFile = new File(audioFilePath);
        
        boolean hasText = !textInput.isEmpty() && textProcessor != null;
        boolean hasAudio = audioFile.exists() && audioProcessor != null;

        if (!hasText && !hasAudio) {
            setStatus("⚠ Provide text or record audio for analysis.");
            return;
        }

        showProgress(true);
        setStatus("Running multimodal analysis...");
        cardFeedback.setVisibility(View.GONE);

        new Thread(() -> {
            float[] textProbs = null;
            float[] audioProbs = null;

            if (hasText) {
                try {
                    textProbs = textProcessor.predict(textInput);
                } catch (Exception e) {
                    Log.e(TAG, "Text prediction error in analyzeAll", e);
                }
            }

            if (hasAudio) {
                try {
                    audioProbs = audioProcessor.predict(audioFilePath);
                } catch (Exception e) {
                    Log.e(TAG, "Audio prediction error in analyzeAll", e);
                }
            }

            FusionEngine.FusionResult fused = fusionEngine.fuse(textProbs, audioProbs);

            final float[] fTextProbs = textProbs;
            final float[] fAudioProbs = audioProbs;

            runOnUiThread(() -> {
                showProgress(false);
                if (fused != null) {
                    layoutResults.setVisibility(View.VISIBLE);
                    if (fTextProbs != null) {
                        int idx = argmax(fTextProbs);
                        tvTextEmotion.setText("Emotion: " + EmotionLabels.LABELS[idx]);
                        tvTextConfidence.setText(String.format("Confidence: %.1f%%", fTextProbs[idx] * 100f));
                        cardTextResult.setVisibility(View.VISIBLE);
                    } else {
                        cardTextResult.setVisibility(View.GONE);
                    }

                    if (fAudioProbs != null) {
                        int idx = argmax(fAudioProbs);
                        tvAudioEmotion.setText("Emotion: " + EmotionLabels.LABELS[idx]);
                        tvAudioConfidence.setText(String.format("Confidence: %.1f%%", fAudioProbs[idx] * 100f));
                        cardAudioResult.setVisibility(View.VISIBLE);
                    } else {
                        cardAudioResult.setVisibility(View.GONE);
                    }

                    tvFusedEmotion.setText("Final Emotion: " + fused.emotion);
                    tvFusedInfo.setText(String.format(
                            "Confidence: %.1f%%\nText weight: %.1f%%  |  Audio weight: %.1f%%",
                            fused.confidence * 100f,
                            fused.textWeight * 100f,
                            fused.audioWeight * 100f));
                    cardFusedResult.setVisibility(View.VISIBLE);
                    setStatus("✅ Analysis complete.");

                    // Show feedback card if text was used
                    if (hasText && fTextProbs != null) {
                        int fusedIdx = 0;
                        for (int i = 0; i < EmotionLabels.LABELS.length; i++) {
                            if (EmotionLabels.LABELS[i].equals(fused.emotion)) {
                                fusedIdx = i;
                                break;
                            }
                        }
                        showFeedbackCard(textInput, fusedIdx);
                    }
                } else {
                    setStatus("⚠ Prediction failed.");
                }
            });
        }).start();
    }

    // ═══════════════════════════════════════════════════════════════════
    // RECORDING
    // ═══════════════════════════════════════════════════════════════════

    private void startRecording() {
        if (!checkAudioPermission()) return;

        try {
            mediaRecorder = new MediaRecorder();
            mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            mediaRecorder.setAudioSamplingRate(22050);
            mediaRecorder.setAudioEncodingBitRate(128000);
            mediaRecorder.setOutputFile(audioFilePath);
            mediaRecorder.prepare();
            mediaRecorder.start();

            isRecording = true;
            btnStartRecording.setEnabled(false);
            btnStopRecording.setEnabled(true);
            setStatus("🎙 Recording...");
            Log.d(TAG, "Recording started: " + audioFilePath);

            new android.os.Handler().postDelayed(() -> {
                if (isRecording) stopRecording();
            }, MAX_RECORDING_SECONDS * 1000L);

        } catch (IOException e) {
            Log.e(TAG, "MediaRecorder failed", e);
            setStatus("Recording failed: " + e.getMessage());
        }
    }

    private void stopRecording() {
        if (!isRecording) return;
        try {
            mediaRecorder.stop();
            mediaRecorder.release();
        } catch (Exception e) {
            Log.e(TAG, "MediaRecorder stop error", e);
        }
        mediaRecorder = null;
        isRecording = false;
        btnStartRecording.setEnabled(true);
        btnStopRecording.setEnabled(false);
        setStatus("Recording saved. Path: " + audioFilePath);
        Log.d(TAG, "Recording saved: " + audioFilePath);
        
        if (new File(audioFilePath).exists()) {
            Log.d(TAG, "Verified audio file exists: " + audioFilePath);
        } else {
            Log.e(TAG, "Audio file NOT found after stop!");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // PERMISSIONS
    // ═══════════════════════════════════════════════════════════════════

    private boolean checkAudioPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO, Manifest.permission.READ_EXTERNAL_STORAGE},
                    PERMISSION_REQUEST_CODE);
            return false;
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startRecording();
            } else {
                Toast.makeText(this, "Permissions required", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // HELPERS
    // ═══════════════════════════════════════════════════════════════════

    private void showProgress(boolean show) {
        progressBar.setVisibility(show ? View.VISIBLE : View.GONE);
        btnAnalyzeText.setEnabled(!show);
        btnAnalyzeAudio.setEnabled(!show);
        btnAnalyzeAll.setEnabled(!show);
    }

    private void setStatus(String msg) {
        tvStatus.setText(msg);
    }

    private static int argmax(float[] arr) {
        if (arr == null) return 0;
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (textProcessor != null) textProcessor.close();
        if (audioProcessor != null) audioProcessor.close();
    }
}
