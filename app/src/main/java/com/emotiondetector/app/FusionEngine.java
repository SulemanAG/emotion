package com.emotiondetector.app;

import android.util.Log;

/**
 * FusionEngine
 *
 * Implements confidence-weighted late fusion of text and audio predictions.
 *
 * Formula:
 *   totalConf = audioConf + textConf
 *   fusedProbs[i] = (audioConf / totalConf) * audioProbs[i]
 *                 + (textConf  / totalConf) * textProbs[i]
 *   finalEmotion = argmax(fusedProbs)
 *
 * Falls back gracefully to whichever modality is available if the other is null.
 */
public class FusionEngine {

    private static final String TAG = "FusionEngine";

    public static class FusionResult {
        public final String  emotion;
        public final float   confidence;   // confidence in [0,1]
        public final float   textWeight;   // proportion contributed by text
        public final float   audioWeight;  // proportion contributed by audio
        public final float[] fusedProbs;

        FusionResult(String emotion, float confidence,
                     float textWeight, float audioWeight, float[] fusedProbs) {
            this.emotion    = emotion;
            this.confidence = confidence;
            this.textWeight = textWeight;
            this.audioWeight = audioWeight;
            this.fusedProbs = fusedProbs;
        }
    }

    /**
     * Fuse text and audio probability arrays.
     *
     * @param textProbs  float[NUM_CLASSES] or null
     * @param audioProbs float[NUM_CLASSES] or null
     * @return FusionResult or null if both inputs are null
     */
    public FusionResult fuse(float[] textProbs, float[] audioProbs) {
        boolean hasText  = textProbs  != null && textProbs.length  == EmotionLabels.NUM_CLASSES;
        boolean hasAudio = audioProbs != null && audioProbs.length == EmotionLabels.NUM_CLASSES;

        if (!hasText && !hasAudio) {
            Log.w(TAG, "Both text and audio probabilities are null – cannot fuse.");
            return null;
        }

        float textConf  = hasText  ? max(textProbs)  : 0f;
        float audioConf = hasAudio ? max(audioProbs) : 0f;
        float totalConf = textConf + audioConf;

        float tWeight, aWeight;
        if (totalConf < 1e-6f) {
            // Equal weights if both confidences are near zero
            tWeight = hasText  ? 0.5f : 0f;
            aWeight = hasAudio ? 0.5f : 0f;
        } else {
            tWeight = hasText  ? textConf  / totalConf : 0f;
            aWeight = hasAudio ? audioConf / totalConf : 0f;
        }

        Log.d(TAG, String.format("Fusion weights → text=%.2f  audio=%.2f", tWeight, aWeight));

        float[] fused = new float[EmotionLabels.NUM_CLASSES];
        for (int i = 0; i < EmotionLabels.NUM_CLASSES; i++) {
            fused[i] = (hasText  ? tWeight * textProbs[i]  : 0f)
                     + (hasAudio ? aWeight * audioProbs[i] : 0f);
        }

        int bestIdx = argmax(fused);
        String emotion = EmotionLabels.LABELS[bestIdx];
        Log.d(TAG, "Fused emotion: " + emotion + " (prob=" + fused[bestIdx] + ")");

        return new FusionResult(emotion, fused[bestIdx], tWeight, aWeight, fused);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static float max(float[] arr) {
        float m = Float.NEGATIVE_INFINITY;
        for (float v : arr) if (v > m) m = v;
        return m;
    }

    private static int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }
}
