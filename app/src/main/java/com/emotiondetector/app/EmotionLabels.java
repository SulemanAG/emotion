package com.emotiondetector.app;

/**
 * Shared emotion class labels. Must match the order used during model training.
 * Index 0=angry, 1=happy, 2=neutral, 3=sad
 */
public class EmotionLabels {
    public static final String[] LABELS = {"angry", "happy", "neutral", "sad"};
    public static final int NUM_CLASSES = LABELS.length;

    private EmotionLabels() {} // utility class
}
