package com.emotiondetector.app;

import android.content.Context;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * AudioProcessor handles the MFCC extraction and inference for the audio model.
 */
public class AudioProcessor {
    private static final String TAG = "AudioProcessor";
    
    // MFCC parameters (must match training)
    private static final int SAMPLE_RATE = 22050;
    private static final int N_MFCC = 40;
    private static final int N_FFT = 2048;
    private static final int HOP_LENGTH = 512;
    private static final int NUM_MEL_BINS = 128;

    private final Interpreter interpreter;
    private final int[] inputShape;

    public AudioProcessor(Context context, Interpreter interpreter) {
        this.interpreter = interpreter;
        this.inputShape = interpreter.getInputTensor(0).shape();
        Log.d(TAG, "Audio model initialized. Input shape: " + Arrays.toString(inputShape));
    }

    public float[] predict(String filePath) throws IOException {
        if (interpreter == null) {
            Log.e(TAG, "Interpreter not initialized");
            return new float[EmotionLabels.NUM_CLASSES];
        }

        File file = new File(filePath);
        if (!file.exists()) {
            throw new IOException("Audio file not found: " + filePath);
        }

        // 1. Decode Audio -> PCM
        float[] samples = AudioDecoder.decodeToPcm(filePath, SAMPLE_RATE);
        Log.d(TAG, "Decoded PCM samples: " + samples.length);

        // 2. Extract MFCCs [40][numFrames]
        float[][] mfcc = MfccExtractor.computeMfcc(samples, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, NUM_MEL_BINS);
        Log.d(TAG, "Extracted MFCCs: " + mfcc.length + "x" + (mfcc.length > 0 ? mfcc[0].length : 0));

        // 3. Prepare Input based on model shape
        // If the shape is [1, 40, 1], we likely need the mean of MFCCs over frames.
        // If the shape is [1, 87, 40, 1], we need the sequence.
        
        Object input;
        if (inputShape.length == 3 && inputShape[1] == N_MFCC && inputShape[2] == 1) {
            // Shape [1, 40, 1]
            float[][][] audioInput = new float[1][N_MFCC][1];
            float[] means = calculateMean(mfcc);
            for (int i = 0; i < N_MFCC; i++) {
                audioInput[0][i][0] = means[i];
            }
            input = audioInput;
        } else if (inputShape.length == 4 && inputShape[2] == N_MFCC) {
            // Shape [1, MAX_FRAMES, 40, 1]
            int maxFrames = inputShape[1];
            float[][][][] audioInput = new float[1][maxFrames][N_MFCC][1];
            float[][] transposed = transpose(mfcc); // [frames][40]
            for (int f = 0; f < Math.min(maxFrames, transposed.length); f++) {
                for (int c = 0; c < N_MFCC; c++) {
                    audioInput[0][f][c][0] = transposed[f][c];
                }
            }
            input = audioInput;
        } else {
            // Fallback: use first frame or whatever matches
            Log.w(TAG, "Unexpected input shape: " + Arrays.toString(inputShape) + ". Attempting fallback.");
            float[][] fallback = new float[1][N_MFCC]; // very simple fallback
            float[] means = calculateMean(mfcc);
            System.arraycopy(means, 0, fallback[0], 0, N_MFCC);
            input = fallback;
        }

        float[][] output = new float[1][EmotionLabels.NUM_CLASSES];
        interpreter.run(input, output);
        
        Log.d(TAG, "Audio model raw output: " + Arrays.toString(output[0]));
        return output[0];
    }

    private float[] calculateMean(float[][] mfcc) {
        int rows = mfcc.length;
        int cols = mfcc[0].length;
        float[] means = new float[rows];
        for (int i = 0; i < rows; i++) {
            float sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += mfcc[i][j];
            }
            means[i] = (cols > 0) ? sum / cols : 0;
        }
        return means;
    }

    private float[][] transpose(float[][] mat) {
        if (mat.length == 0) return new float[0][0];
        int rows = mat.length;
        int cols = mat[0].length;
        float[][] t = new float[cols][rows];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                t[c][r] = mat[r][c];
            }
        }
        return t;
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
        }
    }
}
