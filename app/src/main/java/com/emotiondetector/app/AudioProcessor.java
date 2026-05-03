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
        // CONV_2D layers require 4D input [batch, height, width, channels].
        // If the model reports 3D shape [1, 40, 1], we reshape to 4D [1, 1, 40, 1]
        // because the model internally uses CONV_2D.
        
        Object input;
        float[] means = calculateMean(mfcc);

        if (inputShape.length == 4 && inputShape[2] == N_MFCC) {
            // Shape [1, MAX_FRAMES, 40, 1] — sequence of MFCC frames
            int maxFrames = inputShape[1];
            float[][][][] audioInput = new float[1][maxFrames][N_MFCC][1];
            float[][] transposed = transpose(mfcc); // [frames][40]
            for (int f = 0; f < Math.min(maxFrames, transposed.length); f++) {
                for (int c = 0; c < N_MFCC; c++) {
                    audioInput[0][f][c][0] = transposed[f][c];
                }
            }
            input = audioInput;
            Log.d(TAG, "Using 4D sequence input: [1, " + maxFrames + ", " + N_MFCC + ", 1]");

        } else if (inputShape.length == 4 && inputShape[1] == N_MFCC) {
            // Shape [1, 40, X, 1] — transposed layout
            int dim2 = inputShape[2];
            float[][][][] audioInput = new float[1][N_MFCC][dim2][1];
            for (int i = 0; i < N_MFCC; i++) {
                audioInput[0][i][0][0] = means[i];
            }
            input = audioInput;
            Log.d(TAG, "Using 4D transposed input: [1, " + N_MFCC + ", " + dim2 + ", 1]");

        } else if (inputShape.length == 3 && inputShape[1] == N_MFCC && inputShape[2] == 1) {
            // Shape [1, 40, 1] — but model has CONV_2D internally
            // Try 4D [1, 1, 40, 1] to satisfy CONV_2D requirement
            float[][][][] audioInput4D = new float[1][1][N_MFCC][1];
            for (int i = 0; i < N_MFCC; i++) {
                audioInput4D[0][0][i][0] = means[i];
            }
            input = audioInput4D;
            Log.d(TAG, "Reshaping 3D→4D input: [1, 1, " + N_MFCC + ", 1] for CONV_2D compatibility");

        } else {
            // Generic fallback: create array matching exact inputShape dimensions
            Log.w(TAG, "Unexpected input shape: " + Arrays.toString(inputShape) + ". Using dynamic fallback.");
            if (inputShape.length == 2) {
                float[][] fallback = new float[1][inputShape[1]];
                for (int i = 0; i < Math.min(inputShape[1], N_MFCC); i++) {
                    fallback[0][i] = means[i];
                }
                input = fallback;
            } else {
                // Last resort: 4D with mean MFCCs
                float[][][][] fallback4D = new float[1][1][N_MFCC][1];
                for (int i = 0; i < N_MFCC; i++) {
                    fallback4D[0][0][i][0] = means[i];
                }
                input = fallback4D;
            }
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
