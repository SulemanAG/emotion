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

        // 3. Prepare Input — always match the EXACT declared input shape.
        //    Let TFLite handle any internal reshaping for CONV_2D etc.
        
        float[] means = calculateMean(mfcc);
        Log.d(TAG, "Model input shape: " + Arrays.toString(inputShape) + " (ndim=" + inputShape.length + ")");

        Object input;
        if (inputShape.length == 4) {
            // 4D: [batch, dim1, dim2, channels]
            float[][][][] audioInput = new float[inputShape[0]][inputShape[1]][inputShape[2]][inputShape[3]];
            if (inputShape[2] == N_MFCC) {
                // [1, frames, 40, 1] — fill with MFCC frames
                float[][] transposed = transpose(mfcc); // [frames][40]
                for (int f = 0; f < Math.min(inputShape[1], transposed.length); f++) {
                    for (int c = 0; c < N_MFCC; c++) {
                        audioInput[0][f][c][0] = transposed[f][c];
                    }
                }
            } else if (inputShape[1] == N_MFCC) {
                // [1, 40, X, 1] — fill with mean MFCCs
                for (int i = 0; i < N_MFCC; i++) {
                    audioInput[0][i][0][0] = means[i];
                }
            }
            input = audioInput;
        } else if (inputShape.length == 3) {
            // 3D: [batch, dim1, dim2]
            float[][][] audioInput = new float[inputShape[0]][inputShape[1]][inputShape[2]];
            for (int i = 0; i < Math.min(inputShape[1], N_MFCC); i++) {
                audioInput[0][i][0] = means[i];
            }
            input = audioInput;
        } else if (inputShape.length == 2) {
            // 2D: [batch, features]
            float[][] audioInput = new float[inputShape[0]][inputShape[1]];
            for (int i = 0; i < Math.min(inputShape[1], N_MFCC); i++) {
                audioInput[0][i] = means[i];
            }
            input = audioInput;
        } else {
            Log.e(TAG, "Unsupported input shape: " + Arrays.toString(inputShape));
            return new float[EmotionLabels.NUM_CLASSES];
        }

        float[][] output = new float[1][EmotionLabels.NUM_CLASSES];

        try {
            interpreter.run(input, output);
        } catch (Exception e) {
            Log.e(TAG, "Audio inference failed with shape " + Arrays.toString(inputShape) + ": " + e.getMessage());
            throw new IOException("Audio model inference error: " + e.getMessage());
        }
        
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
