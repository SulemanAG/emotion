package com.emotiondetector.app;

import android.content.Context;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Map;

/**
 * TextProcessor handles the tokenization and inference for the text model.
 */
public class TextProcessor {
    private static final String TAG = "TextProcessor";
    private static final int SEQ_LEN = 100;
    private final Interpreter interpreter;
    private Map<String, Integer> wordIndex;

    public TextProcessor(Context context, Interpreter interpreter) {
        this.interpreter = interpreter;
        loadTokenizer(context);
    }

    private void loadTokenizer(Context context) {
        try {
            InputStream is = context.getAssets().open("tokenizer.json");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            is.close();

            Gson gson = new Gson();
            JsonObject jsonObject = gson.fromJson(sb.toString(), JsonObject.class);
            JsonObject wordIndexJson = jsonObject.getAsJsonObject("word_index");

            Type type = new TypeToken<HashMap<String, Integer>>() {}.getType();
            wordIndex = gson.fromJson(wordIndexJson, type);
            
            // Ensure all keys are lowercase as per instructions
            Map<String, Integer> lowerCaseMap = new HashMap<>();
            for (Map.Entry<String, Integer> entry : wordIndex.entrySet()) {
                lowerCaseMap.put(entry.getKey().toLowerCase(), entry.getValue());
            }
            wordIndex = lowerCaseMap;

            Log.d(TAG, "Tokenizer loaded with " + wordIndex.size() + " words.");
        } catch (IOException e) {
            Log.e(TAG, "Failed to load tokenizer.json", e);
        }
    }

    public float[] predict(String text) {
        if (interpreter == null || wordIndex == null) {
            Log.e(TAG, "Interpreter or Tokenizer not initialized");
            return new float[EmotionLabels.NUM_CLASSES];
        }

        // Tokenize and pre-pad
        int[] sequence = tokenize(text);
        int[] padded = padSequence(sequence, SEQ_LEN);
        
        Log.d(TAG, "Input sequence length: " + padded.length);
        
        // Convert to float[1][100]
        float[][] input = new float[1][SEQ_LEN];
        for (int i = 0; i < SEQ_LEN; i++) {
            input[0][i] = (float) padded[i];
        }

        float[][] output = new float[1][EmotionLabels.NUM_CLASSES];
        interpreter.run(input, output);
        
        Log.d(TAG, "Text model raw output: " + java.util.Arrays.toString(output[0]));
        return output[0];
    }

    private int[] tokenize(String text) {
        String[] words = text.toLowerCase().trim().split("\\s+");
        int[] sequence = new int[words.length];
        for (int i = 0; i < words.length; i++) {
            String cleanWord = words[i].replaceAll("[^a-z0-9]", "");
            Integer idx = wordIndex.get(cleanWord);
            sequence[i] = (idx != null) ? idx : 0;
        }
        return sequence;
    }

    private int[] padSequence(int[] sequence, int length) {
        int[] padded = new int[length];
        if (sequence.length > length) {
            // Truncate (keep last 100)
            System.arraycopy(sequence, sequence.length - length, padded, 0, length);
        } else {
            // Pre-padding with zeros
            int start = length - sequence.length;
            System.arraycopy(sequence, 0, padded, start, sequence.length);
        }
        return padded;
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
        }
    }
}
