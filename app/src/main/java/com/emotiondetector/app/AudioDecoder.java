package com.emotiondetector.app;

import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.util.Log;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * AudioDecoder
 *
 * Uses Android's MediaExtractor + MediaCodec to decode any supported audio
 * format (3GPP, AAC, MP4, WAV, OGG …) into 16-bit PCM samples, then
 * resamples to the target sample rate and converts to float[-1,1].
 *
 * This replaces third-party native libs (e.g. JLibrosa) for maximum
 * compatibility across devices without extra NDK dependencies.
 */
public class AudioDecoder {

    private static final String TAG = "AudioDecoder";

    /**
     * Decode an audio file and return a float PCM array resampled to targetSr.
     *
     * @param filePath   absolute path to the audio file
     * @param targetSr   desired sample rate (e.g. 22050)
     * @return float[] samples in range [-1, 1]
     */
    public static float[] decodeToPcm(String filePath, int targetSr) throws IOException {
        MediaExtractor extractor = new MediaExtractor();
        extractor.setDataSource(filePath);

        // Find first audio track
        MediaFormat format = null;
        int audioTrack = -1;
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            MediaFormat fmt = extractor.getTrackFormat(i);
            String mime = fmt.getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("audio/")) {
                format = fmt;
                audioTrack = i;
                break;
            }
        }

        if (audioTrack == -1 || format == null) {
            extractor.release();
            throw new IOException("No audio track found in: " + filePath);
        }

        extractor.selectTrack(audioTrack);

        int sourceSr = format.getInteger(MediaFormat.KEY_SAMPLE_RATE);
        int channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
        String mime   = format.getString(MediaFormat.KEY_MIME);

        Log.d(TAG, "Decoding: mime=" + mime + " sr=" + sourceSr + " ch=" + channels);

        MediaCodec codec = MediaCodec.createDecoderByType(mime);
        codec.configure(format, null, null, 0);
        codec.start();

        List<Short> pcmSamples = new ArrayList<>();
        ByteBuffer[] inputBuffers  = codec.getInputBuffers();
        ByteBuffer[] outputBuffers = codec.getOutputBuffers();

        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        boolean sawInputEOS  = false;
        boolean sawOutputEOS = false;

        while (!sawOutputEOS) {
            // Feed input
            if (!sawInputEOS) {
                int inputIdx = codec.dequeueInputBuffer(10_000);
                if (inputIdx >= 0) {
                    ByteBuffer buf = inputBuffers[inputIdx];
                    buf.clear();
                    int sampleSize = extractor.readSampleData(buf, 0);
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inputIdx, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                        sawInputEOS = true;
                    } else {
                        long presentationTimeUs = extractor.getSampleTime();
                        codec.queueInputBuffer(inputIdx, 0, sampleSize, presentationTimeUs, 0);
                        extractor.advance();
                    }
                }
            }

            // Drain output
            int outputIdx = codec.dequeueOutputBuffer(info, 10_000);
            if (outputIdx == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED) {
                outputBuffers = codec.getOutputBuffers();
            } else if (outputIdx >= 0) {
                ByteBuffer outBuf = outputBuffers[outputIdx];
                outBuf.order(ByteOrder.LITTLE_ENDIAN);
                ShortBuffer shortBuf = outBuf.asShortBuffer();

                int numSamples = info.size / 2;
                for (int i = 0; i < numSamples; i += channels) {
                    // Mix down to mono by averaging channels
                    int sum = 0;
                    for (int ch = 0; ch < channels; ch++) {
                        if (i + ch < numSamples) sum += shortBuf.get(i + ch);
                    }
                    pcmSamples.add((short) (sum / channels));
                }

                codec.releaseOutputBuffer(outputIdx, false);
                if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    sawOutputEOS = true;
                }
            }
        }

        codec.stop();
        codec.release();
        extractor.release();

        // Resample if necessary
        short[] raw = new short[pcmSamples.size()];
        for (int i = 0; i < raw.length; i++) raw[i] = pcmSamples.get(i);

        float[] resampled = (sourceSr != targetSr)
                ? resample(raw, sourceSr, targetSr)
                : shortsToFloat(raw);

        Log.d(TAG, "Decoded " + raw.length + " samples → resampled to " + resampled.length);
        return resampled;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /** Linear resampling (adequate for feature extraction) */
    private static float[] resample(short[] src, int srcRate, int dstRate) {
        double ratio = (double) dstRate / srcRate;
        int outLen = (int) (src.length * ratio);
        float[] out = new float[outLen];
        for (int i = 0; i < outLen; i++) {
            double srcIdx = i / ratio;
            int lo = (int) srcIdx;
            int hi = Math.min(lo + 1, src.length - 1);
            double frac = srcIdx - lo;
            out[i] = (float) ((src[lo] * (1 - frac) + src[hi] * frac) / 32768.0);
        }
        return out;
    }

    private static float[] shortsToFloat(short[] src) {
        float[] out = new float[src.length];
        for (int i = 0; i < src.length; i++) out[i] = src[i] / 32768.0f;
        return out;
    }
}
