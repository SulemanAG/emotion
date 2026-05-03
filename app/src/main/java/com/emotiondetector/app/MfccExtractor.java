package com.emotiondetector.app;

/**
 * MfccExtractor
 *
 * Pure-Java implementation of MFCC extraction matching librosa defaults:
 *   n_mfcc=40, n_fft=2048, hop_length=512, n_mels=128
 *   window=hann, center=True (zero-padded)
 *
 * Returns a float[n_mfcc][numFrames] matrix.
 *
 * References:
 *  - Mel filterbank: O'Shaughnessy (1987) / HTK Book
 *  - DCT-II for MFCC: librosa default
 */
public class MfccExtractor {

    /**
     * Compute MFCC feature matrix.
     *
     * @param signal    float PCM samples in [-1, 1]
     * @param sr        sample rate
     * @param nMfcc     number of MFCC coefficients
     * @param nFft      FFT size (e.g. 2048)
     * @param hopLength frame hop (e.g. 512)
     * @param nMels     number of mel filter bands (e.g. 128)
     * @return float[nMfcc][numFrames]
     */
    public static float[][] computeMfcc(float[] signal, int sr, int nMfcc,
                                        int nFft, int hopLength, int nMels) {
        // Centre-pad signal by nFft/2 on each side (librosa centre=True)
        float[] padded = centrePad(signal, nFft / 2);

        // Number of frames
        int numFrames = 1 + (padded.length - nFft) / hopLength;
        if (numFrames <= 0) numFrames = 1;

        // Pre-compute Hann window
        float[] window = hannWindow(nFft);

        // Mel filterbank: [nMels][nFft/2+1]
        float[][] melFilters = melFilterbank(sr, nFft, nMels);

        // DCT matrix: [nMfcc][nMels]
        float[][] dctMatrix = dctMatrix(nMfcc, nMels);

        float[][] mfcc = new float[nMfcc][numFrames];

        float[] frame    = new float[nFft];
        float[] windowed = new float[nFft];
        float[] powerSpec = new float[nFft / 2 + 1];

        for (int t = 0; t < numFrames; t++) {
            int start = t * hopLength;

            // Extract frame
            for (int i = 0; i < nFft; i++) {
                int idx = start + i;
                frame[i] = (idx < padded.length) ? padded[idx] : 0f;
            }

            // Apply window
            for (int i = 0; i < nFft; i++) windowed[i] = frame[i] * window[i];

            // FFT → power spectrum
            float[] fftReal = new float[nFft];
            float[] fftImag = new float[nFft];
            System.arraycopy(windowed, 0, fftReal, 0, nFft);
            fftInPlace(fftReal, fftImag);

            for (int k = 0; k <= nFft / 2; k++) {
                powerSpec[k] = fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k];
            }

            // Apply mel filterbank → log mel energies
            float[] melEnergies = new float[nMels];
            for (int m = 0; m < nMels; m++) {
                float energy = 0f;
                for (int k = 0; k <= nFft / 2; k++) {
                    energy += melFilters[m][k] * powerSpec[k];
                }
                melEnergies[m] = (float) Math.log(Math.max(energy, 1e-10));
            }

            // DCT-II  → MFCCs
            for (int c = 0; c < nMfcc; c++) {
                float val = 0f;
                for (int m = 0; m < nMels; m++) {
                    val += dctMatrix[c][m] * melEnergies[m];
                }
                mfcc[c][t] = val;
            }
        }

        return mfcc;
    }

    // -------------------------------------------------------------------------
    // Hann window
    // -------------------------------------------------------------------------

    private static float[] hannWindow(int n) {
        float[] w = new float[n];
        for (int i = 0; i < n; i++) {
            w[i] = (float) (0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1))));
        }
        return w;
    }

    // -------------------------------------------------------------------------
    // Centre-pad
    // -------------------------------------------------------------------------

    private static float[] centrePad(float[] sig, int pad) {
        float[] out = new float[sig.length + 2 * pad];
        // Reflect-pad (librosa default mode='reflect')
        for (int i = 0; i < pad; i++) {
            out[pad - 1 - i] = sig[Math.min(i + 1, sig.length - 1)];
        }
        System.arraycopy(sig, 0, out, pad, sig.length);
        for (int i = 0; i < pad; i++) {
            out[pad + sig.length + i] = sig[Math.max(sig.length - 2 - i, 0)];
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Mel filterbank
    // -------------------------------------------------------------------------

    private static float[][] melFilterbank(int sr, int nFft, int nMels) {
        int nFreqs = nFft / 2 + 1;
        float fMin = 0f;
        float fMax = sr / 2f;

        float melMin = hzToMel(fMin);
        float melMax = hzToMel(fMax);

        // nMels+2 equally spaced mel points
        float[] melPoints = new float[nMels + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = melMin + i * (melMax - melMin) / (nMels + 1);
        }

        // Convert to Hz then to FFT bin indices
        float[] hzPoints = new float[nMels + 2];
        for (int i = 0; i < hzPoints.length; i++) {
            hzPoints[i] = melToHz(melPoints[i]);
        }

        int[] binPoints = new int[nMels + 2];
        for (int i = 0; i < binPoints.length; i++) {
            binPoints[i] = (int) Math.floor((nFft + 1) * hzPoints[i] / sr);
        }

        float[][] filters = new float[nMels][nFreqs];
        for (int m = 1; m <= nMels; m++) {
            int fM  = binPoints[m - 1];
            int fC  = binPoints[m];
            int fP  = binPoints[m + 1];

            for (int k = fM; k < fC && k < nFreqs; k++) {
                filters[m - 1][k] = (float) (k - fM) / (fC - fM);
            }
            for (int k = fC; k < fP && k < nFreqs; k++) {
                filters[m - 1][k] = (float) (fP - k) / (fP - fC);
            }
        }
        return filters;
    }

    private static float hzToMel(float hz) {
        return 2595f * (float) Math.log10(1 + hz / 700f);
    }

    private static float melToHz(float mel) {
        return 700f * ((float) Math.pow(10, mel / 2595f) - 1);
    }

    // -------------------------------------------------------------------------
    // DCT-II matrix  (orthonormal)
    // -------------------------------------------------------------------------

    private static float[][] dctMatrix(int nMfcc, int nMels) {
        float[][] D = new float[nMfcc][nMels];
        for (int i = 0; i < nMfcc; i++) {
            for (int j = 0; j < nMels; j++) {
                D[i][j] = (float) Math.cos(Math.PI * i * (2 * j + 1) / (2.0 * nMels));
            }
            // Orthonormal scale
            D[i][0] *= (float) (1.0 / Math.sqrt(nMels));
            if (i > 0) D[i][0] *= (float) Math.sqrt(2.0 / nMels);
        }
        // Re-apply proper scale to all columns
        for (int i = 0; i < nMfcc; i++) {
            float scale = (i == 0) ? (float) (1.0 / Math.sqrt(nMels))
                                    : (float) Math.sqrt(2.0 / nMels);
            for (int j = 0; j < nMels; j++) {
                D[i][j] = scale * (float) Math.cos(Math.PI * i * (2 * j + 1) / (2.0 * nMels));
            }
        }
        return D;
    }

    // -------------------------------------------------------------------------
    // In-place Cooley–Tukey FFT  (radix-2, power-of-2 nFft required)
    // -------------------------------------------------------------------------

    private static void fftInPlace(float[] re, float[] im) {
        int n = re.length;
        // Bit-reversal permutation
        int j = 0;
        for (int i = 1; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                float t = re[i]; re[i] = re[j]; re[j] = t;
                t = im[i]; im[i] = im[j]; im[j] = t;
            }
        }
        // Butterfly computations
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2 * Math.PI / len;
            float wRe = (float) Math.cos(ang);
            float wIm = (float) Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                float curRe = 1f, curIm = 0f;
                for (int k = 0; k < len / 2; k++) {
                    float uRe = re[i + k];
                    float uIm = im[i + k];
                    float vRe = re[i + k + len/2] * curRe - im[i + k + len/2] * curIm;
                    float vIm = re[i + k + len/2] * curIm + im[i + k + len/2] * curRe;
                    re[i + k]         = uRe + vRe;
                    im[i + k]         = uIm + vIm;
                    re[i + k + len/2] = uRe - vRe;
                    im[i + k + len/2] = uIm - vIm;
                    float newCurRe = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = newCurRe;
                }
            }
        }
    }
}
