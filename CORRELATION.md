# Correlation Metrics Explanation

This document explains the correlation metrics computed by the piezo capture software to analyze the relationship between two vibration signals (CH1 and CH2).

## Overview

The correlation analysis compares the **envelopes** of two piezo sensor signals to determine if they are responding to the same vibration event. The envelope is computed using the Hilbert transform, which extracts the amplitude modulation of the signal.

## Preprocessing

Before computing metrics, the signals undergo preprocessing:

1. **Envelope Extraction**: The Hilbert transform computes the analytic signal, and its absolute value gives the envelope (amplitude over time)
2. **Smoothing**: A moving average filter smooths the envelope to reduce noise
3. **Downsampling**: For computational efficiency, signals longer than 10,000 samples are downsampled
4. **Normalization**: Envelopes are normalized by their peak value for shape comparison

---

## Metrics

### 1. Pearson Envelope Correlation (`pearson_envelope`)

**What it measures**: Linear correlation between the two envelope signals.

**Formula**:
```
r = Σ[(env1 - μ1)(env2 - μ2)] / [σ1 × σ2 × n]
```

Where μ is the mean and σ is the standard deviation.

**Range**: -1 to +1
- +1 = Perfect positive correlation (envelopes move together)
- 0 = No linear relationship
- -1 = Perfect negative correlation (envelopes move opposite)

**Interpretation**: High values (> 0.7) indicate the envelopes have similar shapes and timing. This is the primary indicator of correlated vibration response.

---

### 2. Cosine Similarity (`cosine_similarity`)

**What it measures**: Angular similarity between normalized envelope vectors.

**Formula**:
```
cos_sim = (env1_norm · env2_norm) / (||env1_norm|| × ||env2_norm||)
```

**Range**: -1 to +1 (typically 0 to 1 for positive envelopes)

**Interpretation**: Measures shape similarity independent of amplitude. Two envelopes with identical shapes but different amplitudes will have cosine similarity = 1.

**Difference from Pearson**: Cosine similarity treats signals as vectors in high-dimensional space and measures the angle between them. Pearson correlation also accounts for the mean offset.

---

### 3. Normalized Cross-Correlation Peak (`ncc_peak`)

**What it measures**: Maximum similarity when one signal is shifted relative to the other.

**Computation**:
1. Compute cross-correlation using FFT: `IFFT(FFT(env1) × conj(FFT(env2)))`
2. Find the maximum value

**Range**: Unbounded positive value (higher = more similar)

**Interpretation**: A high NCC peak indicates the signals have similar shapes, even if they are time-shifted. Useful for detecting delayed responses.

---

### 4. NCC Lag (`ncc_lag_samples`)

**What it measures**: Time offset (in samples) where maximum correlation occurs.

**Computation**: The index of the NCC peak, converted to a signed offset from zero-lag.

**Range**: Negative to positive (samples)
- Negative = CH1 leads CH2
- Positive = CH2 leads CH1
- Zero = Signals are aligned

**Interpretation**: Indicates propagation delay between sensors. For piezo sensors on the same structure, this should be small (near zero) for correlated events.

---

### 5. Peak Lag (`peak_lag_samples`)

**What it measures**: Time difference between envelope peaks.

**Formula**:
```
peak_lag = argmax(env1) - argmax(env2)
```

**Range**: Negative to positive (samples)

**Interpretation**: Simple measure of timing alignment. If both sensors respond to the same event, their peaks should occur at similar times. Large values indicate different events or significant propagation delay.

---

### 6. Amplitude Ratio (`amplitude_ratio`)

**What it measures**: Ratio of peak amplitudes between the two envelopes.

**Formula**:
```
ratio = min(peak1, peak2) / max(peak1, peak2)
```

**Range**: 0 to 1
- 1 = Equal peak amplitudes
- 0 = One signal has no amplitude

**Interpretation**: For sensors at similar distances from the vibration source, we expect similar amplitudes (ratio near 1). Unequal amplitudes may indicate different sensor sensitivities, distances, or uncorrelated events.

---

## Composite Score

### Correlation Score (`correlation_score`)

**What it measures**: Weighted combination of individual metrics for overall correlation assessment.

**Formula**:
```
score = 0.35 × pearson_env
      + 0.30 × cosine_sim
      + 0.20 × amplitude_ratio
      + 0.15 × (1 - min(|peak_lag|, 100) / 100)
```

**Weights explained**:
- **Pearson (35%)**: Primary shape/timing correlation
- **Cosine (30%)**: Shape similarity
- **Amplitude (20%)**: Response magnitude similarity
- **Timing (15%)**: Peak alignment penalty (penalizes lags > 100 samples)

**Range**: Approximately 0 to 1

---

## Classification

Based on the correlation score:

| Score Range | Classification | Meaning |
|-------------|----------------|---------|
| > 0.7 | `CORRELATED` | Signals are likely from the same event |
| 0.4 - 0.7 | `WEAKLY_CORRELATED` | Some similarity, possibly related |
| < 0.4 | `UNCORRELATED` | Signals appear independent |

---

## Spectral Centroid

In addition to time-domain correlation, the software displays the **spectral centroid** for each channel.

**What it measures**: The "center of mass" of the frequency spectrum.

**Formula**:
```
centroid = Σ(frequency × magnitude) / Σ(magnitude)
```

**Units**: Hz

**Interpretation**: Indicates the dominant frequency content. Similar centroids suggest the sensors are responding to similar frequency vibrations. Different centroids may indicate different vibration sources or resonance characteristics.

---

## Practical Usage

### Correlated Signals (Good)
- Pearson > 0.7
- Cosine similarity > 0.8
- Amplitude ratio > 0.5
- Peak lag < 50 samples
- Similar spectral centroids

### Uncorrelated Signals
- Low Pearson (< 0.3)
- Large peak lag (> 100 samples)
- Very different amplitude ratios
- Different spectral centroids

### Troubleshooting
- **High Pearson but large lag**: Check sensor placement or propagation effects
- **Similar shapes but low amplitude ratio**: Check sensor sensitivity calibration
- **Good correlation but different centroids**: May indicate frequency-dependent attenuation
