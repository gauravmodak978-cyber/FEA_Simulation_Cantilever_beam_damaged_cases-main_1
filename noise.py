# noise.py
# =============================================================
# Sensor / measurement noise for beam FEA acceleration output
# Adds Gaussian white noise at a specified SNR (dB)
# =============================================================

import numpy as np


def add_sensor_noise(node_accels, snr_db, seed=None):
    """
    Add Gaussian white noise to node acceleration data.

    Noise level is set so that the overall signal-to-noise ratio
    matches the requested SNR in decibels:

        SNR_dB = 20 * log10(RMS_signal / RMS_noise)

    Noise is generated independently for each node.

    Inputs:
        node_accels : (n_nodes, n_steps) clean acceleration array (in/s²)
        snr_db      : signal-to-noise ratio in dB (e.g. 20, 30, 40, 60)
        seed        : random seed for reproducibility (int or None)

    Output:
        noisy_accels : (n_nodes, n_steps) noise-corrupted acceleration array
        noise_array  : (n_nodes, n_steps) the noise that was added
    """
    rng = np.random.default_rng(seed)

    n_nodes, n_steps = node_accels.shape
    noise_array  = np.zeros_like(node_accels)
    noisy_accels = np.zeros_like(node_accels)

    for i in range(n_nodes):
        signal = node_accels[i, :]
        rms_signal = np.sqrt(np.mean(signal ** 2))

        if rms_signal < 1e-15:
            # Node has zero signal (e.g. fixed end) — no noise to add
            noisy_accels[i, :] = signal
            continue

        # Compute required noise RMS from SNR definition
        #   SNR_dB = 20 * log10(RMS_signal / RMS_noise)
        #   RMS_noise = RMS_signal / 10^(SNR_dB / 20)
        rms_noise = rms_signal / (10.0 ** (snr_db / 20.0))

        # Generate Gaussian white noise with the computed RMS
        noise = rng.normal(0.0, rms_noise, n_steps)

        noise_array[i, :]  = noise
        noisy_accels[i, :] = signal + noise

    return noisy_accels, noise_array
