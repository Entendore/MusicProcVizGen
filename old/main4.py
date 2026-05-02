# -------------------- Spectral Analysis --------------------
def spectral_energy(buffer):
    spectrum = np.fft.rfft(buffer)
    freqs = np.fft.rfftfreq(len(buffer),1/sample_rate)
    mag = np.abs(spectrum)

    # Define bands
    low_band = (freqs >= 20) & (freqs < 250)
    mid_band = (freqs >= 250) & (freqs < 2000)
    high_band = (freqs >= 2000) & (freqs < 10000)

    # Compute normalized energy
    low_energy = np.mean(mag[low_band]) if np.any(low_band) else 0
    mid_energy = np.mean(mag[mid_band]) if np.any(mid_band) else 0
    high_energy = np.mean(mag[high_band]) if np.any(high_band) else 0

    return low_energy, mid_energy, high_energy

# -------------------- Apply Spectral Reactivity --------------------
def apply_spectral_reactivity(buffer, env, grains, active_events, active_motifs):
    low, mid, high = spectral_energy(buffer)
    # Scale to 0..1
    low = np.tanh(low*10)
    mid = np.tanh(mid*10)
    high = np.tanh(high*10)

    # Modulate grains
    for grain in grains:
        if grain.melodic:
            grain.amp *= 0.8 + 0.4*(1-mid)  # melodic grains swell if mid energy is low
        else:
            grain.amp *= 0.8 + 0.4*(1-low)  # non-melodic grains swell if low energy is low

    # Modulate events
    for event in active_events:
        event.amp *= 0.8 + 0.5*(1-high)  # shimmers/hits stronger if high band is quiet

    # Modulate motifs
    for motif in active_motifs:
        for i in range(len(motif.amps)):
            motif.amps[i] *= 0.8 + 0.4*(1-mid)  # motifs respond to mid-energy

    # Modulate filter cutoff & reverb
    env.filter_cutoff = 800 + 1500*high
    env.reverb_decay = 0.3 + 0.2*(1-low)
