# dsp.py
import numpy as np
from config import SAMPLE_RATE, DTYPE

class WavetableOscillator:
    """ Efficient Wavetable Synthesis with multiple shapes. """
    def __init__(self, table_size=2048, waveform='sine'):
        self.table_size = table_size
        self.phase = 0.0
        self.waveform = waveform
        
        # Generate Tables
        t = np.linspace(0, 1, table_size)
        if waveform == 'sine':
            self.table = np.sin(2 * np.pi * t)
        elif waveform == 'saw':
            self.table = 2 * t - 1
        elif waveform == 'tri':
            self.table = 2 * np.abs(2 * t - 1) - 1
        elif waveform == 'square':
            self.table = np.where(t > 0.5, 1, -1)
        else:
            self.table = np.sin(2 * np.pi * t)
            
        self.table = self.table.astype(DTYPE)

    def get_samples(self, freq, frames):
        if freq <= 0: return np.zeros(frames, dtype=DTYPE)
        
        phase_inc = (self.table_size * freq) / SAMPLE_RATE
        indices = (self.phase + phase_inc * np.arange(frames)) % self.table_size
        self.phase = (self.phase + phase_inc * frames) % self.table_size
        
        int_indices = indices.astype(int)
        frac = indices - int_indices
        
        idx0 = int_indices % self.table_size
        idx1 = (idx0 + 1) % self.table_size
        
        s0 = self.table[idx0]
        s1 = self.table[idx1]
        
        return s0 + (s1 - s0) * frac

class Voice:
    """ Synth Voice with selectable waveform. """
    def __init__(self, waveform='sine'):
        self.osc = WavetableOscillator(waveform=waveform)
        self.freq = 440.0
        self.velocity = 0.0
        self.env_state = 'off'
        self.env_level = 0.0
        self.attack = 0.01; self.decay = 0.1; self.sustain = 0.5; self.release = 0.2
        
    def note_on(self, freq, vel, attack=0.01, decay=0.1, sustain=0.5):
        self.freq = freq
        self.velocity = vel
        self.env_state = 'a'
        self.env_level = 0.0
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        
    def process(self, frames):
        if self.env_state == 'off': return np.zeros(frames, dtype=DTYPE)
        samples = self.osc.get_samples(self.freq, frames)
        
        frames_f = float(frames)
        if self.env_state == 'a':
            rate = 1.0 / (self.attack * SAMPLE_RATE)
            self.env_level = min(1.0, self.env_level + rate * frames_f)
            if self.env_level >= 1.0: self.env_state = 'd'
        elif self.env_state == 'd':
            rate = 1.0 / (self.decay * SAMPLE_RATE)
            self.env_level = max(self.sustain, self.env_level - rate * frames_f)
            if self.env_level <= self.sustain: self.env_state = 's'
        elif self.env_state == 's':
            pass
            
        return samples * self.env_level * self.velocity

class Filter:
    """ State Variable Filter (Low Pass) """
    def __init__(self):
        self.z = np.zeros(2, dtype=DTYPE)

    def process(self, buffer, cutoff):
        dt = 1.0 / SAMPLE_RATE
        rc = 1.0 / (2 * np.pi * max(20, cutoff))
        alpha = rc / (rc + dt)
        
        new_z = self.z
        for i in range(len(buffer)):
            new_z[0] = alpha * buffer[i, 0] + (1 - alpha) * new_z[0]
            new_z[1] = alpha * buffer[i, 1] + (1 - alpha) * new_z[1]
            buffer[i, 0] = new_z[0]
            buffer[i, 1] = new_z[1]
        self.z = new_z
        return buffer

class Limiter:
    """ Simple Soft Clipper / Limiter """
    def process(self, buffer):
        # Tanh distortion for soft clipping
        return np.tanh(buffer * 1.2)