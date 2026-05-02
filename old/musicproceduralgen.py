import sys
import numpy as np
import sounddevice as sd
import random
import time
import queue
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QComboBox, QPushButton, 
                               QGroupBox, QGridLayout, QCheckBox, QFrame)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QBrush, QFont

# -------------------- CONFIGURATION --------------------
SAMPLE_RATE = 44100
BUFFER_SIZE = 512
DTYPE = 'float32'

# -------------------- MUSIC THEORY ENGINE --------------------

class MusicTheory:
    """
    Centralized music theory definitions.
    Includes Western, Eastern, and World music scales.
    """
    # Semitone intervals from root
    SCALES = {
        # Western
        "Major":            [0, 2, 4, 5, 7, 9, 11],
        "Natural Minor":    [0, 2, 3, 5, 7, 8, 10],
        "Harmonic Minor":   [0, 2, 3, 5, 7, 8, 11],
        "Dorian":           [0, 2, 3, 5, 7, 9, 10],
        "Phrygian":         [0, 1, 3, 5, 7, 8, 10],
        "Lydian":           [0, 2, 4, 6, 7, 9, 11],
        "Mixolydian":       [0, 2, 4, 5, 7, 9, 10],
        "Blues":            [0, 3, 5, 6, 7, 10],
        "Pentatonic":       [0, 2, 4, 7, 9],
        
        # Eastern / World / Exotic
        "Hijaz (Arabic)":   [0, 1, 4, 5, 7, 8, 10],    # Phrygian Dominant
        "Double Harmonic":  [0, 1, 4, 5, 7, 8, 11],    # Byzantine / Arabic Major
        "Hungarian Minor":  [0, 2, 3, 6, 7, 8, 11],    # Exotic gypsy feel
        "Japanese (In-Sen)":[0, 1, 5, 7, 8],           # Pentatonic mood
        "Raga Bhairavi":    [0, 1, 4, 5, 7, 8, 10],    # Similar to Phrygian Dominant
        "Maqam Kurd":       [0, 2, 3, 5, 7, 8, 10],    # Similar to Phrygian
        "Chromatic":        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }
    
    ROOTS = {
        "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
        "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
        "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88
    }

    @staticmethod
    def get_scale_freqs(root_note, scale_name):
        root = MusicTheory.ROOTS[root_note]
        intervals = MusicTheory.SCALES.get(scale_name, MusicTheory.SCALES["Major"])
        return [root * (2**(i/12.0)) for i in intervals]

    @staticmethod
    def merge_scales(scale_a, scale_b, blend_factor):
        """
        Blends two scales. 
        blend_factor 0.0 = 100% Scale A
        blend_factor 1.0 = 100% Scale B
        blend_factor 0.5 = Union of unique notes from both, sorted.
        """
        if not scale_a: return scale_b
        if not scale_b: return scale_a
        
        if blend_factor < 0.01: return scale_a
        if blend_factor > 0.99: return scale_b
        
        # Simple union of intervals for a "Fusion" scale
        merged = list(set(scale_a + scale_b))
        merged.sort()
        return merged

# -------------------- DSP CORE --------------------

class WavetableOscillator:
    """ Efficient Wavetable Synthesis with variable waveforms. """
    def __init__(self, table_size=2048):
        self.table_size = table_size
        # Sine wave
        self.table = np.sin(2 * np.pi * np.arange(table_size) / table_size).astype(DTYPE)
        self.phase = 0.0

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
    """ Represents a single synthesizer voice (Oscillator + Amp Env) """
    def __init__(self):
        self.osc = WavetableOscillator()
        self.freq = 440.0
        self.velocity = 0.0
        self.env_state = 'off' # off, a, d, s, r
        self.env_level = 0.0
        
        # ADSR Params
        self.attack = 0.01
        self.decay = 0.1
        self.sustain = 0.5
        self.release = 0.2
        
    def note_on(self, freq, vel, attack=0.01, decay=0.1, sustain=0.5):
        self.freq = freq
        self.velocity = vel
        self.env_state = 'a'
        self.env_level = 0.0
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        
    def note_off(self):
        self.env_state = 'r'
        
    def process(self, frames):
        if self.env_state == 'off':
            return np.zeros(frames, dtype=DTYPE)
            
        samples = self.osc.get_samples(self.freq, frames)
        
        # Simple Envelope Processing (per frame approximation)
        # In a real app, this would be sample-accurate, but this is efficient.
        env_curve = np.ones(frames, dtype=DTYPE)
        
        # Rough approximation for the block
        if self.env_state == 'a':
            target = 1.0
            rate = 1.0 / (self.attack * SAMPLE_RATE)
            self.env_level = min(1.0, self.env_level + rate * frames)
            if self.env_level >= 1.0: self.env_state = 'd'
        elif self.env_state == 'd':
            rate = 1.0 / (self.decay * SAMPLE_RATE)
            self.env_level = max(self.sustain, self.env_level - rate * frames)
            if self.env_level <= self.sustain: self.env_state = 's'
        elif self.env_state == 's':
            pass # level is self.sustain
        elif self.env_state == 'r':
            rate = 1.0 / (self.release * SAMPLE_RATE)
            self.env_level = max(0.0, self.env_level - rate * frames)
            if self.env_level <= 0.0: self.env_state = 'off'
            
        return samples * self.env_level * self.velocity

class AudioEngine:
    def __init__(self):
        self.cmd_queue = queue.Queue()
        
        # --- Parameters ---
        self.params = {
            'master_amp': 0.7, 'tempo': 100, 
            'root_idx': 0, 'scale_idx': 0, 'scale_idx_2': 0, 'fusion': 0.0,
            'reverb_mix': 0.4, 'brightness': 0.8, 'groove': 0.0,
            'melody_dist': 0.5, 'chord_dist': 0.2
        }
        
        # --- Musical State ---
        self.root_names = list(MusicTheory.ROOTS.keys())
        self.scale_names = list(MusicTheory.SCALES.keys())
        self.current_scale_freqs = []
        
        # --- Synth Voices ---
        # Pool of voices for Melody
        self.melody_v = Voice()
        # Voices for Chords (Pad)
        self.chord_v1 = Voice()
        self.chord_v2 = Voice()
        self.chord_v3 = Voice()
        self.bass_v = Voice()
        
        # --- Sequencer State ---
        self.beat_counter = 0
        self.samples_per_beat = int((60.0 / 100) * SAMPLE_RATE)
        self.current_sample = 0
        
        # --- Effects ---
        self.delay_len = int(SAMPLE_RATE * 0.6) # 600ms
        self.delay_buf = np.zeros((self.delay_len, 2), dtype=DTYPE)
        self.delay_idx = 0
        self.filter_z = np.zeros(2, dtype=DTYPE)
        
        # Analysis
        self.spectrum_data = np.zeros(64, dtype=DTYPE)
        
        # Initialize
        self.update_scales()

    def update_scales(self):
        root_name = self.root_names[self.params['root_idx']]
        scale_name_1 = self.scale_names[self.params['scale_idx']]
        scale_name_2 = self.scale_names[self.params['scale_idx_2']]
        fusion = self.params['fusion']
        
        intervals_1 = MusicTheory.SCALES[scale_name_1]
        intervals_2 = MusicTheory.SCALES[scale_name_2]
        
        # Calculate Frequency Lists
        root = MusicTheory.ROOTS[root_name]
        
        freqs_1 = [root * (2**(i/12.0)) for i in intervals_1]
        freqs_2 = [root * (2**(i/12.0)) for i in intervals_2]
        
        # Logic to merge scales if fusion is active
        if fusion > 0.01:
            # Mix logic: Weighted random selection or Union.
            # Here we create a "Hybrid" list based on proximity.
            merged_intervals = list(set(intervals_1 + intervals_2))
            merged_intervals.sort()
            self.current_scale_freqs = [root * (2**(i/12.0)) for i in merged_intervals]
            # Color the chord generation to favor scale 1 or 2
        else:
            self.current_scale_freqs = freqs_1

    def handle_commands(self):
        try:
            while True:
                cmd, val = self.cmd_queue.get_nowait()
                if cmd in self.params:
                    self.params[cmd] = val
                    if cmd in ['root_idx', 'scale_idx', 'scale_idx_2', 'fusion']:
                        self.update_scales()
                    elif cmd == 'tempo':
                        self.samples_per_beat = int((60.0 / val) * SAMPLE_RATE)
        except queue.Empty:
            pass

    def get_chord_freqs(self):
        """ Generates a triad from the current scale """
        if not self.current_scale_freqs: return []
        # Basic triad: Root, Third, Fifth (indices 0, 2, 4)
        # Random inversion for variety
        try:
            idx = random.choice([0, 2, 4, 5]) # Root, 3rd, 5th, or 7th base
            s = self.current_scale_freqs
            # Ensure indices exist
            n = len(s)
            f1 = s[0] # Bass root always
            f2 = s[min(2, n-1)]
            f3 = s[min(4, n-1)]
            return [f1, f2, f3]
        except IndexError:
            return [self.current_scale_freqs[0]]

    def audio_callback(self, outdata, frames, time_info, status):
        try:
            self.handle_commands()
            
            out = np.zeros((frames, 2), dtype=DTYPE)
            
            # --- Sequencer ---
            # Apply Groove (Swing) - adjust trigger threshold slightly
            swing = math.sin(self.beat_counter * math.pi) * self.params['groove'] * (self.samples_per_beat * 0.2)
            
            self.current_sample += frames
            if self.current_sample >= self.samples_per_beat + swing:
                self.current_sample = 0
                self.beat_counter = (self.beat_counter + 1) % 16
                
                # --- Events ---
                is_start = (self.beat_counter % 4 == 0)
                
                # 1. Melody Trigger (Probability based)
                if random.random() < self.params['melody_dist']:
                    # Accent Logic: First beat louder, random ghost notes
                    accent = 1.0 if is_start else random.choice([0.6, 0.8, 0.5, 0.4])
                    if self.current_scale_freqs:
                        freq = random.choice(self.current_scale_freqs)
                        # If fusion is high, occasionally pick from the *other* scale to highlight dissonance
                        if self.params['fusion'] > 0.5 and random.random() < 0.3:
                             s2_name = self.scale_names[self.params['scale_idx_2']]
                             freq = random.choice(MusicTheory.get_scale_freqs(
                                 self.root_names[self.params['root_idx']], s2_name))
                        
                        self.melody_v.note_on(freq, accent * 0.4, attack=0.01, decay=0.2, sustain=0.1)
                
                # 2. Chord Trigger (Less frequent)
                if self.beat_counter % 8 == 0: # Every 2 bars
                    chord_freqs = self.get_chord_freqs()
                    # Trigger chord voices
                    if len(chord_freqs) > 0:
                        self.chord_v1.note_on(chord_freqs[0] * 0.5, 0.15, attack=0.8, decay=1.0, sustain=0.8)
                    if len(chord_freqs) > 1:
                        self.chord_v2.note_on(chord_freqs[1], 0.15, attack=0.8, decay=1.0, sustain=0.8)
                    if len(chord_freqs) > 2:
                        self.chord_v3.note_on(chord_freqs[2], 0.15, attack=0.8, decay=1.0, sustain=0.8)

                # 3. Bass Trigger
                if self.beat_counter % 2 == 0: # Every half bar
                     if self.current_scale_freqs:
                        self.bass_v.note_on(self.current_scale_freqs[0]*0.5, 0.6, attack=0.01, decay=0.1, sustain=0.0)
            
            # --- Render Audio ---
            # Melody (Pan Center)
            out[:,0] += self.melody_v.process(frames)
            out[:,1] += self.melody_v.process(frames)
            
            # Chords (Wide)
            out[:,0] += self.chord_v1.process(frames) * 0.8 + self.chord_v3.process(frames) * 0.2
            out[:,1] += self.chord_v2.process(frames) * 0.8 + self.chord_v1.process(frames) * 0.2
            
            # Bass (Center)
            out[:,0] += self.bass_v.process(frames)
            out[:,1] += self.bass_v.process(frames)
            
            # --- Effects ---
            # Filter
            cutoff = 200 + (4000 * self.params['brightness'])
            dt = 1.0 / SAMPLE_RATE
            rc = 1.0 / (2 * np.pi * cutoff)
            alpha = rc / (rc + dt)
            
            # Stateful filter loop
            new_z = self.filter_z
            for i in range(frames):
                new_z[0] = alpha * out[i, 0] + (1 - alpha) * new_z[0]
                new_z[1] = alpha * out[i, 1] + (1 - alpha) * new_z[1]
                out[i, 0] = new_z[0]
                out[i, 1] = new_z[1]
            self.filter_z = new_z
            
            # Delay
            delay_time = int(SAMPLE_RATE * 0.4)
            mix = self.params['reverb_mix']
            
            read_idx = (self.delay_idx - delay_time + self.delay_len) % self.delay_len
            rc1 = min(frames, self.delay_len - read_idx)
            rc2 = frames - rc1
            
            wet = np.zeros_like(out)
            wet[:rc1] = self.delay_buf[read_idx : read_idx + rc1]
            if rc2 > 0: wet[rc1:] = self.delay_buf[:rc2]
            
            out += wet * mix * 0.5
            
            # Write back
            wc1 = min(frames, self.delay_len - self.delay_idx)
            wc2 = frames - wc1
            self.delay_buf[self.delay_idx : self.delay_idx + wc1] = out[:wc1]
            if wc2 > 0: self.delay_buf[:wc2] = out[wc1:]
            self.delay_idx = (self.delay_idx + frames) % self.delay_len
            
            # --- Master ---
            out *= self.params['master_amp']
            out = np.tanh(out)
            
            outdata[:] = out
            
            if random.random() < 0.1:
                self.spectrum_data = np.abs(np.fft.rfft(out[:, 0] + out[:, 1]))[:64]

        except Exception as e:
            print(f"Audio Error: {e}", file=sys.stderr)

# -------------------- GUI --------------------
class Visualizer(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.minHeight = 120
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(15, 15, 15))
        data = self.engine.spectrum_data
        w, h = self.width(), self.height()
        
        grad = QLinearGradient(0, h, 0, 0)
        grad.setColorAt(0.0, QColor(20, 80, 180))
        grad.setColorAt(0.5, QColor(0, 200, 150))
        grad.setColorAt(1.0, QColor(200, 255, 50))
        
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        
        bar_w = w / 64.0
        for i in range(64):
            val = np.log1p(data[i]) * 15.0
            bar_h = min(h, val * h / 5.0)
            p.drawRect(int(i*bar_w), int(h-bar_h), int(bar_w-1), int(bar_h))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("World Music Procedural Engine")
        self.resize(700, 650)
        
        self.engine = AudioEngine()
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE,
            channels=2, dtype=DTYPE, callback=self.engine.audio_callback
        )
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        lbl = QLabel("Eastern & Western Synthesis Engine")
        lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
        lbl.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(lbl)
        
        # Visualizer
        self.viz = Visualizer(self.engine)
        self.viz.setMinimumHeight(140)
        main_layout.addWidget(self.viz)
        
        # Controls Grid
        grid = QGridLayout()
        main_layout.addLayout(grid)
        
        # --- Theory Controls ---
        grid.addWidget(QLabel("Root Note:"), 0, 0)
        self.root_box = QComboBox()
        self.root_box.addItems(self.engine.root_names)
        self.root_box.currentIndexChanged.connect(lambda i: self.engine.cmd_queue.put(('root_idx', i)))
        grid.addWidget(self.root_box, 0, 1)

        grid.addWidget(QLabel("Scale 1 (Main):"), 1, 0)
        self.scale_box = QComboBox()
        self.scale_box.addItems(self.engine.scale_names)
        self.scale_box.setCurrentIndex(0) # Major
        self.scale_box.currentIndexChanged.connect(lambda i: self.engine.cmd_queue.put(('scale_idx', i)))
        grid.addWidget(self.scale_box, 1, 1)

        grid.addWidget(QLabel("Scale 2 (Fusion):"), 2, 0)
        self.scale_box_2 = QComboBox()
        self.scale_box_2.addItems(self.engine.scale_names)
        self.scale_box_2.setCurrentIndex(8) # Hijaz
        self.scale_box_2.currentIndexChanged.connect(lambda i: self.engine.cmd_queue.put(('scale_idx_2', i)))
        grid.addWidget(self.scale_box_2, 2, 1)

        # --- Mixing Controls ---
        grid.addWidget(QLabel("Scale Fusion:"), 3, 0)
        self.fusion_s = QSlider(Qt.Horizontal)
        self.fusion_s.setRange(0, 100)
        self.fusion_s.setValue(0)
        self.fusion_s.valueChanged.connect(lambda v: self.engine.cmd_queue.put(('fusion', v/100.0)))
        grid.addWidget(self.fusion_s, 3, 1)

        grid.addWidget(QLabel("Tempo:"), 4, 0)
        self.tempo_s = QSlider(Qt.Horizontal)
        self.tempo_s.setRange(60, 180)
        self.tempo_s.setValue(100)
        self.tempo_s.valueChanged.connect(lambda v: self.engine.cmd_queue.put(('tempo', v)))
        grid.addWidget(self.tempo_s, 4, 1)
        
        grid.addWidget(QLabel("Groove (Swing):"), 5, 0)
        self.groove_s = QSlider(Qt.Horizontal)
        self.groove_s.setRange(0, 100)
        self.groove_s.valueChanged.connect(lambda v: self.engine.cmd_queue.put(('groove', v/100.0)))
        grid.addWidget(self.groove_s, 5, 1)

        grid.addWidget(QLabel("Brightness:"), 6, 0)
        self.bright_s = QSlider(Qt.Horizontal)
        self.bright_s.setRange(0, 100)
        self.bright_s.setValue(80)
        self.bright_s.valueChanged.connect(lambda v: self.engine.cmd_queue.put(('brightness', v/100.0)))
        grid.addWidget(self.bright_s, 6, 1)
        
        grid.addWidget(QLabel("Melody Density:"), 7, 0)
        self.mel_s = QSlider(Qt.Horizontal)
        self.mel_s.setRange(0, 100)
        self.mel_s.setValue(50)
        self.mel_s.valueChanged.connect(lambda v: self.engine.cmd_queue.put(('melody_dist', v/100.0)))
        grid.addWidget(self.mel_s, 7, 1)

        grid.addWidget(QLabel("Space:"), 8, 0)
        self.rev_s = QSlider(Qt.Horizontal)
        self.rev_s.setRange(0, 100)
        self.rev_s.setValue(40)
        self.rev_s.valueChanged.connect(lambda v: self.engine.cmd_queue.put(('reverb_mix', v/100.0)))
        grid.addWidget(self.rev_s, 8, 1)

        grid.addWidget(QLabel("Master:"), 9, 0)
        self.vol_s = QSlider(Qt.Horizontal)
        self.vol_s.setRange(0, 100)
        self.vol_s.setValue(70)
        self.vol_s.valueChanged.connect(lambda v: self.engine.cmd_queue.put(('master_amp', v/100.0)))
        grid.addWidget(self.vol_s, 9, 1)

        # Play Button
        self.btn = QPushButton("Start Engine")
        self.btn.setCheckable(True)
        self.btn.setStyleSheet("""
            QPushButton { background-color: #333; color: white; border-radius: 5px; padding: 15px; font-weight: bold; }
            QPushButton:checked { background-color: #d44; }
        """)
        self.btn.clicked.connect(self.toggle)
        main_layout.addWidget(self.btn)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.viz.update)
        self.timer.start(30)

    def toggle(self, state):
        if state:
            self.stream.start()
            self.btn.setText("Stop Engine")
        else:
            self.stream.stop()
            self.btn.setText("Start Engine")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())