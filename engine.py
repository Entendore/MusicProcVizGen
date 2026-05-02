# engine.py
import sys
import numpy as np
import random
import math
import queue
import threading
import wave
import time  # <--- ADDED IMPORT
from config import SAMPLE_RATE, BUFFER_SIZE, DTYPE
from dsp import Voice, Filter, Limiter
from theory import MusicTheory

def generate_euclidean_pattern(pulses, steps):
    """Generates a Euclidean rhythm pattern (binary list)."""
    if steps == 0: return []
    if pulses >= steps: return [1] * steps
    if pulses == 0: return [0] * steps
    
    pattern = [1] * pulses + [0] * (steps - pulses)
    
    # Bjorklund algorithm approximation for even distribution
    bucket = 0
    output = []
    for _ in range(steps):
        bucket += pulses
        if bucket >= steps:
            bucket -= steps
            output.append(1)
        else:
            output.append(0)
    return output

class AudioEngine:
    def __init__(self):
        self.cmd_queue = queue.Queue()
        self.save_filename = "output.wav"
        
        # --- Parameters ---
        self.params = {
            'master_amp': 0.75, 'tempo': 100, 
            'root_idx': 0, 'scale_idx': 0, 'scale_idx_2': 8, 'fusion': 0.0,
            'reverb_mix': 0.4, 'brightness': 0.7, 'groove': 0.2,
            'melody_dist': 0.6, 'modulation': 0.3, 'chord_freq': 0.5,
            'algo_idx': 0 
        }
        
        # --- Musical State ---
        self.root_names = list(MusicTheory.ROOTS.keys())
        self.scale_names = list(MusicTheory.SCALES.keys())
        self.current_scale_freqs = []
        
        # --- Chord & Morphing State ---
        self.current_progression = [0, 3, 4, 4] # Degrees I, IV, V, V
        self.current_chord_degree = 0 # Index in progression
        self.morph_timer = 0
        self.auto_morph_speed = 0.001 
        
        # --- Synth Voices ---
        self.melody_v = Voice(waveform='saw')
        self.chord_voices = [Voice(waveform='tri') for _ in range(4)] 
        self.bass_v = Voice(waveform='sine')
        self.bass_v_sub = Voice(waveform='sine')
        
        # --- Sequencer State ---
        self.beat_counter = 0
        self.samples_per_beat = int((60.0 / 100) * SAMPLE_RATE)
        self.current_sample = 0
        
        # --- Algorithm Specific State ---
        self.markov_current_idx = 0 
        self.euclid_pattern_melody = generate_euclidean_pattern(5, 16)
        self.euclid_pattern_bass = generate_euclidean_pattern(3, 8)
        self.euclid_pattern_chord = generate_euclidean_pattern(1, 4)
        
        # --- Effects ---
        self.delay_buf = np.zeros((int(SAMPLE_RATE * 1.0), 2), dtype=DTYPE)
        self.delay_idx = 0
        self.filter = Filter()
        self.limiter = Limiter()
        
        # --- Recording ---
        self.is_recording = False
        self.record_buffer = []
        self.rec_lock = threading.Lock()

        # Analysis
        self.spectrum_data = np.zeros(64, dtype=DTYPE)
        self.waveform_data = np.zeros(128, dtype=DTYPE) 
        self.vu_levels = [0.0, 0.0]
        
        self.update_scales()

    def update_scales(self):
        root_name = self.root_names[self.params['root_idx']]
        scale_name_1 = self.scale_names[self.params['scale_idx']]
        scale_name_2 = self.scale_names[self.params['scale_idx_2']]
        
        fusion_val = self.params.get('auto_fusion', self.params['fusion'])
        
        intervals_1 = MusicTheory.SCALES[scale_name_1]
        intervals_2 = MusicTheory.SCALES[scale_name_2]
        root = MusicTheory.ROOTS[root_name]
        
        merged_intervals = MusicTheory.merge_scales(intervals_1, intervals_2, fusion_val)
        
        self.current_scale_freqs = [root * (2**(i/12.0)) for i in merged_intervals]
        
    def set_recording(self, state):
        with self.rec_lock:
            if state and not self.is_recording:
                self.record_buffer = []
                self.is_recording = True
            elif not state and self.is_recording:
                self.is_recording = False

    def save_audio(self):
        if not self.record_buffer:
            return
        
        full_audio = np.concatenate(self.record_buffer)
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val
        
        int_audio = (full_audio * 32767).astype(np.int16)
        
        try:
            with wave.open(self.save_filename, 'w') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(int_audio.tobytes())
            print(f"Saved {self.save_filename}")
        except Exception as e:
            print(f"Error saving: {e}")

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

    def get_current_chord_freqs(self):
        """ Returns frequencies for the current chord in the progression """
        if not self.current_scale_freqs: return []
        
        degree = self.current_progression[self.current_chord_degree % len(self.current_progression)]
        return MusicTheory.build_chord_from_scale(self.current_scale_freqs, degree, octave_shift=-1)

    def audio_callback(self, outdata, frames, time_info, status):
        try:
            self.handle_commands()
            
            out = np.zeros((frames, 2), dtype=DTYPE)
            
            # --- Sequencer ---
            swing = math.sin(self.beat_counter * math.pi) * self.params['groove'] * (self.samples_per_beat * 0.1)
            
            self.current_sample += frames
            if self.current_sample >= self.samples_per_beat + swing:
                self.current_sample = 0
                self.beat_counter = (self.beat_counter + 1) % 16
                
                if self.beat_counter % 4 == 0:
                    self.evolve_music()
                
                algo = self.params['algo_idx']
                
                if algo == 0:
                    self.run_weighted_random_gen()
                elif algo == 1:
                    self.run_markov_gen()
                elif algo == 2:
                    self.run_euclidean_gen()

            
            # --- Render Audio ---
            mix = np.zeros((frames, 2), dtype=DTYPE)
            
            m_sig = self.melody_v.process(frames)
            mix[:,0] += m_sig * 0.7
            mix[:,1] += m_sig * 0.9
            
            for v in self.chord_voices:
                sig = v.process(frames)
                mix[:,0] += sig * 0.4
                mix[:,1] += sig * 0.4
            
            bass_sig = self.bass_v.process(frames) + self.bass_v_sub.process(frames)
            mix[:,0] += bass_sig
            mix[:,1] += bass_sig
            
            # --- Effects ---
            cutoff = 300 + (4000 * self.params['brightness'])
            mix = self.filter.process(mix, cutoff)
            
            delay_time = int(SAMPLE_RATE * 0.5)
            mix_len = len(mix)
            
            read_idx = (self.delay_idx - delay_time) % len(self.delay_buf)
            wet = np.zeros_like(mix)
            
            for i in range(mix_len):
                r_idx = (read_idx + i) % len(self.delay_buf)
                wet[i] = self.delay_buf[r_idx]
            
            mix += wet * self.params['reverb_mix'] * 0.6
            
            for i in range(mix_len):
                self.delay_buf[self.delay_idx] = mix[i]
                self.delay_idx = (self.delay_idx + 1) % len(self.delay_buf)
            
            mix *= self.params['master_amp']
            mix = self.limiter.process(mix)
            
            outdata[:] = mix
            
            if self.is_recording:
                with self.rec_lock:
                    self.record_buffer.append(mix.copy())
            
            if random.random() < 0.2:
                self.spectrum_data = np.abs(np.fft.rfft(mix[:, 0]))[:64]
                step = max(1, len(mix) // 128)
                self.waveform_data = mix[::step, 0][:128]
                self.vu_levels = [np.max(np.abs(mix[:,0])), np.max(np.abs(mix[:,1]))]

        except Exception as e:
            print(f"Audio Error: {e}", file=sys.stderr)

    # ---------------------------------------------------------
    # MUSICAL EVOLUTION (MORPHING)
    # ---------------------------------------------------------
    def evolve_music(self):
        """ Handles automatic morphing between scales and chords """
        
        # 1. Chord Progression Change
        if random.random() < 0.2:
            self.current_progression = random.choice(MusicTheory.PROGRESSIONS)
        
        # 2. Advance Chord Degree
        self.current_chord_degree = (self.current_chord_degree + 1) % len(self.current_progression)
        
        # 3. Scale Morphing
        morph_time = time.time()
        lfo = (math.sin(morph_time * 0.1) + 1.0) / 2.0
        
        manual_fusion = self.params['fusion']
        self.params['auto_fusion'] = max(0, min(1, manual_fusion + (lfo * 0.3 - 0.15)))
        
        self.update_scales()

    # ---------------------------------------------------------
    # GENERATION ALGORITHMS
    # ---------------------------------------------------------

    def run_weighted_random_gen(self):
        """ Algorithm 0: Pop/Standard structure with defined chords """
        
        if random.random() < self.params['melody_dist']:
            if self.current_scale_freqs:
                chord_freqs = self.get_current_chord_freqs()
                
                if random.random() < 0.6 and chord_freqs:
                    freq = random.choice(chord_freqs) * 2
                else:
                    freq = random.choice(self.current_scale_freqs) * (1 + random.randint(0,1))
                
                self.melody_v.note_on(freq, 0.6, attack=0.01, decay=0.1, sustain=0.1)
        
        chord_chance = 0.1 + (self.params['chord_freq'] * 0.5)
        
        if self.beat_counter % 4 == 0 or (self.beat_counter % 2 == 0 and random.random() < chord_chance):
            chord_freqs = self.get_current_chord_freqs()
            if len(chord_freqs) >= 3:
                for i, v in enumerate(self.chord_voices):
                    if i < len(chord_freqs):
                        v.note_on(chord_freqs[i], 0.3, attack=0.1, decay=0.5, sustain=0.3)

        if self.beat_counter % 2 == 0:
             if self.current_scale_freqs:
                chord_root_idx = self.current_progression[self.current_chord_degree % len(self.current_progression)]
                try:
                    freq = self.current_scale_freqs[chord_root_idx % len(self.current_scale_freqs)] * 0.5
                    self.bass_v.note_on(freq, 0.7, attack=0.01, decay=0.1, sustain=0.0)
                except:
                    pass

    def run_markov_gen(self):
        """ Algorithm 1: Flowing melody with scale morphing emphasis """
        if not self.current_scale_freqs: return
        
        max_idx = len(self.current_scale_freqs) - 1
        
        trans_probs = [0.35, 0.35, 0.1, 0.1, 0.1]
        move = random.choices(range(5), weights=trans_probs)[0]
        step_size = 1 if move < 2 else random.randint(2, 3)
        
        if move == 0: self.markov_current_idx = min(max_idx, self.markov_current_idx + step_size)
        elif move == 1: self.markov_current_idx = max(0, self.markov_current_idx - step_size)
        elif move == 2: self.markov_current_idx = min(max_idx, self.markov_current_idx + step_size)
        elif move == 3: self.markov_current_idx = max(0, self.markov_current_idx - step_size)
        
        if random.random() < self.params['melody_dist']:
            freq = self.current_scale_freqs[self.markov_current_idx]
            self.melody_v.note_on(freq, 0.4, attack=0.05, decay=0.3, sustain=0.4)
            
        if self.beat_counter % 8 == 0:
            chord_freqs = self.get_current_chord_freqs()
            for i, v in enumerate(self.chord_voices):
                if i < len(chord_freqs):
                    v.note_on(chord_freqs[i], 0.2, attack=1.5, decay=2.0, sustain=0.6)
        
        if self.beat_counter % 4 == 0:
            root_freq = self.current_scale_freqs[0]
            self.bass_v.note_on(root_freq*0.5, 0.6, attack=0.01, decay=0.2, sustain=0.0)

    def run_euclidean_gen(self):
        """ Algorithm 2: Complex Rhythms based on Euclidean patterns """
        if not self.current_scale_freqs: return
        
        step = self.beat_counter
        
        if random.random() < 0.1:
             pulses = int(3 + self.params['melody_dist'] * 10)
             self.euclid_pattern_melody = generate_euclidean_pattern(pulses, 16)

        if self.euclid_pattern_melody[step % len(self.euclid_pattern_melody)]:
            chord_freqs = self.get_current_chord_freqs()
            if chord_freqs:
                idx = step % len(chord_freqs)
                freq = chord_freqs[idx] * 2
                self.melody_v.note_on(freq, 0.5, attack=0.005, decay=0.05, sustain=0.0)
            
        if self.euclid_pattern_bass[step % len(self.euclid_pattern_bass)]:
            self.bass_v.note_on(self.current_scale_freqs[0]*0.5, 0.7, attack=0.01, decay=0.2, sustain=0.0)
            
        if self.euclid_pattern_chord[step % len(self.euclid_pattern_chord)]:
            chord_freqs = self.get_current_chord_freqs()
            if len(chord_freqs) > 0:
                for i, v in enumerate(self.chord_voices):
                     if i < len(chord_freqs):
                        v.note_on(chord_freqs[i], 0.3, attack=0.005, decay=0.1, sustain=0.0)