import numpy as np
import sounddevice as sd
import random, time

# ---------------- SETTINGS ----------------
sample_rate = 44100
buffer_size = 2048
dtype = 'float32'

# ---------------- UTILS ----------------
def interpolate_param(current, target, speed=0.01):
    return current + (target - current) * speed

def pan_lfo(base_pan=0.0, speed=0.8, depth=0.15):
    # speed in Hz (cycles per second). Using time.time() is OK for LFO here.
    return np.clip(base_pan + np.sin(2*np.pi*time.time()*speed)*depth, -1.0, 1.0)

def microtonal_drift(freq, drift_speed=0.2, drift_depth=0.001):
    # drift_speed in Hz
    return freq * (1 + np.sin(2*np.pi*time.time()*drift_speed)*drift_depth)

def add_dynamic_harmonics(freq, amp, t, num_harmonics=3):
    osc = HarmonicOsc(num_harmonics=num_harmonics)
    return osc.generate(freq, amp, len(t), num_harmonics=num_harmonics)


# ----------- HARMONIC OSCILLATORS -----------
class HarmonicOsc:
    def __init__(self, num_harmonics=3):
        self.num_harmonics = num_harmonics
        self.phases = np.zeros(num_harmonics, dtype=np.float32)

    def generate(self, freq, amp, frames, num_harmonics=None):
        if num_harmonics is None:
            num_harmonics = self.num_harmonics
        t = np.arange(frames, dtype=np.float32) / sample_rate
        signal = np.zeros(frames, dtype=np.float32)
        ph_shift = np.sin(time.time()*0.02) * np.pi
        det_mod = np.sin(time.time()*0.01) * 0.01

        for i in range(num_harmonics):
            f = freq * (i+1 + det_mod)
            phase_inc = 2*np.pi*f/sample_rate
            phase_array = self.phases[i] + phase_inc*np.arange(frames)
            self.phases[i] = (phase_array[-1] + phase_inc) % (2*np.pi)
            signal += (amp/(i*1.5+1.2)) * np.sin(phase_array+ph_shift).astype(np.float32)

        return signal

# ---------------- PROGRESSIONS ----------------
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
PROGRESSIONS = [
    [0, 3, 4, 1],   # I - IV - V - vi
    [0, 4, 3, 5],   # I - V - IV - vii°
    [0, 5, 3, 4],   # I - vi - IV - V
    [0, 3, 5, 4],   # I - IV - vi - V
]

def semitone_freq(f0, semitones):
    return f0 * (2 ** (semitones/12))

def chord_from_scale(root_freq, degree, chord_type='major'):
    note_idx = degree % len(MAJOR_SCALE)
    octave_shift = degree // len(MAJOR_SCALE)
    note = semitone_freq(root_freq, MAJOR_SCALE[note_idx]) * (2 ** octave_shift)
    if chord_type == 'major':
        return [note, note*5/4, note*3/2]
    elif chord_type == 'minor':
        return [note, note*6/5, note*3/2]
    elif chord_type == 'sus2':
        return [note, note*9/8, note*3/2]
    elif chord_type == 'sus4':
        return [note, note*4/3, note*3/2]
    else:
        return [note, note*5/4, note*3/2]

def random_inversion(chord):
    c = chord.copy()
    random.shuffle(c)
    if random.random() < 0.4:
        idx = random.randrange(len(c))
        c[idx] *= 2 if random.random() < 0.5 else 0.5
    return c

# ---------------- ENV CONTROL ----------------
class EnvControl:
    def __init__(self):
        self.key_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
        self.current_chord_idx = 0
        self.progression = random.choice(PROGRESSIONS)
        self.progression_chords = []
        self.last_mod_time = time.time()
        self.mod_interval = 30

    def adaptive_modulation(self, low, mid, high, motif_density):
        now = time.time()
        if now - self.last_mod_time < self.mod_interval:
            return
        self.last_mod_time = now
        tension = low*0.3 + mid*0.4 + high*0.3 + motif_density*0.05
        root = random.choice(self.key_notes)
        scale_type = 'major' if tension < 0.8 else 'minor'
        self.progression = random.choice(PROGRESSIONS)
        self.progression_chords = [chord_from_scale(root, deg, scale_type) for deg in self.progression]
        self.current_chord_idx = 0

    def current_chord(self):
        if not self.progression_chords:
            root = random.choice(self.key_notes)
            self.progression_chords = [chord_from_scale(root, d, 'major') for d in self.progression]
        return self.progression_chords[self.current_chord_idx]

    def next_chord(self):
        if not self.progression_chords:
            self.current_chord()
        self.current_chord_idx = (self.current_chord_idx + 1) % len(self.progression_chords)
        return self.current_chord()

env = EnvControl()

# ---------------- STORY ARC ----------------
STORY_ARC = [
    {'mood':'calm',       'amp':0.3, 'density':10, 'harmonics':2, 'duration':60},
    {'mood':'expansive',  'amp':0.5, 'density':15, 'harmonics':3, 'duration':90},
    {'mood':'mysterious', 'amp':0.6, 'density':20, 'harmonics':3, 'duration':120},
    {'mood':'ethereal',   'amp':0.5, 'density':15, 'harmonics':3, 'duration':90},
    {'mood':'climactic',  'amp':0.8, 'density':30, 'harmonics':5, 'duration':120},
    {'mood':'resolution', 'amp':0.4, 'density':10, 'harmonics':2, 'duration':60},
]

class SceneScript:
    def __init__(self, story_arc):
        self.scenes = story_arc
        self.current_index = 0
        self.start_time = time.time()
        self.on_change_callbacks = []

    def current_scene(self):
        return self.scenes[self.current_index]

    def register_callback(self, func):
        self.on_change_callbacks.append(func)

    def update(self):
        elapsed = time.time() - self.start_time
        current_scene = self.scenes[self.current_index]
        if elapsed > current_scene['duration']:
            self.current_index = (self.current_index + 1) % len(self.scenes)
            self.start_time = time.time()
            print(f"Scene {self.current_index}: {self.scenes[self.current_index]['mood']}")
            for func in self.on_change_callbacks:
                try:
                    func()
                except Exception as e:
                    print("Callback error:", e)

scene_script = SceneScript(STORY_ARC)
global_amp = 0.3

# ---------------- FILTERS (one-pole implementations) ----------------
class LowPass:
    def __init__(self, cutoff=2000):
        self.cutoff = float(cutoff)
        self.prev_left = 0.0
        self.prev_right = 0.0

    def process(self, stereo):
        # standard one-pole lowpass per-sample but vectorized per buffer with scalar alpha
        dt = 1.0 / sample_rate
        # convert cutoff -> alpha for simple one-pole: alpha = dt / (RC + dt), RC = 1/(2*pi*fc)
        rc = 1.0 / (2*np.pi*max(1.0, self.cutoff))
        alpha = dt / (rc + dt)
        left = alpha * stereo[:,0] + (1.0 - alpha) * self.prev_left
        right = alpha * stereo[:,1] + (1.0 - alpha) * self.prev_right
        self.prev_left = float(left[-1])
        self.prev_right = float(right[-1])
        stereo[:,0] = left
        stereo[:,1] = right
        return stereo

class HighPass:
    def __init__(self, cutoff=20):
        self.cutoff = float(cutoff)
        self.prev_in_l = 0.0; self.prev_out_l = 0.0
        self.prev_in_r = 0.0; self.prev_out_r = 0.0

    def process(self, stereo):
        dt = 1.0 / sample_rate
        rc = 1.0 / (2*np.pi*max(0.1, self.cutoff))
        alpha = rc / (rc + dt)  # simple high-pass derived from bilinear-like form
        # vectorized difference equation: y[n] = alpha*(y[n-1] + x[n] - x[n-1])
        x_l = stereo[:,0]
        x_r = stereo[:,1]
        y_l = np.empty_like(x_l)
        y_r = np.empty_like(x_r)
        y_prev = self.prev_out_l
        x_prev = self.prev_in_l
        for i in range(len(x_l)):
            y = alpha * (y_prev + x_l[i] - x_prev)
            y_l[i] = y
            y_prev = y
            x_prev = x_l[i]
        self.prev_out_l = float(y_prev); self.prev_in_l = float(x_prev)
        y_prev = self.prev_out_r
        x_prev = self.prev_in_r
        for i in range(len(x_r)):
            y = alpha * (y_prev + x_r[i] - x_prev)
            y_r[i] = y
            y_prev = y
            x_prev = x_r[i]
        self.prev_out_r = float(y_prev); self.prev_in_r = float(x_prev)
        stereo[:,0] = y_l; stereo[:,1] = y_r
        return stereo

my_lp = LowPass(2000)
my_hp = HighPass(20)

def update_filter_sweep(scene_data):
    mood_cutoffs = {
        'calm': (2000, 4000),
        'expansive': (3000, 6000),
        'mysterious': (1500, 3500),
        'ethereal': (4000, 7000),
        'climactic': (6000, 12000),
        'resolution': (2000, 4000)
    }
    lo, hi = mood_cutoffs.get(scene_data['mood'], (2000, 5000))
    target = lo + (hi - lo) * (0.5 + 0.5*np.sin(time.time()*0.05))
    my_lp.cutoff = interpolate_param(my_lp.cutoff, target, speed=0.01)

# ---------------- REVERB ----------------
class Reverb:
    def __init__(self, decay=0.6):
        self.buf_len = int(sample_rate * 1.0)
        self.buffer_left = np.zeros(self.buf_len, dtype=np.float32)
        self.buffer_right = np.zeros(self.buf_len, dtype=np.float32)
        self.decay = float(decay)
        self.ptr = 0

    def process(self, stereo):
        n = len(stereo)
        idxs = (self.ptr + np.arange(n)) % self.buf_len
        # read
        stereo[:,0] += self.buffer_left[idxs] * (self.decay * 0.7)
        stereo[:,1] += self.buffer_right[idxs] * (self.decay * 0.7)
        # write with some feedback smoothing
        self.buffer_left[idxs] = stereo[:,0] * 0.3 + self.buffer_left[idxs] * 0.7
        self.buffer_right[idxs] = stereo[:,1] * 0.3 + self.buffer_right[idxs] * 0.7
        self.ptr = (self.ptr + n) % self.buf_len
        return stereo

my_reverb = Reverb(decay=0.6)

# ---------------- SPATIAL MOD ----------------
def spatial_modulation(stereo, scene_data):
    width_map = {'calm':0.5,'expansive':0.8,'mysterious':0.6,'ethereal':0.9,'climactic':1.0,'resolution':0.5}
    width = width_map.get(scene_data['mood'], 0.7)
    mid = (stereo[:,0] + stereo[:,1]) * 0.5
    side = (stereo[:,0] - stereo[:,1]) * 0.5 * width
    stereo[:,0] = mid + side
    stereo[:,1] = mid - side
    return stereo

# ---------------- SPECTRAL FLARES ----------------
def spectral_flares(stereo):
    left = stereo[:,0].astype(np.float32); right = stereo[:,1].astype(np.float32)
    fft_l = np.fft.rfft(left)
    fft_r = np.fft.rfft(right)
    freqs = np.fft.rfftfreq(len(left), 1/sample_rate)
    # sub-bass reduction
    fft_l[freqs<30] *= 0.01
    fft_r[freqs<30] *= 0.01
    # high shelf
    fft_l[freqs>8000] *= 1.15
    fft_r[freqs>8000] *= 1.15
    flare = 1 + 0.05*np.sin(time.time()*5.0)
    mask = (freqs>2000) & (freqs<6000)
    fft_l[mask] *= flare
    fft_r[mask] *= flare
    stereo[:,0] = np.fft.irfft(fft_l, n=len(left)).astype(np.float32)
    stereo[:,1] = np.fft.irfft(fft_r, n=len(right)).astype(np.float32)
    return stereo

# ---------------- PAD ----------------
class ChordPad:
    def __init__(self, chord, fade_in=1.0):
        self.chord = chord.copy()
        self.amp = random.uniform(0.04,0.07)
        self.pans = [random.uniform(-0.3,0.3) for _ in chord]
        self.fade = 0.0
        self.fade_target = fade_in
        self.fade_speed = 0.005
        self.note_targets = self.chord.copy()
        self.move_speed = 0.001 + random.random()*0.003
        self.osc = HarmonicOsc(num_harmonics=6)

    def generate(self, t, num_harmonics=3):
        self.fade = interpolate_param(self.fade, self.fade_target, speed=self.fade_speed)
        left = np.zeros_like(t, dtype=np.float32); right = np.zeros_like(t, dtype=np.float32)
        swell = (np.sin(time.time()*0.05)+1)/2 * 0.05
        for i in range(len(self.chord)):
            if random.random() < 0.001:
                self.note_targets[i] = self.chord[i] * (0.98 + 0.04*random.random())
            self.chord[i] += (self.note_targets[i] - self.chord[i]) * self.move_speed
            freq = microtonal_drift(self.chord[i], drift_speed=0.3, drift_depth=0.002)
            wave = self.osc.generate(freq, (self.amp+swell)*self.fade, len(t), num_harmonics=num_harmonics)
            attack = np.linspace(0, 1, len(t), dtype=np.float32)
            wave *= attack
            pan = pan_lfo(self.pans[i], speed=0.5, depth=0.2)
            left += wave*np.sqrt(max(0.0, 0.5*(1-pan))); right += wave*np.sqrt(max(0.0, 0.5*(1+pan)))
        return left, right

# ---------------- MOTIF ----------------
class Motif:
    def __init__(self, chord, length=4, fade_in=1.0):
        self.chord = chord.copy()
        self.length = length
        self.notes = [random.choice(chord) for _ in range(length)]
        self.counter_notes = [note*(0.98 + random.random()*0.04) for note in self.notes]
        self.durations = [random.uniform(0.8,1.5) for _ in range(length)]
        self.amps = [random.uniform(0.03,0.1) for _ in range(length)]
        self.counter_amps = [amp*0.5 for amp in self.amps]
        self.pans = [random.uniform(-0.5,0.5) for _ in range(length)]
        self.start_time = time.time()
        self.note_index = 0
        self.fade = 0.0
        self.fade_target = fade_in
        self.fade_speed = 0.01
        self.target_notes = self.notes.copy()
        self.target_counter = self.counter_notes.copy()
        self.move_speed = 0.002
        self.osc_main = HarmonicOsc(num_harmonics=4)
        self.osc_counter = HarmonicOsc(num_harmonics=3)

    def generate(self, t, num_harmonics=3):
        self.fade += (self.fade_target - self.fade) * self.fade_speed
        elapsed = time.time() - self.start_time
        dur_cumsum = np.cumsum(self.durations)
        while self.note_index < len(dur_cumsum) and elapsed > dur_cumsum[self.note_index]:
            self.note_index += 1
            if self.note_index < len(self.notes):
                self.target_notes[self.note_index] = random.choice(self.chord)
                self.target_counter[self.note_index] = self.target_notes[self.note_index] * (0.98 + random.random()*0.04)
        if self.note_index >= len(self.notes):
            return np.zeros_like(t, dtype=np.float32), np.zeros_like(t, dtype=np.float32), False
        # move toward target
        self.notes[self.note_index] += (self.target_notes[self.note_index] - self.notes[self.note_index]) * self.move_speed
        self.counter_notes[self.note_index] += (self.target_counter[self.note_index] - self.counter_notes[self.note_index]) * self.move_speed
        freq_main = microtonal_drift(self.notes[self.note_index], drift_speed=0.8)
        freq_counter = microtonal_drift(self.counter_notes[self.note_index], drift_speed=0.9)
        amp_main = self.amps[self.note_index]*self.fade
        amp_counter = self.counter_amps[self.note_index]*self.fade
        wave_main = self.osc_main.generate(freq_main, amp_main, len(t), num_harmonics=num_harmonics)
        wave_counter = self.osc_counter.generate(freq_counter, amp_counter, len(t), num_harmonics=max(2, num_harmonics-1))
        pan = pan_lfo(self.pans[self.note_index], speed=1.0)
        left = wave_main*np.sqrt(0.5*(1-pan)) + wave_counter*np.sqrt(0.5*(1-(pan*0.5)))
        right = wave_main*np.sqrt(0.5*(1+pan)) + wave_counter*np.sqrt(0.5*(1+(pan*0.5)))
        return left, right, True

#---------------- MELODY ----------------
class Melody:
    def __init__(self, chord, bpm=60):
        self.chord = chord
        self.bpm = bpm
        self.note_length = 60.0 / bpm
        self.last_note_time = time.time()
        self.current_freq = None
        self.amp = 0.05
        self.osc = HarmonicOsc(num_harmonics=4)

    def generate(self, frames):
        t = np.arange(frames, dtype=np.float32) / sample_rate
        now = time.time()
        if self.current_freq is None or (now - self.last_note_time > self.note_length):
            # pick chord tone or passing tone
            base = random.choice(self.chord)
            if random.random() < 0.3:  # 30% chance to drift outside chord
                semitone = random.choice([-2, -1, 1, 2])
                base = base * (2**(semitone/12))
            self.current_freq = base
            self.amp = random.uniform(0.03, 0.07)
            self.last_note_time = now

        wave = self.osc.generate(self.current_freq, self.amp, frames, num_harmonics=3)
        # gentle ADSR envelope
        env = np.hanning(len(wave))
        return wave*env*0.8, wave*env*0.8
    
melody = Melody(env.current_chord(), bpm=70)

# ---------------- SOFT PERCUSSION ----------------
class SoftPercussion:
    def __init__(self, rate=0.3, amp=0.015, decay=0.04):
        self.rate = rate
        self.amp = amp
        self.decay = decay
        self.active = []

    def generate(self, frames):
        t = np.arange(frames, dtype=np.float32)/sample_rate
        out = np.zeros(frames, dtype=np.float32)
        if random.random() < self.rate * (frames / sample_rate):
            self.active.append({'t0': time.time(), 'freq': random.uniform(250, 700), 'phase': random.random()*2*np.pi})
        rem = []
        for g in self.active:
            age = time.time() - g['t0']
            if age < 0.15:
                env = np.exp(-age/self.decay)
                wave = env * np.sin(2*np.pi*g['freq']*t + g['phase'])
                if age < 0.005:  # 5ms fade-in
                    wave *= (age/0.005)
                out += self.amp * wave
                rem.append(g)
        self.active = rem
        return out

soft_perc = SoftPercussion(rate=0.25, amp=0.012, decay=0.05)

# ---------------- STATE ----------------
active_motifs = []
chord_pads = []
grains = []

def handle_scene_change():
    # fade out existing layers
    for pad in chord_pads: pad.fade_target = 0.0
    for motif in active_motifs: motif.fade_target = 0.0
    # advance chord
    current = env.next_chord()
    # add two pad layers + one motif
    for _ in range(2):
        chord_pads.append(ChordPad(random_inversion(current), fade_in=1.0))
    active_motifs.append(Motif(random_inversion(current), fade_in=1.0))

scene_script.register_callback(handle_scene_change)

# ---------------- TENSION MAPPING ----------------
def tension_to_music_params(low, mid, high, motif_density):
    tension = low*0.3 + mid*0.4 + high*0.3
    density = int(5 + tension*30)
    motif_rate = 0.002 + tension*0.008
    pad_swell = 0.03 + tension*0.07
    return tension, density, motif_rate, pad_swell

# ---------------- AUDIO CALLBACK ----------------
def audio_callback(outdata, frames, time_info, status):
    global active_motifs, chord_pads, global_amp
    t = (np.arange(frames, dtype=np.float32) / sample_rate).astype(np.float32)
    stereo = np.zeros((frames,2), dtype=np.float32)

    # Scene update
    scene_script.update()
    scene = scene_script.current_scene()
    global_amp = interpolate_param(global_amp, scene['amp'], speed=0.005)
    num_harm = scene['harmonics']

    # Adaptive env modulation (periodic)
    env.adaptive_modulation(0.2, 0.5, 0.7, len(active_motifs)/10)

    # Tension mapping
    tension, density, motif_rate, pad_swell = tension_to_music_params(0.2, 0.5, 0.7, len(active_motifs)/10)

    # Spawn new motif occasionally based on tension
    if random.random() < motif_rate:
        active_motifs.append(Motif(env.current_chord(), fade_in=1.0))

    # Pads
    for pad in chord_pads:
        pad.amp = interpolate_param(pad.amp, pad_swell, speed=0.01)
        l,r = pad.generate(t, num_harmonics=num_harm)
        stereo[:,0] += l; stereo[:,1] += r

    # Motifs
    remaining = []
    for motif in active_motifs:
        l,r,alive = motif.generate(t, num_harmonics=num_harm)
        stereo[:,0] += l; stereo[:,1] += r
        if alive: remaining.append(motif)
    active_motifs = remaining

    # Melody
    l, r = melody.generate(frames)
    stereo[:,0] += l; stereo[:,1] += r

    # Soft percussion
    perc = soft_perc.generate(frames)
    stereo[:,0] += perc * 0.7
    stereo[:,1] += perc * 0.7

    # Reverb
    stereo = my_reverb.process(stereo)

    # Dynamic filter sweep + filters
    update_filter_sweep(scene)
    stereo = my_lp.process(stereo)
    stereo = my_hp.process(stereo)

    # Spectral flares/shimmer (FFT)
    stereo = spectral_flares(stereo)

    # Spatial width per scene
    stereo = spatial_modulation(stereo, scene)

    # Output soft clip & global amp (ensure within -1..1)
    out = np.tanh(stereo * (1.1 * global_amp)).astype(np.float32)
    outdata[:] = out

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("Cinematic ambient engine running. Press Ctrl+C to stop.")
    with sd.OutputStream(channels=2, callback=audio_callback,
                         samplerate=sample_rate, blocksize=buffer_size,
                         dtype=dtype):
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            print("Stopped.")
