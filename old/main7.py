import numpy as np
import sounddevice as sd
import random, time

# ---------------- SETTINGS ----------------
sample_rate = 44100
buffer_size = 1024
t = np.arange(buffer_size) / sample_rate

# ---------------- UTILITY FUNCTIONS ----------------
def interpolate_param(current, target, speed=0.002):
    return current + (target - current) * speed

def pan_lfo(base_pan=0.0, speed=0.01, depth=0.2):
    return np.clip(base_pan + np.sin(time.time()*speed)*depth, -1.0, 1.0)

def microtonal_drift(freq, drift_speed=0.01, drift_depth=0.002):
    return freq * (1 + np.sin(time.time()*drift_speed)*drift_depth)

def add_dynamic_harmonics(freq, base_amp, t, num_harmonics=3):
    signal = np.zeros_like(t)
    for i in range(1, num_harmonics+1):
        detune = freq*(i + np.sin(time.time()*0.01)*0.01)
        amp = base_amp / (i*1.5)
        signal += amp * np.sin(2*np.pi*detune*t + np.sin(time.time()*0.02)*np.pi)
    return signal

# ---------------- ENVIRONMENT CONTROL ----------------
class EnvControl:
    def __init__(self):
        self.key_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
        self.chords = [[261.63, 329.63, 392.00]]
        self.current_chord_idx = 0
        self.last_mod_time = time.time()
        self.mod_interval = 30

    def adaptive_modulation(self, low, mid, high, motif_density):
        now = time.time()
        if now - self.last_mod_time < self.mod_interval: return
        self.last_mod_time = now
        tension = mid + high + motif_density*0.1
        root = random.choice(self.key_notes)
        if tension<0.4: scale_type='major'
        elif tension<0.6: scale_type='sus2'
        elif tension<0.8: scale_type='sus4'
        else: scale_type='minor'
        if scale_type=='major': self.chords=[[root, root*5/4, root*3/2]]
        elif scale_type=='minor': self.chords=[[root, root*6/5, root*3/2]]
        elif scale_type=='sus2': self.chords=[[root, root*9/8, root*3/2]]
        elif scale_type=='sus4': self.chords=[[root, root*4/3, root*3/2]]

env = EnvControl()

# ---------------- SCENE SCRIPT ----------------
class SceneScript:
    def __init__(self):
        # Scene list: duration, amp, density, harmonics, mood
        self.scenes = [
            {'duration':60, 'amp':0.3, 'density':10, 'harmonics':2, 'mood':'calm'},
            {'duration':90, 'amp':0.6, 'density':20, 'harmonics':3, 'mood':'expansive'},
            {'duration':120,'amp':0.8,'density':25,'harmonics':4,'mood':'mysterious'},
            {'duration':90, 'amp':0.5,'density':15,'harmonics':3,'mood':'ethereal'},
            {'duration':120,'amp':0.7,'density':30,'harmonics':5,'mood':'climactic'}
        ]
        self.current_index=0
        self.start_time=time.time()

    def current_scene(self):
        return self.scenes[self.current_index]

    def update(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.scenes[self.current_index]['duration']:
            self.current_index = (self.current_index + 1) % len(self.scenes)
            self.start_time = time.time()
            print(f"Scene {self.current_index}: {self.scenes[self.current_index]['mood']}")

scene_script = SceneScript()
global_amp = 0.3

# ---------------- SOUND ELEMENTS ----------------
grains = []
active_motifs = []
drones = []
chord_pads = []
active_shimmers = []
active_env_triggers = []
max_harmonics = 3
bg_texture = np.random.randn(buffer_size)*0.002

# ---------------- MOTIF CLASS ----------------
class Motif:
    def __init__(self,chord,length=4):
        self.chord = chord
        self.notes=[random.choice(chord) for _ in range(length)]
        self.durations=[random.uniform(0.8,1.5) for _ in range(length)]
        self.amps=[random.uniform(0.03,0.1) for _ in range(length)]
        self.pans=[random.uniform(-0.5,0.5) for _ in range(length)]
        self.start_time = time.time()
        self.note_index=0

    def generate(self,t):
        elapsed = time.time() - self.start_time
        dur_cumsum = np.cumsum(self.durations)
        while self.note_index < len(dur_cumsum) and elapsed > dur_cumsum[self.note_index]:
            self.note_index +=1
        if self.note_index>=len(self.notes):
            return np.zeros_like(t), np.zeros_like(t), False
        freq = microtonal_drift(self.notes[self.note_index])
        amp = self.amps[self.note_index]
        wave = add_dynamic_harmonics(freq, amp, t)
        pan = pan_lfo(self.pans[self.note_index])
        left = wave*np.sqrt(0.5*(1-pan))
        right = wave*np.sqrt(0.5*(1+pan))
        return left, right, True

    def contextual_adapt(self, low, mid, high):
        self.amps = [a*(1 + (high-low)*0.05) for a in self.amps]
        self.pans = [np.clip(p + np.random.uniform(-0.01,0.01), -0.5,0.5) for p in self.pans]

# ---------------- CHORD PAD ----------------
class ChordPad:
    def __init__(self,chord):
        self.chord = chord
        self.amp = 0.05
        self.pans = [random.uniform(-0.3,0.3) for _ in chord]

    def generate(self,t):
        left = np.zeros_like(t)
        right = np.zeros_like(t)
        swell = (np.sin(time.time()*0.05)+1)/2 * 0.05
        for i,freq in enumerate(self.chord):
            freq = microtonal_drift(freq, drift_speed=0.003, drift_depth=0.002)
            wave = add_dynamic_harmonics(freq, self.amp+swell, t)
            pan = pan_lfo(self.pans[i], speed=0.005, depth=0.2)
            left += wave*np.sqrt(0.5*(1-pan))
            right += wave*np.sqrt(0.5*(1+pan))
        return left,right

# ---------------- FILTER & REVERB ----------------
class Filter:
    def __init__(self, cutoff=2000):
        self.cutoff = cutoff
        self.prev_left = 0
        self.prev_right = 0

    def process(self, stereo):
        alpha = self.cutoff / (self.cutoff + sample_rate/2)
        left = alpha*stereo[:,0] + (1-alpha)*self.prev_left
        right = alpha*stereo[:,1] + (1-alpha)*self.prev_right
        self.prev_left = left[-1]; self.prev_right = right[-1]
        stereo[:,0] = left; stereo[:,1] = right
        return stereo

my_filter = Filter(2000)

def evolving_reverb(stereo):
    room_size = 0.4 + 0.3*np.sin(time.time()*0.005)
    decay = 0.5 + 0.4*np.sin(time.time()*0.002)
    left = stereo[:,0]*0.7 + np.roll(stereo[:,0], int(200*room_size))*0.3*decay
    right = stereo[:,1]*0.7 + np.roll(stereo[:,1], int(-180*room_size))*0.3*decay
    stereo[:,0] = left; stereo[:,1] = right
    return stereo

# ---------------- AUDIO CALLBACK ----------------
def audio_callback(outdata, frames, time_info, status):
    global grains, active_motifs, chord_pads, active_shimmers, active_env_triggers, max_harmonics, global_amp, drones
    stereo = np.zeros((buffer_size,2))
    buffer = np.zeros(buffer_size)

    # Scene update
    scene_script.update()
    scene_data = scene_script.current_scene()
    global_amp = interpolate_param(global_amp, scene_data['amp'])
    target_density = int(interpolate_param(len(grains), scene_data['density']))
    max_harmonics = scene_data['harmonics']
    mood = scene_data['mood']

    # Filter adaptation
    if mood=='calm': my_filter.cutoff = 3000
    elif mood=='expansive': my_filter.cutoff = 5000
    elif mood=='mysterious': my_filter.cutoff = 2000
    elif mood=='ethereal': my_filter.cutoff = 6000
    elif mood=='climactic': my_filter.cutoff = 8000

    # Grains
    while len(grains) < target_density:
        grains.append({'freq':random.uniform(220,880),'amp':random.uniform(0.02,0.08),'pan':random.uniform(-0.5,0.5)})
    for grain in grains:
        freq = microtonal_drift(grain['freq'])
        wave = add_dynamic_harmonics(freq, grain['amp']*global_amp, t, num_harmonics=max_harmonics)
        pan = pan_lfo(grain['pan'])
        stereo[:,0]+=wave*np.sqrt(0.5*(1-pan))
        stereo[:,1]+=wave*np.sqrt(0.5*(1+pan))
        buffer += wave

    # Motifs
    remaining_motifs=[]
    for motif in active_motifs:
        l,r,alive = motif.generate(t)
        stereo[:,0]+=l; stereo[:,1]+=r
        if alive: remaining_motifs.append(motif)
    active_motifs=remaining_motifs
    if random.random()<0.005: active_motifs.append(Motif(env.chords[env.current_chord_idx]))

    # Chord pads
    if not chord_pads: chord_pads = [ChordPad(env.chords[env.current_chord_idx])]
    for pad in chord_pads:
        l,r = pad.generate(t)
        stereo[:,0]+=l; stereo[:,1]+=r

    # Reverb & filter
    stereo = evolving_reverb(stereo)
    stereo = my_filter.process(stereo)

    # Normalize
    stereo = stereo / max(1, np.max(np.abs(stereo)))
    outdata[:] = stereo

# ---------------- RUN ----------------
with sd.OutputStream(channels=2, callback=audio_callback, samplerate=sample_rate, blocksize=buffer_size):
    print("Full professional cinematic ambient engine running. Press Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")
