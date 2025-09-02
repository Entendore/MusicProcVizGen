import numpy as np
import sounddevice as sd
import random, time

# ---------------- SETTINGS ----------------
sample_rate = 44100
buffer_size = 1024
t = np.arange(buffer_size) / sample_rate

# ---------------- ENVIRONMENT CONTROL ----------------
class EnvControl:
    def __init__(self):
        self.key_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
        self.chords = [[261.63, 329.63, 392.00]]
        self.current_chord_idx = 0
        self.last_mod_time = time.time()
        self.mod_interval = 40
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

# ---------------- PAN LFO ----------------
def pan_lfo(base_pan=0.0, speed=0.01, depth=0.2):
    return np.clip(base_pan + np.sin(time.time()*speed)*depth, -1.0, 1.0)

# ---------------- MICROTONAL DRIFT & HARMONICS ----------------
def microtonal_drift(freq, drift_speed=0.01, drift_depth=0.002):
    return freq * (1 + np.sin(time.time()*drift_speed)*drift_depth)

def add_harmonics(freq, base_amp, t, num_harmonics=3):
    signal = np.zeros_like(t)
    for i in range(1, num_harmonics+1):
        detune = freq*(i + random.uniform(-0.01,0.01))
        amp = base_amp / (i*1.5)
        signal += amp * np.sin(2*np.pi*detune*t)
    return signal

# ---------------- GRAINS ----------------
class Grain:
    def __init__(self):
        self.freq = random.uniform(220,880)
        self.amp = random.uniform(0.02,0.08)
        self.base_pan = random.uniform(-0.5,0.5)
        self.phase = 0.0
        self.attack = random.uniform(0.05,0.2)
        self.decay = random.uniform(0.2,0.5)
        self.sustain = random.uniform(0.5,0.8)
        self.release = random.uniform(0.5,1.0)
        self.start_time = time.time()
    def generate(self,t):
        elapsed = time.time() - self.start_time
        if elapsed<self.attack:
            env_amp = self.amp*(elapsed/self.attack)
        elif elapsed<self.attack+self.decay:
            env_amp = self.amp*(1-0.5*((elapsed-self.attack)/self.decay))
        elif elapsed<self.attack+self.decay+self.sustain:
            env_amp = self.amp*self.sustain
        else:
            env_amp = self.amp*self.sustain*(1 - min((elapsed-self.attack-self.decay-self.sustain)/self.release,1))
        freq = microtonal_drift(self.freq, drift_speed=0.005, drift_depth=0.002)
        wave = add_harmonics(freq, env_amp, t)
        pan = pan_lfo(self.base_pan, speed=0.01, depth=0.2)
        left = np.sqrt(0.5*(1-pan))*wave
        right = np.sqrt(0.5*(1+pan))*wave
        return left,right

# ---------------- EVENTS ----------------
class Event:
    def __init__(self, type='drone'):
        self.type = type
        self.amp = random.uniform(0.05,0.15)
        self.base_pan = random.uniform(-0.5,0.5)
        self.duration = random.uniform(0.5,1.5)
        self.start_time = time.time()
        self.freq = 440
        self.phase = 0.0
    def generate(self,t):
        elapsed = time.time()-self.start_time
        alive = elapsed<self.duration
        freq = microtonal_drift(self.freq, drift_speed=0.01, drift_depth=0.001)
        wave = add_harmonics(freq, self.amp, t)*(alive)
        pan = pan_lfo(self.base_pan, speed=0.01, depth=0.2)
        left = wave*np.sqrt(0.5*(1-pan))
        right = wave*np.sqrt(0.5*(1+pan))
        return left,right,alive

# ---------------- MOTIFS ----------------
class Motif:
    def __init__(self,chord,length=4):
        self.chord = chord
        self.notes=[random.choice(chord) for _ in range(length)]
        self.durations=[random.uniform(0.8,1.5) for _ in range(length)]
        self.amps=[random.uniform(0.05,0.15) for _ in range(length)]
        self.base_pans=[random.uniform(-0.5,0.5) for _ in range(length)]
        self.start_time=time.time()
        self.note_index=0
        self.phase=0.0
    def generate(self,t):
        elapsed = time.time()-self.start_time
        duration_cumsum=np.cumsum(self.durations)
        while self.note_index<len(duration_cumsum) and elapsed>duration_cumsum[self.note_index]:
            self.note_index+=1; self.phase=0
        if self.note_index>=len(self.notes):
            return np.zeros_like(t),np.zeros_like(t),False
        freq=microtonal_drift(self.notes[self.note_index], drift_speed=0.005, drift_depth=0.002)
        amp=self.amps[self.note_index]
        wave=add_harmonics(freq, amp, t)
        pan = pan_lfo(self.base_pans[self.note_index], speed=0.01, depth=0.2)
        left=np.sqrt(0.5*(1-pan))*wave
        right=np.sqrt(0.5*(1+pan))*wave
        return left,right,True
    def contextual_adapt(self,low,mid,high):
        if mid<0.3 and random.random()<0.2: self.amps=[min(a*1.05,0.2) for a in self.amps]
        if high>0.5 and random.random()<0.2: self.notes=[note*(1+random.uniform(0,0.01)) for note in self.notes]
        if low<0.2 and random.random()<0.2: self.durations=[d*(1+random.uniform(0,0.05)) for d in self.durations]
        self.base_pans=[np.clip(p+random.uniform(-0.01,0.01),-0.5,0.5) for p in self.base_pans]

# ---------------- DRONES ----------------
class Drone:
    def __init__(self, root_freq):
        self.base_freq = root_freq
        self.amp = random.uniform(0.02,0.06)
        self.base_pan = random.uniform(-0.3,0.3)
        self.mod_speed = random.uniform(0.01,0.05)
    def generate(self,t):
        freq = microtonal_drift(self.base_freq, drift_speed=self.mod_speed, drift_depth=0.003)
        wave = add_harmonics(freq, self.amp, t)
        pan = pan_lfo(self.base_pan, speed=0.005, depth=0.2)
        left = np.sqrt(0.5*(1-pan))*wave
        right = np.sqrt(0.5*(1+pan))*wave
        return left,right

# ---------------- CHORD PADS ----------------
class ChordPad:
    def __init__(self,chord):
        self.chord = chord
        self.phase = [0.0]*len(chord)
        self.amp = 0.05
        self.base_pans = [random.uniform(-0.3,0.3) for _ in chord]
    def generate(self,t):
        left = np.zeros_like(t)
        right = np.zeros_like(t)
        swell = (np.sin(time.time()*0.05)+1)/2 * 0.05
        for i,freq in enumerate(self.chord):
            freq = microtonal_drift(freq, drift_speed=0.003, drift_depth=0.002)
            wave = add_harmonics(freq, self.amp+swell, t)
            pan = pan_lfo(self.base_pans[i], speed=0.005, depth=0.2)
            left += np.sqrt(0.5*(1-pan))*wave
            right += np.sqrt(0.5*(1+pan))*wave
        return left,right

# ---------------- BACKGROUND TEXTURE ----------------
class BackgroundTexture:
    def __init__(self, amp=0.005):
        self.amp = amp
    def generate(self,t):
        noise = (np.random.rand(len(t)) - 0.5) * 2 * self.amp
        smooth_noise = np.convolve(noise, np.ones(10)/10, mode='same')
        return smooth_noise, smooth_noise

# ---------------- SHIMMER EFFECT ----------------
class Shimmer:
    def __init__(self):
        self.freq = random.uniform(4000,8000)
        self.amp = random.uniform(0.002,0.005)
        self.base_pan = random.uniform(-0.5,0.5)
        self.duration = random.uniform(0.1,0.3)
        self.start_time = time.time()
        self.phase = 0.0
    def generate(self, t):
        elapsed = time.time()-self.start_time
        alive = elapsed<self.duration
        freq = microtonal_drift(self.freq, drift_speed=0.02, drift_depth=0.01)
        wave = self.amp*np.sin(2*np.pi*freq*t) * (alive)
        pan = pan_lfo(self.base_pan, speed=0.02, depth=0.5)
        left = wave*np.sqrt(0.5*(1-pan))
        right = wave*np.sqrt(0.5*(1+pan))
        return left,right,alive

# ---------------- ENVIRONMENTAL TRIGGERS ----------------
class EnvTrigger:
    def __init__(self, type='bell'):
        self.type = type
        self.start_time = time.time()
        self.duration = random.uniform(0.3,1.5)
        self.amp = random.uniform(0.01,0.08)
        self.base_pan = random.uniform(-0.7,0.7)
        self.phase = 0.0
        self.freqs = [random.uniform(400,800), random.uniform(900,1200), random.uniform(1300,1800)] if type=='bell' else []
    def generate(self, t):
        elapsed = time.time() - self.start_time
        alive = elapsed < self.duration
        if not alive: return np.zeros_like(t), np.zeros_like(t), False
        if self.type == 'bell':
            wave = sum(self.amp/len(self.freqs)*np.sin(2*np.pi*f*t + self.phase) for f in self.freqs)
        elif self.type == 'wind':
            wave = (np.random.rand(len(t))-0.5)*self.amp
            wave = np.convolve(wave, np.ones(20)/20, mode='same')
        elif self.type == 'hit':
            wave = self.amp*np.random.rand(len(t)) * np.exp(-5*elapsed)
        else:
            wave = np.zeros_like(t)
        self.phase += 2*np.pi*np.mean(self.freqs)*len(t)/sample_rate if self.freqs else 0
        pan = pan_lfo(self.base_pan, speed=0.01, depth=0.3)
        left = wave*np.sqrt(0.5*(1-pan))
        right = wave*np.sqrt(0.5*(1+pan))
        return left,right,alive

def spawn_spectral_trigger(low, mid, high):
    r = random.random()
    if r < high*0.02: return EnvTrigger('bell')
    if r < low*0.01: return EnvTrigger('wind')
    if r < mid*0.015: return EnvTrigger('hit')
    return None

# ---------------- FILTER ----------------
class Filter:
    def __init__(self, cutoff=2000):
        self.cutoff = cutoff
        self.prev_left = 0
        self.prev_right = 0
    def process(self, stereo):
        alpha = self.cutoff / (self.cutoff + sample_rate/2)
        left = alpha*stereo[:,0] + (1-alpha)*self.prev_left
        right = alpha*stereo[:,1] + (1-alpha)*self.prev_right
        self.prev_left = left[-1]
        self.prev_right = right[-1]
        stereo[:,0] = left
        stereo[:,1] = right
        return stereo

# ---------------- REVERB ----------------
def apply_reverb(stereo):
    left = stereo[:,0]*0.7 + np.roll(stereo[:,0], 200)*0.3
    right = stereo[:,1]*0.7 + np.roll(stereo[:,1], -180)*0.3
    stereo[:,0] = left
    stereo[:,1] = right
    return stereo

# ---------------- STATE ----------------
grains=[]
active_events=[]
active_motifs=[]
drones = [Drone(random.choice(env.key_notes)) for _ in range(3)]
chord_pads = [ChordPad(env.chords[env.current_chord_idx]) for _ in range(2)]
bg_texture = BackgroundTexture()
active_shimmers = []
active_env_triggers = []
my_filter = Filter(2000)

# ---------------- AUDIO CALLBACK ----------------
def audio_callback(outdata, frames, time_info, status):
    global grains, active_events, active_motifs, drones, chord_pads, my_filter, bg_texture, active_shimmers, active_env_triggers
    stereo = np.zeros((buffer_size,2))
    buffer = np.zeros(buffer_size)

    # --- GRAINS ---
    for grain in grains:
        l,r = grain.generate(t)
        stereo[:,0]+=l; stereo[:,1]+=r; buffer+=l+r

    # --- EVENTS ---
    remaining_events=[]
    for event in active_events:
        l,r,alive = event.generate(t)
        stereo[:,0]+=l; stereo[:,1]+=r
        if alive: remaining_events.append(event)
    active_events=remaining_events

    # --- MOTIFS ---
    remaining_motifs=[]
    for motif in active_motifs:
        l,r,alive = motif.generate(t)
        stereo[:,0]+=l; stereo[:,1]+=r
        if alive: remaining_motifs.append(motif)
    active_motifs=remaining_motifs

    # --- DRONES ---
    for drone in drones:
        l,r = drone.generate(t)
        stereo[:,0]+=l; stereo[:,1]+=r

    # --- CHORD PADS ---
    for pad in chord_pads:
        l,r = pad.generate(t)
        stereo[:,0]+=l; stereo[:,1]+=r

    # --- BACKGROUND TEXTURE ---
    l,r = bg_texture.generate(t)
    stereo[:,0]+=l; stereo[:,1]+=r

    # --- SHIMMERS ---
    remaining_shimmers=[]
    for shimmer in active_shimmers:
        l,r,alive = shimmer.generate(t)
        stereo[:,0]+=l; stereo[:,1]+=r
        if alive: remaining_shimmers.append(shimmer)
    active_shimmers = remaining_shimmers
    if random.random() < 0.01: active_shimmers.append(Shimmer())

    # --- FFT ---
    spectrum = np.fft.rfft(buffer)
    freqs = np.fft.rfftfreq(len(buffer),1/sample_rate)
    mag = np.abs(spectrum)
    low = np.mean(mag[(freqs>=20)&(freqs<250)]) if np.any((freqs>=20)&(freqs<250)) else 0
    mid = np.mean(mag[(freqs>=250)&(freqs<2000)]) if np.any((freqs>=250)&(freqs<2000)) else 0
    high = np.mean(mag[(freqs>=2000)&(freqs<10000)]) if np.any((freqs>=2000)&(freqs<10000)) else 0

    # --- MOTIF ADAPTATION ---
    for motif in active_motifs: motif.contextual_adapt(low,mid,high)

    # --- ENV MODULATION ---
    motif_density = len(active_motifs)/10.0
    env.adaptive_modulation(low,mid,high,motif_density)

    # --- SPAWN GRAINS & MOTIFS ---
    if len(grains)<20: grains.append(Grain())
    if random.random()<0.005: active_motifs.append(Motif(env.chords[env.current_chord_idx]))
    if random.random()<0.01: active_events.append(Event(random.choice(['drone','shimmer','hit'])))

    # --- SPECTRAL TRIGGERS ---
    new_trigger = spawn_spectral_trigger(low, mid, high)
    if new_trigger: active_env_triggers.append(new_trigger)

    remaining_triggers=[]
    for trigger in active_env_triggers:
        # optional amplitude scaling based on FFT
        if trigger.type=='bell': trigger.amp *= min(1.0, high*10)
        if trigger.type=='wind': trigger.amp *= min(1.0, low*10)
        if trigger.type=='hit': trigger.amp *= min(1.0, mid*10)
        l,r,alive = trigger.generate(t)
        stereo[:,0]+=l; stereo[:,1]+=r
        if alive: remaining_triggers.append(trigger)
    active_env_triggers = remaining_triggers

    # --- REVERB ---
    stereo = apply_reverb(stereo)

    # --- FILTER ---
    my_filter.cutoff = 2000 + 1000*np.sin(time.time()*0.01)
    stereo = my_filter.process(stereo)

    # --- NORMALIZE ---
    stereo = stereo / max(1, np.max(np.abs(stereo)))

    # --- OUTPUT ---
    outdata[:] = stereo

# ---------------- RUN ----------------
with sd.OutputStream(channels=2, callback=audio_callback, samplerate=sample_rate, blocksize=buffer_size):
    print("Fully procedural cinematic ambient engine running. Press Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")
