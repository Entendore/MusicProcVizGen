import numpy as np
import sounddevice as sd
import time
import random
from scipy.signal import butter, lfilter

# -------------------- Audio Settings --------------------
sample_rate = 44100
buffer_size = 1024

# -------------------- Reverb --------------------
reverb_buffer = np.zeros(44100)
reverb_index = 0
reverb_decay = 0.4

# -------------------- Time Vector --------------------
t = np.arange(buffer_size) / sample_rate

# -------------------- Chord Progressions --------------------
chords = [
    [261.63, 329.63, 392.00],  # C major
    [293.66, 369.99, 440.00],  # D minor
    [349.23, 440.00, 523.25],  # F major
    [392.00, 493.88, 587.33]   # G major
]

# -------------------- Environment Control --------------------
class EnvControl:
    def __init__(self):
        self.grain_count = 15
        self.amp_mod = 0.3
        self.freq_mod = 200
        self.lfo_speed = 0.01
        self.reverb_decay = 0.4
        self.event_prob = 0.01
        self.current_chord_idx = 0
        self.last_chord_time = time.time()
        self.chord_change_speed = 20  # seconds per chord
        self.filter_cutoff = 2000  # Hz

    def update(self, spectrum_energy):
        now = time.time()
        # Long-term parameter evolution
        self.amp_mod = 0.2 + 0.15*np.sin(now*0.005)
        self.freq_mod = 150 + 150*np.sin(now*0.003)
        self.lfo_speed = 0.005 + 0.015*np.sin(now*0.004)
        self.reverb_decay = 0.35 + 0.1*np.sin(now*0.002)
        self.filter_cutoff = 1000 + 1500*(0.5 + 0.5*np.sin(now*0.001))

        # Adaptive grain count & event probability
        if spectrum_energy < 0.1:
            self.grain_count = min(self.grain_count + 1, 35)
            self.event_prob = 0.03
        else:
            self.grain_count = max(self.grain_count - 1, 10)
            self.event_prob = 0.008

        # Chord progression
        if now - self.last_chord_time > self.chord_change_speed:
            self.current_chord_idx = (self.current_chord_idx + 1) % len(chords)
            self.last_chord_time = now

env = EnvControl()

# -------------------- Parametric Filter --------------------
def apply_filter(buffer, cutoff, sample_rate):
    b, a = butter(2, cutoff/(sample_rate/2), btype='low')
    return lfilter(b, a, buffer)

# -------------------- Grain Class --------------------
class Grain:
    def __init__(self, melodic=False, size='medium'):
        self.melodic = melodic
        self.size = size
        self.freq = random.uniform(50,400)
        self.amp = random.uniform(0.1,0.4)
        self.pan = random.uniform(-1,1)
        self.phase = 0
        self.lfo_speed = random.uniform(0.003,0.02)
        self.stereo_speed = random.uniform(0.0005, 0.002)
        self.vibrato_depth = random.uniform(0.1,0.5)
        self.tremolo_depth = random.uniform(0.1,0.3)

    def generate(self, t, env):
        # Micro-modulation
        drift = np.sin(2*np.pi*self.lfo_speed*time.time())*self.vibrato_depth*10
        amp_mod = 1 + np.sin(2*np.pi*self.lfo_speed*time.time())*self.tremolo_depth

        # Melodic grains align with chord tones
        if self.melodic:
            chord = chords[env.current_chord_idx]
            target_freq = random.choice(chord)
        else:
            target_freq = self.freq

        wave = amp_mod * env.amp_mod * self.amp * (
            np.sin(2*np.pi*(target_freq+drift)*t + self.phase) +
            0.5*np.sin(2*np.pi*2*(target_freq+drift)*t + self.phase)
        )
        self.phase += 2*np.pi*(target_freq+drift)*len(t)/sample_rate

        # Stereo movement
        pan = self.pan + np.sin(time.time()*self.stereo_speed)
        pan = np.clip(pan,-1,1)
        left = np.sqrt(0.5*(1-pan))
        right = np.sqrt(0.5*(1+pan))

        return wave, left, right

# -------------------- Event Class --------------------
class Event:
    def __init__(self, type):
        self.type = type
        self.start_time = time.time()
        self.duration = random.uniform(2,10)
        self.freq = random.uniform(200,1200)
        self.amp = random.uniform(0.05,0.3)
        self.phase = 0
        self.pan = random.uniform(-1,1)

    def generate(self, t):
        elapsed = time.time() - self.start_time
        envelope = max(0, np.sin(np.pi*(1-elapsed/self.duration)/2)**2)
        if self.type == "drone":
            wave = self.amp * envelope * np.sin(2*np.pi*self.freq*t + self.phase)
        elif self.type == "shimmer":
            wave = self.amp * envelope * (np.sin(2*np.pi*self.freq*t + self.phase)+0.5*np.sin(2*np.pi*2*self.freq*t + self.phase))
        elif self.type == "hit":
            wave = self.amp * envelope * (np.random.rand(len(t))*2-1)
        else:
            wave = np.zeros_like(t)
        self.phase += 2*np.pi*self.freq*len(t)/sample_rate
        return wave, envelope>0

# -------------------- Initialization --------------------
grains = [Grain(melodic=random.random()<0.3, size=random.choice(['small','medium','long'])) for _ in range(15)]
active_events = []

# -------------------- FFT & Spectral Feedback --------------------
def dynamic_fft_shape(buffer, env):
    spectrum = np.fft.rfft(buffer)
    spectrum_energy = np.mean(np.abs(spectrum))
    freqs = np.fft.rfftfreq(len(buffer),1/sample_rate)
    sweep = env.freq_mod
    spectrum *= 1/(1+((freqs/sweep)**2))
    spectrum += (np.random.rand(len(spectrum))-0.5)*0.005
    return np.fft.irfft(spectrum), spectrum_energy

# -------------------- Reverb --------------------
def apply_reverb(buffer):
    global reverb_buffer, reverb_index
    out = np.zeros_like(buffer)
    for i in range(len(buffer)):
        out[i] = buffer[i] + reverb_decay*reverb_buffer[reverb_index]
        reverb_buffer[reverb_index] = out[i]
        reverb_index = (reverb_index+1)%len(reverb_buffer)
    return out

# -------------------- Audio Callback --------------------
def audio_callback(outdata, frames, time_info, status):
    global grains, active_events, env
    buffer = np.zeros(buffer_size)
    stereo = np.zeros((buffer_size,2))

    # Generate grains
    for grain in grains:
        wave, left, right = grain.generate(t, env)
        stereo[:,0] += wave*left/len(grains)
        stereo[:,1] += wave*right/len(grains)
        buffer += wave

    # FFT & spectral feedback
    buffer, spectrum_energy = dynamic_fft_shape(buffer, env)
    env.update(spectrum_energy)

    # Adaptive grain count
    while len(grains) < env.grain_count:
        grains.append(Grain(melodic=random.random()<0.3, size=random.choice(['small','medium','long'])))
    while len(grains) > env.grain_count:
        grains.pop(random.randint(0,len(grains)-1))

    # Spawn events
    if random.random()<env.event_prob:
        event_type = random.choice(["drone","shimmer","hit"])
        active_events.append(Event(event_type))

    # Generate events
    remaining_events=[]
    for event in active_events:
        wave, alive = event.generate(t)
        stereo[:,0] += wave*np.sqrt(0.5*(1-event.pan))
        stereo[:,1] += wave*np.sqrt(0.5*(1+event.pan))
        if alive:
            remaining_events.append(event)
    active_events = remaining_events

    # Reverb & Filter
    stereo[:,0] = apply_reverb(stereo[:,0])
    stereo[:,1] = apply_reverb(stereo[:,1])
    stereo[:,0] = apply_filter(stereo[:,0], env.filter_cutoff, sample_rate)
    stereo[:,1] = apply_filter(stereo[:,1], env.filter_cutoff, sample_rate)

    outdata[:] = stereo

# -------------------- Playback --------------------
with sd.OutputStream(channels=2, callback=audio_callback, samplerate=sample_rate, blocksize=buffer_size):
    print("Playing fully integrated cinematic ambient engine. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping playback.")
