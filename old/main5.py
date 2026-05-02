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
        self.filter_cutoff = 1000
        self.reverb_decay = 0.5
        self.current_chord_idx = 0
        self.scale_types = ['major','minor','sus2','sus4']
        self.key_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
        self.last_mod_time = time.time()
        self.mod_interval = 40
        self.chords = [[261.63, 329.63, 392.00]]  # initial chord

    def adaptive_modulation(self, low, mid, high, motif_density):
        now = time.time()
        if now - self.last_mod_time < self.mod_interval: return
        self.last_mod_time = now
        tension = mid + high + motif_density*0.1
        if tension < 0.4: scale_type='major'
        elif tension<0.6: scale_type='sus2'
        elif tension<0.8: scale_type='sus4'
        else: scale_type='minor'
        root = random.choice(self.key_notes)
        if scale_type=='major': self.chords=[[root, root*5/4, root*3/2]]
        elif scale_type=='minor': self.chords=[[root, root*6/5, root*3/2]]
        elif scale_type=='sus2': self.chords=[[root, root*9/8, root*3/2]]
        elif scale_type=='sus4': self.chords=[[root, root*4/3, root*3/2]]

env = EnvControl()

# ---------------- GRAINS ----------------
class Grain:
    def __init__(self, melodic=True, size='medium'):
        self.melodic = melodic
        self.size = size
        self.amp = random.uniform(0.02,0.1)
        self.pan = random.uniform(-0.5,0.5)
        self.freq = random.uniform(220,880)
        # ADSR
        self.attack = random.uniform(0.01,0.2)
        self.decay = random.uniform(0.1,0.5)
        self.sustain = random.uniform(0.5,0.8)
        self.release = random.uniform(0.2,0.6)
        self.start_time = time.time()
    def generate(self, t, env):
        elapsed = time.time() - self.start_time
        if elapsed < self.attack:
            env_amp = self.amp*(elapsed/self.attack)
        elif elapsed < self.attack+self.decay:
            env_amp = self.amp*(1 - 0.5*((elapsed-self.attack)/self.decay))
        elif elapsed < self.attack+self.decay+self.sustain:
            env_amp = self.amp*self.sustain
        else:
            env_amp = self.amp*self.sustain*(1 - min((elapsed-self.attack-self.decay-self.sustain)/self.release,1))
        wave = env_amp*np.sin(2*np.pi*self.freq*t*(1 + random.uniform(-0.002,0.002)))
        left = np.sqrt(0.5*(1-self.pan))*wave
        right = np.sqrt(0.5*(1+self.pan))*wave
        return wave,left,right

# ---------------- EVENTS ----------------
class Event:
    def __init__(self, type='drone'):
        self.type = type
        self.amp = random.uniform(0.05,0.2)
        self.pan = random.uniform(-0.5,0.5)
        self.duration = random.uniform(0.2,0.8)
        self.start_time = time.time()
    def generate(self,t):
        elapsed = time.time()-self.start_time
        alive = elapsed<self.duration
        wave = self.amp*np.sin(2*np.pi*440*t*(1+random.uniform(-0.002,0.002)))*(alive)
        return wave, alive

# ---------------- MOTIFS ----------------
class Motif:
    def __init__(self,chord,length=4):
        self.chord = chord
        self.notes=[random.choice(chord) for _ in range(length)]
        self.durations=[random.uniform(0.5,1.5) for _ in range(length)]
        self.amps=[random.uniform(0.05,0.2) for _ in range(length)]
        self.pans=[random.uniform(-0.5,0.5) for _ in range(length)]
        self.start_time=time.time()
        self.note_index=0
        self.phase=0
    def generate(self,t):
        elapsed = time.time()-self.start_time
        duration_cumsum=np.cumsum(self.durations)
        while self.note_index<len(duration_cumsum) and elapsed>duration_cumsum[self.note_index]:
            self.note_index+=1; self.phase=0
        if self.note_index>=len(self.notes):
            return np.zeros_like(t),np.zeros_like(t),False
        freq=self.notes[self.note_index]*(1+random.uniform(-0.003,0.003))
        amp=self.amps[self.note_index]
        pan=self.pans[self.note_index]
        wave=amp*np.sin(2*np.pi*freq*t+self.phase)
        self.phase+=2*np.pi*freq*len(t)/sample_rate
        left=np.sqrt(0.5*(1-pan))*wave
        right=np.sqrt(0.5*(1+pan))*wave
        return left,right,True
    def mutate(self):
        self.notes=[note*(1+random.uniform(-0.02,0.02)) if random.random()<0.2 else note for note in self.notes]
        self.durations=[d*(1+random.uniform(-0.1,0.1)) for d in self.durations]
        self.amps=[min(max(a*(1+random.uniform(-0.1,0.1)),0.01),0.3) for a in self.amps]
        self.pans=[np.clip(p+random.uniform(-0.05,0.05),-1,1) for p in self.pans]
        self.start_time=time.time(); self.note_index=0; self.phase=0
    def contextual_adapt(self,low,mid,high):
        if mid<0.3 and random.random()<0.2: self.amps=[min(a*1.1,0.3) for a in self.amps]
        if high>0.5 and random.random()<0.2: self.notes=[min(note*(1+random.uniform(0,0.02)), max(self.chord)) for note in self.notes]
        if low<0.2 and random.random()<0.2: self.durations=[d*(1+random.uniform(0,0.1)) for d in self.durations]
        self.pans=[np.clip(p+random.uniform(-0.02,0.02),-1,1) for p in self.pans]

# ---------------- STATE ----------------
grains=[]
active_events=[]
active_motifs=[]

# ---------------- SIMPLE REVERB ----------------
def apply_reverb(stereo):
    left = stereo[:,0]*0.7 + np.roll(stereo[:,0], 200)*0.3
    right = stereo[:,1]*0.7 + np.roll(stereo[:,1], -180)*0.3
    left = left*np.sqrt(0.5*(1 + np.sin(time.time()*0.1)*0.2))
    right = right*np.sqrt(0.5*(1 - np.sin(time.time()*0.1)*0.2))
    stereo[:,0] = left
    stereo[:,1] = right
    return stereo

# ---------------- AUDIO CALLBACK ----------------
def audio_callback(outdata, frames, time_info, status):
    global grains,active_events,active_motifs
    buffer = np.zeros(buffer_size)
    stereo = np.zeros((buffer_size,2))
    # GRAINS
    for grain in grains:
        wave,l,r = grain.generate(t,env)
        stereo[:,0]+=l; stereo[:,1]+=r; buffer+=wave
    # EVENTS
    remaining_events=[]
    for event in active_events:
        wave,alive = event.generate(t)
        stereo[:,0]+=wave*np.sqrt(0.5*(1-event.pan))
        stereo[:,1]+=wave*np.sqrt(0.5*(1+event.pan))
        if alive: remaining_events.append(event)
    active_events=remaining_events
    # MOTIFS
    remaining_motifs=[]
    for motif in active_motifs:
        l,r,alive=motif.generate(t)
        stereo[:,0]+=l; stereo[:,1]+=r
        if alive: remaining_motifs.append(motif)
    active_motifs=remaining_motifs
    # FFT
    spectrum = np.fft.rfft(buffer)
    freqs = np.fft.rfftfreq(len(buffer),1/sample_rate)
    mag = np.abs(spectrum)
    low = np.mean(mag[(freqs>=20)&(freqs<250)]) if np.any((freqs>=20)&(freqs<250)) else 0
    mid = np.mean(mag[(freqs>=250)&(freqs<2000)]) if np.any((freqs>=250)&(freqs<2000)) else 0
    high = np.mean(mag[(freqs>=2000)&(freqs<10000)]) if np.any((freqs>=2000)&(freqs<10000)) else 0
    # MOTIF CONTEXTUAL ADAPT
    for motif in active_motifs: motif.contextual_adapt(low,mid,high)
    # ENV ADAPTIVE MODULATION
    motif_density = len(active_motifs)/10.0
    env.adaptive_modulation(low,mid,high,motif_density)
    # CROSS-LAYER INTERACTIONS
    grain_density=len(grains)/30.0
    event_energy=sum([e.amp for e in active_events])
    for motif in active_motifs:
        for grain in grains: grain.amp*=1+0.02*np.random.rand()
    for event in active_events:
        if grain_density>0.8: event.amp*=(1+0.05*np.random.rand())
    for motif in active_motifs:
        if event_energy>0.5: motif.amps=[a*0.9 for a in motif.amps]
        elif event_energy<0.1: motif.amps=[min(a*1.05,0.3) for a in motif.amps]
    # RANDOM SPAWN
    if len(grains)<20: grains.append(Grain())
    if random.random()<0.005: active_motifs.append(Motif(env.chords[env.current_chord_idx]))
    if random.random()<0.01: active_events.append(Event(random.choice(['drone','shimmer','hit'])))
    # APPLY REVERB/SPATIALIZATION
    stereo = apply_reverb(stereo)
    outdata[:]=stereo

# ---------------- RUN ----------------
with sd.OutputStream(channels=2, callback=audio_callback, samplerate=sample_rate, blocksize=buffer_size):
    print("Cinematic ambient engine running. Press Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")
