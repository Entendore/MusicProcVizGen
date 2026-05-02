# -------------------- Motif Class with Evolution --------------------
class Motif:
    def __init__(self, chord, length=4):
        self.chord = chord
        self.notes = [random.choice(chord) for _ in range(length)]
        self.durations = [random.uniform(0.5,1.5) for _ in range(length)]
        self.amps = [random.uniform(0.05,0.2) for _ in range(length)]
        self.pans = [random.uniform(-0.5,0.5) for _ in range(length)]
        self.start_time = time.time()
        self.note_index = 0
        self.phase = 0

    def generate(self, t):
        elapsed = time.time() - self.start_time
        duration_cumsum = np.cumsum(self.durations)
        while self.note_index < len(duration_cumsum) and elapsed > duration_cumsum[self.note_index]:
            self.note_index += 1
            self.phase = 0
        if self.note_index >= len(self.notes):
            return np.zeros_like(t), np.zeros_like(t), False  # motif finished
        # Generate waveform for current note with slight detuning
        freq = self.notes[self.note_index] * (1 + random.uniform(-0.005,0.005))
        amp = self.amps[self.note_index]
        pan = self.pans[self.note_index]
        wave = amp * np.sin(2*np.pi*freq*t + self.phase)
        self.phase += 2*np.pi*freq*len(t)/sample_rate
        left = np.sqrt(0.5*(1-pan)) * wave
        right = np.sqrt(0.5*(1+pan)) * wave
        return left, right, True

    def mutate(self):
        # Small chance to change notes
        self.notes = [note*(1+random.uniform(-0.02,0.02)) if random.random()<0.2 else note for note in self.notes]
        # Small chance to adjust durations
        self.durations = [d*(1+random.uniform(-0.1,0.1)) for d in self.durations]
        # Slight amplitude evolution
        self.amps = [min(max(a*(1+random.uniform(-0.1,0.1)),0.01),0.3) for a in self.amps]
        # Slight pan evolution
        self.pans = [np.clip(p+random.uniform(-0.05,0.05),-1,1) for p in self.pans]
        self.start_time = time.time()
        self.note_index = 0
        self.phase = 0

# -------------------- Motif Manager Update --------------------
active_motifs = []

def spawn_motif():
    chord = chords[env.current_chord_idx]
    if active_motifs and random.random()<0.5:
        # Reuse last motif with mutation
        motif = active_motifs[-1]
        new_motif = Motif(chord)
        new_motif.notes = motif.notes.copy()
        new_motif.durations = motif.durations.copy()
        new_motif.amps = motif.amps.copy()
        new_motif.pans = motif.pans.copy()
        new_motif.mutate()
        active_motifs.append(new_motif)
    else:
        # Spawn new motif
        motif = Motif(chord)
        active_motifs.append(motif)

# -------------------- Integrate in Audio Callback --------------------
# Inside the audio_callback function, after generating events:
# Randomly spawn motifs
if random.random() < 0.005:
    spawn_motif()
remaining_motifs = []
for motif in active_motifs:
    left_wave, right_wave, alive = motif.generate(t)
    stereo[:,0] += left_wave
    stereo[:,1] += right_wave
    if alive:
        remaining_motifs.append(motif)
active_motifs = remaining_motifs
