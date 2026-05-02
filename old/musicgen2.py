import random
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
import os
import tempfile
import math
import struct
import wave

class AdvancedMusicGeneratorApp(App):
    def build(self):
        self.title = "Advanced Procedural Music Generator"
        self.root = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Style selection
        style_layout = BoxLayout(size_hint_y=None, height=50)
        style_label = Label(text="Style:", size_hint_x=None, width=100)
        self.style_spinner = Spinner(
            text='Classical',
            values=('Classical', 'Jazz', 'Pop', 'Rock', 'Blues', 'Electronic'),
            size_hint_x=None,
            width=200
        )
        style_layout.add_widget(style_label)
        style_layout.add_widget(self.style_spinner)
        self.root.add_widget(style_layout)
        
        # Key selection
        key_layout = BoxLayout(size_hint_y=None, height=50)
        key_label = Label(text="Key:", size_hint_x=None, width=100)
        self.key_spinner = Spinner(
            text='C Major',
            values=('C Major', 'G Major', 'D Minor', 'A Minor', 'F Major', 'E Minor'),
            size_hint_x=None,
            width=200
        )
        key_layout.add_widget(key_label)
        key_layout.add_widget(self.key_spinner)
        self.root.add_widget(key_layout)
        
        # Tempo control
        tempo_layout = BoxLayout(size_hint_y=None, height=50)
        tempo_label = Label(text="Tempo:", size_hint_x=None, width=100)
        self.tempo_slider = Slider(min=60, max=200, value=120, size_hint_x=0.7)
        self.tempo_value = Label(text="120 BPM", size_hint_x=None, width=80)
        self.tempo_slider.bind(value=self.on_tempo_change)
        tempo_layout.add_widget(tempo_label)
        tempo_layout.add_widget(self.tempo_slider)
        tempo_layout.add_widget(self.tempo_value)
        self.root.add_widget(tempo_layout)
        
        # Melody length
        length_layout = BoxLayout(size_hint_y=None, height=50)
        length_label = Label(text="Length:", size_hint_x=None, width=100)
        self.length_slider = Slider(min=8, max=32, value=16, step=1, size_hint_x=0.7)
        self.length_value = Label(text="16 notes", size_hint_x=None, width=80)
        self.length_slider.bind(value=self.on_length_change)
        length_layout.add_widget(length_label)
        length_layout.add_widget(self.length_slider)
        length_layout.add_widget(self.length_value)
        self.root.add_widget(length_layout)
        
        # Harmony options
        harmony_layout = BoxLayout(size_hint_y=None, height=40)
        harmony_label = Label(text="Add Harmony:", size_hint_x=None, width=120)
        self.harmony_checkbox = CheckBox(size_hint_x=None, width=30)
        harmony_layout.add_widget(harmony_label)
        harmony_layout.add_widget(self.harmony_checkbox)
        self.root.add_widget(harmony_layout)
        
        # Rhythm options
        rhythm_layout = BoxLayout(size_hint_y=None, height=40)
        rhythm_label = Label(text="Complex Rhythm:", size_hint_x=None, width=120)
        self.rhythm_checkbox = CheckBox(size_hint_x=None, width=30)
        rhythm_layout.add_widget(rhythm_label)
        rhythm_layout.add_widget(self.rhythm_checkbox)
        self.root.add_widget(rhythm_layout)
        
        # Controls
        controls_layout = BoxLayout(size_hint_y=None, height=50)
        self.generate_btn = Button(text="Generate Melody")
        self.generate_btn.bind(on_press=self.generate_melody)
        self.play_btn = Button(text="Play", disabled=True)
        self.play_btn.bind(on_press=self.play_melody)
        self.stop_btn = Button(text="Stop")
        self.stop_btn.bind(on_press=self.stop_melody)
        controls_layout.add_widget(self.generate_btn)
        controls_layout.add_widget(self.play_btn)
        controls_layout.add_widget(self.stop_btn)
        self.root.add_widget(controls_layout)
        
        # Status display
        self.status_label = Label(text="Select options and click 'Generate Melody'")
        self.root.add_widget(self.status_label)
        
        # Note display area
        self.notes_layout = GridLayout(cols=8, size_hint_y=None, height=100)
        self.root.add_widget(self.notes_layout)
        
        # Audio setup
        self.sound = None
        self.melody_notes = []
        self.harmony_notes = []
        self.tempo = 120
        self.melody_length = 16
        
        return self.root

    def on_tempo_change(self, instance, value):
        self.tempo = int(value)
        self.tempo_value.text = f"{self.tempo} BPM"

    def on_length_change(self, instance, value):
        self.melody_length = int(value)
        self.length_value.text = f"{self.melody_length} notes"

    def generate_melody(self, instance):
        # Get selections
        style = self.style_spinner.text
        key = self.key_spinner.text
        add_harmony = self.harmony_checkbox.active
        complex_rhythm = self.rhythm_checkbox.active
        
        # Clear previous notes display
        self.notes_layout.clear_widgets()
        
        # Generate notes based on style and key
        self.melody_notes = self.create_melody(
            style, key, self.melody_length, complex_rhythm
        )
        
        # Generate harmony if requested
        if add_harmony:
            self.harmony_notes = self.create_harmony(self.melody_notes, style)
        else:
            self.harmony_notes = []
        
        # Display generated melody
        for note in self.melody_notes:
            note_label = Label(text=self.note_to_name(note), size_hint_y=None, height=30)
            self.notes_layout.add_widget(note_label)
        
        self.status_label.text = f"Generated {len(self.melody_notes)} notes in {style} style"
        self.play_btn.disabled = False

    def create_melody(self, style, key, length, complex_rhythm):
        # Define scales
        scales = {
            'C Major': [0, 2, 4, 5, 7, 9, 11],
            'G Major': [7, 9, 11, 0, 2, 4, 6],
            'D Minor': [2, 4, 5, 7, 9, 10, 0],
            'A Minor': [9, 11, 0, 2, 4, 5, 7],
            'F Major': [5, 7, 9, 10, 0, 2, 4],
            'E Minor': [4, 6, 7, 9, 11, 0, 2]
        }
        
        # Get base scale
        base_notes = scales.get(key, scales['C Major'])
        root = base_notes[0]
        
        # Style-specific patterns
        patterns = {
            'Classical': [0, 2, 4, 3, 2, 1, 2, 4, 5, 3, 4, 2, 1, 0, 2, 4],
            'Jazz': [0, 3, 1, 4, 2, 5, 3, 6, 4, 1, 5, 2, 6, 3, 0, 4],
            'Pop': [0, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 3, 2, 1, 0],
            'Rock': [0, 4, 2, 5, 1, 3, 6, 2, 0, 5, 3, 1, 4, 2, 6, 3],
            'Blues': [0, 3, 5, 6, 3, 0, 4, 2, 5, 3, 6, 0, 3, 5, 6, 3],
            'Electronic': [0, 2, 7, 5, 3, 6, 1, 4, 0, 5, 2, 7, 4, 1, 6, 3]
        }
        
        pattern = patterns.get(style, patterns['Classical'])
        
        # Generate melody
        melody = []
        current_octave = 4
        last_note = root
        
        for i in range(length):
            # Use pattern to select note
            pattern_index = i % len(pattern)
            scale_degree = pattern[pattern_index]
            
            # Get note from scale
            note = base_notes[scale_degree % len(base_notes)]
            
            # Adjust octave based on melodic movement
            if note < last_note and last_note - note > 6:
                current_octave += 1
            elif note > last_note and note - last_note > 6:
                current_octave -= 1
                
            # Apply style-specific modifications
            if style == 'Blues' and random.random() < 0.3:
                # Add blue notes (flattened 5th)
                note = (base_notes[4] - 1) % 12
            elif style == 'Jazz' and random.random() < 0.2:
                # Add chromatic approach notes
                note = (note + (1 if random.random() < 0.5 else -1)) % 12
            elif style == 'Electronic' and random.random() < 0.4:
                # Add octave jumps
                if random.random() < 0.5:
                    note = (note + 12) % 12
                else:
                    note = (note - 12) % 12
            
            # Create MIDI note (C4 = 60)
            midi_note = 12 * (current_octave + 1) + note
            melody.append(midi_note)
            last_note = note
            
        return melody

    def create_harmony(self, melody, style):
        harmony = []
        for note in melody:
            # Simple harmony: third below or fifth above
            if style in ['Classical', 'Pop']:
                # Third below
                harmony_note = note - 4
            elif style in ['Jazz', 'Blues']:
                # Fifth above
                harmony_note = note + 7
            elif style == 'Rock':
                # Octave below
                harmony_note = note - 12
            else:
                # Fifth below
                harmony_note = note - 7
                
            harmony.append(harmony_note)
        return harmony

    def note_to_name(self, midi_note):
        # Convert MIDI note to name
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = midi_note // 12 - 1
        note_name = notes[midi_note % 12]
        return f"{note_name}{octave}"

    def play_melody(self, instance):
        if not self.melody_notes:
            return
            
        # Create temporary WAV file
        temp_file = self.create_wav_file(
            self.melody_notes, 
            self.harmony_notes, 
            self.tempo
        )
        if temp_file:
            # Load and play sound
            if self.sound:
                self.sound.stop()
            self.sound = SoundLoader.load(temp_file)
            if self.sound:
                self.sound.play()
                # Clean up after playing
                Clock.schedule_once(
                    lambda dt: self.cleanup_file(temp_file), 
                    self.sound.length
                )
            else:
                self.status_label.text = "Error: Could not load audio"
        else:
            self.status_label.text = "Error: Could not create audio file"

    def stop_melody(self, instance):
        if self.sound:
            self.sound.stop()
            self.sound = None
        self.status_label.text = "Playback stopped"

    def create_wav_file(self, melody_notes, harmony_notes, tempo):
        try:
            # Parameters
            sample_rate = 44100
            beat_duration = 60.0 / tempo  # seconds per beat
            volume = 0.3
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            wf = wave.open(temp_file.name, 'wb')
            wf.setnchannels(2)  # Stereo
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            
            # Generate audio data
            for i, note in enumerate(melody_notes):
                # Note duration (quarter notes for now)
                duration = beat_duration
                
                # Get frequencies
                melody_freq = 440 * (2 ** ((note - 69) / 12))
                harmony_freq = 0
                if harmony_notes and i < len(harmony_notes):
                    harmony_freq = 440 * (2 ** ((harmony_notes[i] - 69) / 12))
                
                samples = int(duration * sample_rate)
                
                for j in range(samples):
                    # Generate melody wave
                    melody_sample = volume * math.sin(2 * math.pi * melody_freq * j / sample_rate)
                    
                    # Generate harmony wave
                    if harmony_freq > 0:
                        harmony_sample = (volume * 0.7) * math.sin(2 * math.pi * harmony_freq * j / sample_rate)
                    else:
                        harmony_sample = 0
                    
                    # Apply envelope
                    envelope = 1.0
                    if j < samples * 0.1:
                        envelope = j / (samples * 0.1)  # Attack
                    elif j > samples * 0.9:
                        envelope = 1 - ((j - samples * 0.9) / (samples * 0.1))  # Release
                    
                    melody_sample *= envelope
                    harmony_sample *= envelope
                    
                    # Mix channels (melody in left, harmony in right)
                    left_sample = int(melody_sample * 32767)
                    right_sample = int(harmony_sample * 32767)
                    
                    # Write stereo frames
                    wf.writeframes(struct.pack('<hh', left_sample, right_sample))
            
            wf.close()
            return temp_file.name
        except Exception as e:
            self.status_label.text = f"Error: {str(e)}"
            return None

    def cleanup_file(self, filepath):
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
        except:
            pass

if __name__ == '__main__':
    AdvancedMusicGeneratorApp().run()