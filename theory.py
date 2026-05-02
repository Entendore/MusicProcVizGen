# theory.py
import random

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
        "Hijaz (Arabic)":   [0, 1, 4, 5, 7, 8, 10],
        "Double Harmonic":  [0, 1, 4, 5, 7, 8, 11],
        "Hungarian Minor":  [0, 2, 3, 6, 7, 8, 11],
        "Japanese (In-Sen)":[0, 1, 5, 7, 8],
        "Raga Bhairavi":    [0, 1, 4, 5, 7, 8, 10],
        "Maqam Kurd":       [0, 2, 3, 5, 7, 8, 10],
        "Chromatic":        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }
    
    ROOTS = {
        "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
        "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
        "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88
    }

    # Standard chord progressions (degrees: I, IV, V, vi etc.)
    PROGRESSIONS = [
        [0, 3, 4, 4],    # I - IV - V - V
        [0, 5, 3, 4],    # I - vi - IV - V
        [0, 4, 5, 3],    # I - V - vi - IV (Axis of Awesome)
        [0, 0, 3, 4],    # Bluesy I - I - IV - V
        [5, 4, 0, 3],    # vi - V - I - IV
        [0, 2, 3, 4]     # I - iii - IV - V
    ]

    @staticmethod
    def get_scale_freqs(root_note, scale_name):
        root = MusicTheory.ROOTS[root_note]
        intervals = MusicTheory.SCALES.get(scale_name, MusicTheory.SCALES["Major"])
        return [root * (2**(i/12.0)) for i in intervals]

    @staticmethod
    def merge_scales(scale_a, scale_b, blend_factor):
        if not scale_a: return scale_b
        if not scale_b: return scale_a
        
        if blend_factor < 0.01: return scale_a
        if blend_factor > 0.99: return scale_b
        
        # Union of intervals for a "Fusion" scale
        merged = list(set(scale_a + scale_b))
        merged.sort()
        return merged

    @staticmethod
    def build_chord_from_scale(scale_freqs, degree_index, octave_shift=0, chord_type='triad'):
        """
        Builds a chord (list of frequencies) from a scale.
        degree_index: 0=Root, 1=2nd, 2=3rd...
        Returns frequencies for the chord tones.
        """
        if not scale_freqs: return []
        
        # Create an extended scale list to handle wrapping octaves
        # We repeat the scale frequencies shifted by octaves
        extended_scale = []
        for oct_shift in range(-1, 3): # cover bass to high
            for f in scale_freqs:
                extended_scale.append(f * (2 ** oct_shift))
        
        # Map degree index to extended index
        # Scale usually has 7 notes. Degree 0 is index 0. Degree 1 is index 1.
        # Bass note starts at octave -1 usually for synth bass?
        # Let's standardise: Chord root is at octave 0 (the scale_freqs provided are usually octave 0)
        
        # For simplicity, let's assume scale_freqs is one octave.
        # We need to pick Root, 3rd, 5th (and 7th).
        # Indices in scale: Root=0, 3rd=2, 5th=4, 7th=6.
        
        try:
            # Adjust indices for extended scale access
            base_idx = degree_index % len(scale_freqs)
            
            # Triad: 1, 3, 5
            idx_root = base_idx
            idx_3rd = (base_idx + 2) % len(scale_freqs)
            idx_5th = (base_idx + 4) % len(scale_freqs)
            
            freq_root = scale_freqs[idx_root] * (2 ** octave_shift)
            freq_3rd = scale_freqs[idx_3rd] * (2 ** octave_shift)
            freq_5th = scale_freqs[idx_5th] * (2 ** octave_shift)
            
            chord = [freq_root, freq_3rd, freq_5th]
            
            if chord_type == 'seventh':
                idx_7th = (base_idx + 6) % len(scale_freqs)
                chord.append(scale_freqs[idx_7th] * (2 ** octave_shift))
                
            return chord
        except IndexError:
            return []