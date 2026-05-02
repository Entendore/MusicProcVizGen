#!/usr/bin/env python3
"""
GPU-accelerated, audio-reactive “electric-sheep” visualiser
Usage:
    python electric_sheep.py                    # auto-pick first .mp3/.wav in ./input_files
    python electric_sheep.py -f song.flac       # use a specific file
    python electric_sheep.py --fullscreen       # real fullscreen
All dependencies:  pip install numpy sounddevice vispy scipy pydub
(On Windows you also need ffmpeg.exe in PATH for pydub.)
"""

import os, glob, argparse, queue, numpy as np, sounddevice as sd
from vispy import gloo, app, scene
from scipy.signal import stft
from pydub import AudioSegment

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
INPUT_FOLDER = "input_files"
DEFAULT_W, DEFAULT_H, DEFAULT_FPS = 1920, 1080, 60

# --------------------------------------------------------------------------- #
# Audio loader (MP3, WAV, FLAC, M4A, …)
# --------------------------------------------------------------------------- #
def load_audio(path: str):
    audio = AudioSegment.from_file(path)        # pydub auto-detects format
    audio = audio.set_channels(1)               # force mono
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.max(np.abs(samples)) + 1e-8   # -1..1
    return audio.frame_rate, samples

# --------------------------------------------------------------------------- #
# Shaders
# --------------------------------------------------------------------------- #
VERT_SHADER = """
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

FRAG_SHADER = """
uniform float u_time;
uniform vec2  u_resolution;
uniform sampler2D u_tex;
uniform float u_bass;
uniform float u_mid;
uniform float u_treble;
varying vec2 v_uv;

float rand(vec2 co){ return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453); }
float noise(vec2 st){
    vec2 i = floor(st), f = fract(st);
    float a = rand(i), b = rand(i + vec2(1.0, 0.0));
    float c = rand(i + vec2(0.0, 1.0)), d = rand(i + vec2(1.0, 1.0));
    vec2 u = f*f*(3.0 - 2.0*f);
    return mix(a, b, u.x) + (c - a)*u.y*(1.0 - u.x) + (d - b)*u.x*u.y;
}
vec3 hsv2rgb(vec3 c){
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
void main(){
    vec2 uv = v_uv;
    vec3 prev = texture2D(u_tex, uv).rgb;

    // warp
    vec2 warp = uv + 0.02 * vec2(noise(uv*5.0 + u_time*0.1), noise(uv*5.0 - u_time*0.1));

    // particles
    float particles = 0.0;
    for(int i=0;i<3;i++){
        vec2 p = vec2(noise(uv*10.0 + float(i) + u_time*0.5),
                      noise(uv*10.0 + float(i) - u_time*0.3));
        particles += smoothstep(0.02, 0.0, length(uv - p));
    }

    float rd = noise(warp*10.0 + u_time*0.2);
    float bass = u_bass*0.5, mid = u_mid*0.5, treble = u_treble*0.5;

    vec3 col = prev * 0.95;                 // feedback trail
    col += vec3(rd*bass, rd*mid, rd*treble);
    col += vec3(particles*0.3, particles*0.2, particles*0.4);

    float h = mod(u_time*0.05 + treble, 1.0);
    vec3 hsv = vec3(h, 1.0, clamp(col.r + col.g + col.b, 0.0, 1.0));
    gl_FragColor = vec4(hsv2rgb(hsv), 1.0);
}
"""

# --------------------------------------------------------------------------- #
# Visualiser
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Visualiser  (fixed version)
# --------------------------------------------------------------------------- #
class ElectricSheepGPU:
    def __init__(self, audio_path: str, width=DEFAULT_W, height=DEFAULT_H, fps=DEFAULT_FPS,
                 fullscreen=False):
        self.w, self.h, self.fps = width, height, fps

        # load audio + STFT
        self.rate, self.data = load_audio(audio_path)
        n_fft, hop = 2048, 512
        f, t, Z = stft(self.data, fs=self.rate, nperseg=n_fft, noverlap=n_fft-hop)
        self.mag = np.abs(Z)
        self.freqs = f
        self.bass_idx = np.where((f >= 20) & (f <= 250))[0]
        self.mid_idx  = np.where((f > 250) & (f <= 4000))[0]
        self.treble_idx = np.where((f > 4000) & (f <= 16000))[0]

        # vispy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', size=(self.w, self.h),
                                        show=True, fullscreen=fullscreen, always_on_top=True)
        self.canvas.events.key_press.connect(self.on_key)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(rect=(-1, -1, 2, 2))

        # quad covering screen
        verts = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = verts
        self.program['u_resolution'] = (self.w, self.h)
        self.program['u_time'] = 0.0
        self.program['u_bass'] = 0.0
        self.program['u_mid']  = 0.0
        self.program['u_treble'] = 0.0

        # feedback texture
        self.tex = gloo.Texture2D((np.random.rand(self.h, self.w, 3) * 0.1).astype(np.float32))
        self.fbo = gloo.FrameBuffer(self.tex)
        self.program['u_tex'] = self.fbo.color_buffer

        # timing
        self.frame = 0
        self.spp = self.rate / self.fps          # samples per frame

    # -------------------------------------------------------------- #
    def on_key(self, e):
        if e.key == 'Escape':
            sd.stop()
            app.quit()

    # -------------------------------------------------------------- #
    def update(self, ev):
        # audio indices
        idx = int(self.frame * self.spp)
        idx = min(idx, self.mag.shape[1] - 1)

        # band energies
        bass  = np.mean(self.mag[self.bass_idx, idx])
        mid   = np.mean(self.mag[self.mid_idx, idx])
        treble= np.mean(self.mag[self.treble_idx, idx])
        mx = max(bass, mid, treble, 1e-8)

        # uniforms
        self.program['u_bass']   = bass / mx
        self.program['u_mid']    = mid / mx
        self.program['u_treble'] = treble / mx
        self.program['u_time']   = self.frame * 0.03

        # render to fbo  (FIX 1:  activate() not bind() )
        with self.fbo:
            gloo.set_viewport(0, 0, self.w, self.h)
            gloo.clear()
            self.program.draw('triangle_strip')

        # feedback
        self.program['u_tex'] = self.fbo.color_buffer
        self.frame += 1

    # -------------------------------------------------------------- #
    def run(self):
        # PortAudio callback – always deliver exactly `frames` samples
        def callback(outdata: np.ndarray, frames: int, time, status):
            # outdata shape == (frames, 1)
            start = int(self.frame * self.spp)
            stop  = start + frames
            if start >= len(self.data):
                outdata.fill(0)
                return
            avail = self.data[start:stop]
            if len(avail) < frames:            # pad end with zeros
                outdata[:len(avail), 0] = avail
                outdata[len(avail):, 0] = 0.0
            else:
                outdata[:, 0] = avail

        # start audio + vispy timer
        stream = sd.OutputStream(samplerate=self.rate, channels=1,
                                 callback=callback, finished_callback=app.quit)
        with stream:
            timer = app.Timer(interval=1/self.fps, connect=self.update, start=True)
            app.run()

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def find_audio(folder):
    exts = ('*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return files[0] if files else None

def main():
    parser = argparse.ArgumentParser(description="Electric Sheep Audio Visualiser")
    parser.add_argument("-f", "--file", help="path to audio file")
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--height",type=int, default=DEFAULT_H)
    parser.add_argument("--fps",   type=int, default=DEFAULT_FPS)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    if args.file:
        audio_file = args.file
    else:
        audio_file = find_audio(INPUT_FOLDER)
        if audio_file is None:
            raise SystemExit(f"No audio file found in {INPUT_FOLDER}")

    print(f"Loading: {audio_file}")
    ElectricSheepGPU(audio_file, args.width, args.height, args.fps,
                     fullscreen=args.fullscreen).run()

if __name__ == "__main__":
    main()