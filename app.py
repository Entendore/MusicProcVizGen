# app.py
import sys
import sounddevice as sd
import threading
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QComboBox, QPushButton, 
                               QGridLayout, QGroupBox, QSpinBox, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPalette, QColor

from config import SAMPLE_RATE, BUFFER_SIZE, DTYPE
from engine import AudioEngine
from widgets import SpectrumWidget

class MainWindow(QMainWindow):
    recording_signal = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("World Music Procedural Engine Pro")
        self.resize(1000, 780)
        
        self.setup_theme()
        
        self.engine = AudioEngine()
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE,
            channels=2, dtype=DTYPE, callback=self.engine.audio_callback
        )
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(15)
        
        # --- Header ---
        header_layout = QHBoxLayout()
        
        header_layout.addWidget(QLabel("Visual Mode:"))
        self.viz_mode_combo = QComboBox()
        self.viz_mode_combo.addItems(["Spectrum", "Waveform", "Circular"])
        self.viz_mode_combo.currentIndexChanged.connect(self.change_viz_mode)
        self.style_widget(self.viz_mode_combo)
        header_layout.addWidget(self.viz_mode_combo)
        
        header_layout.addStretch()
        
        self.status_lbl = QLabel("● STANDBY")
        self.status_lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.status_lbl.setStyleSheet("color: #555;")
        header_layout.addWidget(self.status_lbl)
        main_layout.addLayout(header_layout)

        # --- Visualizer ---
        self.viz = SpectrumWidget(self.engine)
        self.viz.setMinimumHeight(200)
        self.viz.setStyleSheet("background-color: #121212; border-radius: 8px; border: 1px solid #333;")
        main_layout.addWidget(self.viz)
        
        # --- Controls Container ---
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        def create_group(title, width=None):
            group = QGroupBox(title)
            group.setFont(QFont("Segoe UI", 11, QFont.Bold))
            group.setStyleSheet("""
                QGroupBox {
                    border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 12px;
                    padding-top: 10px; background-color: #1e1e1e;
                }
                QGroupBox::title {
                    subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #AAA;
                }
                QLabel { color: #CCC; }
            """)
            layout = QGridLayout()
            layout.setSpacing(12)
            layout.setContentsMargins(12, 18, 12, 12)
            group.setLayout(layout)
            if width: group.setFixedWidth(width)
            return group, layout

        # Column 1: Theory & Composition
        theory_group, theory_grid = create_group("Music Theory & Composition")
        row = 0
        theory_grid.addWidget(QLabel("Root Note:"), row, 0)
        self.root_box = self.create_combo()
        self.root_box.currentIndexChanged.connect(lambda i: self.engine.cmd_queue.put(('root_idx', i)))
        theory_grid.addWidget(self.root_box, row, 1)
        
        row += 1
        theory_grid.addWidget(QLabel("Scale 1 (Main):"), row, 0)
        self.scale_box = self.create_combo()
        self.scale_box.addItems(self.engine.scale_names)
        self.scale_box.currentIndexChanged.connect(lambda i: self.engine.cmd_queue.put(('scale_idx', i)))
        theory_grid.addWidget(self.scale_box, row, 1)
        
        row += 1
        theory_grid.addWidget(QLabel("Scale 2 (Fusion):"), row, 0)
        self.scale_box_2 = self.create_combo()
        self.scale_box_2.addItems(self.engine.scale_names)
        self.scale_box_2.setCurrentIndex(8)
        self.scale_box_2.currentIndexChanged.connect(lambda i: self.engine.cmd_queue.put(('scale_idx_2', i)))
        theory_grid.addWidget(self.scale_box_2, row, 1)

        row += 1
        self.add_slider(theory_grid, "Scale Fusion:", 'fusion', row, 0, 100, 0, is_float=True)
        
        row += 1
        theory_grid.addWidget(QLabel("Algorithm:"), row, 0)
        self.algo_combo = self.create_combo()
        self.algo_combo.addItems(["Weighted Random", "Markov Flow", "Euclidean Rhythm"])
        self.algo_combo.currentIndexChanged.connect(lambda i: self.engine.cmd_queue.put(('algo_idx', i)))
        theory_grid.addWidget(self.algo_combo, row, 1)
        
        controls_layout.addWidget(theory_group)

        # Column 2: Sound Design
        sound_group, sound_grid = create_group("Sound Design")
        row = 0
        self.add_slider(sound_grid, "Brightness:", 'brightness', row, 0, 100, 70, is_float=True)
        row += 1
        self.add_slider(sound_grid, "Space (Reverb):", 'reverb_mix', row, 0, 100, 40, is_float=True)
        row += 1
        self.add_slider(sound_grid, "Modulation:", 'modulation', row, 0, 100, 30, is_float=True)
        row += 1
        self.add_slider(sound_grid, "Master Volume:", 'master_amp', row, 0, 100, 75, is_float=True)
        
        controls_layout.addWidget(sound_group)
        
        # Column 3: Rhythm
        rhythm_group, rhythm_grid = create_group("Rhythm & Seq")
        row = 0
        self.add_slider(rhythm_grid, "Tempo:", 'tempo', row, 60, 180, 100, suffix=" BPM")
        row += 1
        self.add_slider(rhythm_grid, "Groove (Swing):", 'groove', row, 0, 100, 20, is_float=True)
        row += 1
        self.add_slider(rhythm_grid, "Melody Density:", 'melody_dist', row, 0, 100, 60, is_float=True)
        row += 1
        self.add_slider(rhythm_grid, "Chord Freq:", 'chord_freq', row, 0, 100, 50, is_float=True)
        
        controls_layout.addWidget(rhythm_group)
        
        main_layout.addLayout(controls_layout)
        
        # --- Transport & Recording ---
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(20)
        
        self.engine_btn = QPushButton("▶ START ENGINE")
        self.engine_btn.setCheckable(True)
        self.engine_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.engine_btn.setMinimumHeight(50)
        self.engine_btn.setStyleSheet("""
            QPushButton {
                background-color: #222; color: #0f0; border: 2px solid #444; border-radius: 6px;
            }
            QPushButton:checked {
                background-color: #4a1515; color: #f44; border: 2px solid #a33;
            }
        """)
        self.engine_btn.clicked.connect(self.toggle_engine)
        bottom_layout.addWidget(self.engine_btn, 3)

        export_group, export_grid = create_group("Recording & Export", width=450)
        
        self.rec_btn = QPushButton("● REC")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setEnabled(False)
        self.rec_btn.setStyleSheet("""
            QPushButton { background-color: #333; color: #888; border-radius: 4px; font-weight: bold; padding: 5px; }
            QPushButton:enabled { color: #fff; }
            QPushButton:checked { background-color: #d00; color: white; }
        """)
        self.rec_btn.clicked.connect(self.toggle_recording)
        export_grid.addWidget(self.rec_btn, 0, 0)

        self.timed_rec_btn = QPushButton("Record Duration")
        self.timed_rec_btn.setEnabled(False)
        self.timed_rec_btn.setStyleSheet("QPushButton { background-color: #333; color: #aaa; border-radius: 4px; padding: 5px; }")
        self.timed_rec_btn.clicked.connect(self.start_timed_recording)
        export_grid.addWidget(self.timed_rec_btn, 0, 1)

        export_grid.addWidget(QLabel("Seconds:"), 0, 2)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(5, 300)
        self.duration_spin.setValue(30)
        self.duration_spin.setStyleSheet("background: #333; color: white; border-radius: 3px; padding: 3px;")
        export_grid.addWidget(self.duration_spin, 0, 3)
        
        bottom_layout.addWidget(export_group)
        main_layout.addLayout(bottom_layout)
        
        # Connections
        self.recording_signal.connect(self.engine.set_recording)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(33)

    def style_widget(self, widget):
        widget.setStyleSheet("""
            QComboBox { background-color: #333; border: 1px solid #555; border-radius: 4px; padding: 4px; color: white;}
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #333; selection-background-color: #555; color: white; }
        """)

    def create_combo(self):
        combo = QComboBox()
        self.style_widget(combo)
        return combo

    def change_viz_mode(self, index):
        self.viz.set_mode(index)

    def setup_theme(self):
        app = QApplication.instance()
        app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(25, 25, 25))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.Highlight, QColor(40, 120, 200))
        app.setPalette(palette)

    def add_slider(self, layout, label_text, param_name, row, min_val, max_val, default_val, is_float=False, suffix=""):
        layout.addWidget(QLabel(label_text), row, 0)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        
        val_label = QLabel(f"{default_val}{suffix}")
        val_label.setMinimumWidth(40)
        val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val_label.setStyleSheet("color: #888; font-size: 10px;")
        
        if not hasattr(self, 'slider_labels'): self.slider_labels = {}
        self.slider_labels[param_name] = (val_label, suffix, is_float)
        
        def on_change(v):
            txt = f"{v}{suffix}" if not is_float else f"{v/100.0:.2f}"
            val_label.setText(txt)
            val = v if not is_float else v/100.0
            self.engine.cmd_queue.put((param_name, val))
                
        slider.valueChanged.connect(on_change)
        layout.addWidget(slider, row, 1)
        layout.addWidget(val_label, row, 2)

    def toggle_engine(self, state):
        if state:
            self.stream.start()
            self.engine_btn.setText("■ STOP ENGINE")
            self.rec_btn.setEnabled(True)
            self.timed_rec_btn.setEnabled(True)
            self.status_lbl.setText("● RUNNING")
            self.status_lbl.setStyleSheet("color: #0f0; font-weight: bold;")
        else:
            self.stream.stop()
            self.engine_btn.setText("▶ START ENGINE")
            self.rec_btn.setEnabled(False)
            self.timed_rec_btn.setEnabled(False)
            self.status_lbl.setText("● STANDBY")
            self.status_lbl.setStyleSheet("color: #555;")
            if self.rec_btn.isChecked():
                self.rec_btn.setChecked(False)

    def toggle_recording(self, checked, filename=None):
        if checked:
            if filename is None:
                filename, _ = QFileDialog.getSaveFileName(self, "Save Audio", "output.wav", "WAV Files (*.wav)")
                if not filename:
                    self.rec_btn.setChecked(False)
                    return
            
            self.status_lbl.setText("● RECORDING")
            self.status_lbl.setStyleSheet("color: #f00; font-weight: bold;")
            self.recording_signal.emit(True)
            self.engine.save_filename = filename
        else:
            self.recording_signal.emit(False)
            self.status_lbl.setText("● SAVING...")
            self.status_lbl.setStyleSheet("color: #fa0;")
            threading.Thread(target=self.save_thread, daemon=True).start()

    def start_timed_recording(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Timed Recording", "timed_output.wav", "WAV Files (*.wav)")
        if not filename: return
        
        duration = self.duration_spin.value()
        self.rec_btn.setEnabled(False)
        self.timed_rec_btn.setEnabled(False)
        
        self.rec_btn.setChecked(True)
        self.toggle_recording(True, filename)
        QTimer.singleShot(duration * 1000, self.auto_stop_recording)

    def auto_stop_recording(self):
        if self.rec_btn.isChecked():
            self.rec_btn.setChecked(False)
            self.toggle_recording(False)
            self.rec_btn.setEnabled(True)
            self.timed_rec_btn.setEnabled(True)

    def save_thread(self):
        self.engine.save_audio()
        QTimer.singleShot(100, lambda: self.status_lbl.setText("● RUNNING"))
        QTimer.singleShot(100, lambda: self.status_lbl.setStyleSheet("color: #0f0; font-weight: bold;"))
        QTimer.singleShot(100, lambda: self.rec_btn.setEnabled(True))
        QTimer.singleShot(100, lambda: self.timed_rec_btn.setEnabled(True))

    def update_ui(self):
        self.viz.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())