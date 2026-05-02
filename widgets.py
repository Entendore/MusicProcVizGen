# widgets.py
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QPen, QPainterPath, QBrush, QRadialGradient

class SpectrumWidget(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.setMinimumHeight(180)
        self.vu_l = 0.0
        self.vu_r = 0.0
        self.mode = 0 # 0: Spectrum, 1: Waveform, 2: Circular
        
    def set_mode(self, mode_index):
        self.mode = mode_index
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Background
        p.fillRect(self.rect(), QColor(15, 15, 15))
        
        # Grid Lines
        pen = QPen(QColor(35, 35, 35), 1)
        p.setPen(pen)
        for i in range(0, w, 40): p.drawLine(i, 0, i, h)
        for i in range(0, h, 40): p.drawLine(0, i, w, i)

        # --- VU Meters (Always visible on right side) ---
        self.vu_l = self.vu_l * 0.8 + self.engine.vu_levels[0] * 0.2
        self.vu_r = self.vu_r * 0.8 + self.engine.vu_levels[1] * 0.2
        
        vu_w = 15
        vu_gap = 5
        vu_x = w - vu_w - 15
        
        # Left VU
        h_l = int(self.vu_l * h * 3.0)
        grad = QLinearGradient(0, h, 0, 0)
        grad.setColorAt(0.0, QColor(0, 200, 0))
        grad.setColorAt(0.8, QColor(255, 255, 0))
        grad.setColorAt(1.0, QColor(255, 0, 0))
        p.fillRect(vu_x, h - h_l, vu_w, h_l, grad)
        
        # Right VU
        h_r = int(self.vu_r * h * 3.0)
        p.fillRect(vu_x + vu_w + vu_gap, h - h_r, vu_w, h_r, grad)
        
        # Border for VU
        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.drawRect(vu_x, 0, vu_w, h)
        p.drawRect(vu_x + vu_w + vu_gap, 0, vu_w, h)

        # --- Visualization Modes ---
        viz_w = w - 60 # Leave space for VU
        
        if self.mode == 0:
            self.draw_spectrum(p, viz_w, h)
        elif self.mode == 1:
            self.draw_waveform(p, viz_w, h)
        elif self.mode == 2:
            self.draw_circular(p, viz_w, h)

    def draw_spectrum(self, p, w, h):
        data = self.engine.spectrum_data
        bar_count = 64
        bar_w = w / bar_count
        
        for i in range(bar_count):
            val = np.log1p(data[i]) * 25.0
            bar_h = min(h-5, val)
            
            grad = QLinearGradient(0, h, 0, 0)
            grad.setColorAt(0.1, QColor(42, 130, 218))
            grad.setColorAt(0.9, QColor(130, 220, 255))
            
            x = int(i * (bar_w + 1))
            y = int(h - bar_h)
            
            path = QPainterPath()
            path.addRoundedRect(float(x), float(y), float(bar_w-1), float(bar_h), 2.0, 2.0)
            p.fillPath(path, grad)

    def draw_waveform(self, p, w, h):
        data = self.engine.waveform_data
        if len(data) == 0: return
        
        mid_y = h / 2
        pen = QPen(QColor(0, 200, 255), 2)
        p.setPen(pen)
        
        path = QPainterPath()
        path.moveTo(0, mid_y)
        
        step = w / len(data)
        for i, sample in enumerate(data):
            # sample is -1.0 to 1.0
            y = mid_y + (sample * (h * 0.8))
            path.lineTo(i * step, y)
            
        p.drawPath(path)

    def draw_circular(self, p, w, h):
        data = self.engine.spectrum_data
        center_x = w / 2
        center_y = h / 2
        radius = min(w, h) * 0.3
        
        # Draw radial lines
        for i, val in enumerate(data):
            angle = (i / 64) * 360
            length = radius + (np.log1p(val) * 10.0)
            
            rad = np.deg2rad(angle)
            
            x1 = center_x + radius * np.cos(rad)
            y1 = center_y + radius * np.sin(rad)
            x2 = center_x + length * np.cos(rad)
            y2 = center_y + length * np.sin(rad)
            
            # Color based on frequency
            hue = (i / 64) * 360
            color = QColor.fromHsv(int(hue), 200, 255)
            p.setPen(QPen(color, 2))
            p.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Center circle
        p.setBrush(QBrush(QColor(20, 20, 20)))
        p.setPen(QPen(QColor(100, 100, 100), 2))
        p.drawEllipse(int(center_x - radius/2), int(center_y - radius/2), int(radius), int(radius))