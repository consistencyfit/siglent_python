#!/usr/bin/env python
"""
Continuous piezo capture with DWT-based correlation analysis.
Uses Discrete Wavelet Transform for fast multiresolution analysis.
"""
import json
import socket
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import hilbert
from scipy.interpolate import CubicSpline
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import pyqtgraph.exporters

# Enable antialiasing for smoother lines (like matplotlib)
pg.setConfigOptions(antialias=True)

SCOPE_IP = '192.168.1.142'
PORT = 5025


def send_command(sock, cmd, wait=True):
    """Send a command to the scope, optionally waiting for completion using *OPC?."""
    sock.sendall((cmd + '\n').encode())
    if wait:
        sock.sendall(b'*OPC?\n')
        old_timeout = sock.gettimeout()
        sock.settimeout(5)
        response = b''
        try:
            while True:
                chunk = sock.recv(256)
                if not chunk:
                    break
                response += chunk
                if b'1' in response:
                    break
        except socket.timeout:
            pass
        finally:
            sock.settimeout(old_timeout)


def query(sock, cmd, timeout=2):
    """Send a query and return the response."""
    sock.sendall((cmd + '\n').encode())
    sock.settimeout(timeout)
    time.sleep(0.05)

    response = b''
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            response += chunk
            if response.endswith(b'\n'):
                break
    except socket.timeout:
        pass

    return response.decode('utf-8', errors='replace').strip()


def query_binary(sock, cmd, timeout=2):
    """Send a query and return binary response."""
    sock.sendall((cmd + '\n').encode())
    sock.settimeout(timeout)
    time.sleep(0.05)

    response = b''
    expected_length = None
    start_time = time.time()

    try:
        while True:
            if time.time() - start_time > timeout:
                break
            chunk = sock.recv(65536)
            if not chunk:
                break
            response += chunk

            if expected_length is None and b'#' in response[:50]:
                hash_pos = response.find(b'#')
                if len(response) > hash_pos + 2:
                    try:
                        num_digits = int(chr(response[hash_pos + 1]))
                        if num_digits > 0 and len(response) > hash_pos + 2 + num_digits:
                            length_str = response[hash_pos + 2:hash_pos + 2 + num_digits]
                            data_length = int(length_str)
                            expected_length = hash_pos + 2 + num_digits + data_length + 2
                    except (ValueError, IndexError):
                        pass

            if expected_length and len(response) >= expected_length:
                break
    except socket.timeout:
        pass

    return response


def get_waveform(sock, channel='C1', vdiv=None, offset=0.0, upsample=4, debug=False):
    """Get waveform data from the specified channel with optional upsampling."""
    if vdiv is None:
        vdiv_resp = query(sock, f'{channel}:VDIV?')
        try:
            vdiv = float(vdiv_resp.split()[-1].replace('V', ''))
        except:
            vdiv = 1.0

    data_raw = query_binary(sock, f'{channel}:WF? DAT2', timeout=2)

    if debug:
        print(f"  DEBUG {channel}: got {len(data_raw)} bytes")

    if b'#' not in data_raw:
        return np.array([]), 0

    hash_pos = data_raw.find(b'#')
    try:
        num_digits = int(chr(data_raw[hash_pos + 1]))
        length_str = data_raw[hash_pos + 2:hash_pos + 2 + num_digits]
        data_length = int(length_str)
        data_start = hash_pos + 2 + num_digits
        waveform_data = data_raw[data_start:data_start + data_length]
    except (ValueError, IndexError):
        return np.array([]), 0

    if len(waveform_data) > 2:
        waveform_data = waveform_data[:-2]

    values = np.frombuffer(waveform_data, dtype=np.int8)
    voltage = (values.astype(float) * vdiv / 25.0) - offset

    if upsample > 1 and len(voltage) > 10:
        x_orig = np.arange(len(voltage))
        x_new = np.linspace(0, len(voltage) - 1, len(voltage) * upsample)
        cs = CubicSpline(x_orig, voltage)
        voltage = cs(x_new)

    return voltage, vdiv


def compute_envelope(signal, smooth_window=51):
    """Compute the envelope of a signal using Hilbert transform with smoothing."""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    if smooth_window > 1 and len(envelope) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        envelope = np.convolve(envelope, kernel, mode='same')

    return envelope


def fft_correlate(a, b):
    """Fast cross-correlation using FFT."""
    n = len(a) + len(b) - 1
    n_fft = 1 << (n - 1).bit_length()
    fft_a = np.fft.fft(a, n_fft)
    fft_b = np.fft.fft(b, n_fft)
    corr = np.fft.ifft(fft_a * np.conj(fft_b)).real
    return corr[:n]


def find_onset(env, threshold_pct=0.1):
    """Find onset index where envelope rises above threshold before the peak."""
    peak_idx = np.argmax(env)
    peak_val = env[peak_idx]
    threshold = peak_val * threshold_pct

    for i in range(peak_idx, -1, -1):
        if env[i] < threshold:
            return i + 1
    return 0


def analyze_correlation(env1, env2):
    """Analyze correlation between two envelope signals."""
    min_len = min(len(env1), len(env2))
    env1, env2 = env1[:min_len], env2[:min_len]

    max_samples = 10000
    if min_len > max_samples:
        factor = min_len // max_samples
        env1_ds = env1[::factor]
        env2_ds = env2[::factor]
    else:
        env1_ds, env2_ds = env1, env2
        factor = 1

    env1_norm = env1_ds / (np.max(env1_ds) + 1e-10)
    env2_norm = env2_ds / (np.max(env2_ds) + 1e-10)

    pearson_env = np.corrcoef(env1_ds, env2_ds)[0, 1]

    ncc = fft_correlate(env1_norm, env2_norm)
    ncc_peak = np.max(ncc)
    ncc_lag = (np.argmax(ncc) - (len(env1_ds) - 1)) * factor

    peak1_idx = np.argmax(env1)
    peak2_idx = np.argmax(env2)
    peak_lag_samples = peak1_idx - peak2_idx

    onset1_idx = find_onset(env1, threshold_pct=0.1)
    onset2_idx = find_onset(env2, threshold_pct=0.1)
    onset_lag_samples = onset1_idx - onset2_idx

    peak1_amp = np.max(env1)
    peak2_amp = np.max(env2)
    amplitude_ratio = min(peak1_amp, peak2_amp) / (max(peak1_amp, peak2_amp) + 1e-10)

    cosine_sim = np.dot(env1_norm, env2_norm) / (np.linalg.norm(env1_norm) * np.linalg.norm(env2_norm) + 1e-10)

    peak1_significant = peak1_amp > 1.5 * np.mean(env1)
    peak2_significant = peak2_amp > 1.5 * np.mean(env2)
    both_significant = peak1_significant and peak2_significant

    onset_score = 1 - min(abs(onset_lag_samples), 100) / 100

    correlation_score = (
        0.25 * pearson_env +
        0.20 * cosine_sim +
        0.15 * amplitude_ratio +
        0.40 * onset_score
    )

    if not both_significant:
        classification = "NO_PEAK"
    elif correlation_score > 0.7:
        classification = "CORRELATED"
    elif correlation_score > 0.4:
        classification = "WEAKLY_CORRELATED"
    else:
        classification = "UNCORRELATED"

    return {
        'pearson_envelope': float(pearson_env),
        'ncc_peak': float(ncc_peak),
        'ncc_lag_samples': int(ncc_lag),
        'peak_lag_samples': int(peak_lag_samples),
        'onset_lag_samples': int(onset_lag_samples),
        'onset1_idx': int(onset1_idx),
        'onset2_idx': int(onset2_idx),
        'amplitude_ratio': float(amplitude_ratio),
        'cosine_similarity': float(cosine_sim),
        'correlation_score': float(correlation_score),
        'classification': classification,
    }


def compute_dwt(signal, wavelet='db4', max_level=None):
    """
    Compute Discrete Wavelet Transform.

    Returns:
        coeffs: list of coefficient arrays [cA_n, cD_n, cD_n-1, ..., cD_1]
        levels: number of decomposition levels
        wavelet_name: name of wavelet used
    """
    signal = signal - np.mean(signal)

    # Get the maximum safe level for this signal length
    safe_max = pywt.dwt_max_level(len(signal), wavelet)

    if max_level is None:
        max_level = min(safe_max, 10)
    else:
        # Cap at safe maximum to avoid boundary effect warnings
        max_level = min(max_level, safe_max)

    coeffs = pywt.wavedec(signal, wavelet, level=max_level)

    return coeffs, max_level, wavelet


def dwt_to_scalogram(coeffs, signal_len):
    """
    Convert DWT coefficients to a scalogram-like 2D representation.

    Each level is upsampled to match the original signal length for visualization.
    """
    n_levels = len(coeffs)
    scalogram = np.zeros((n_levels, signal_len))

    for i, coeff in enumerate(coeffs):
        # Upsample coefficients to signal length
        if len(coeff) < signal_len:
            # Repeat each coefficient to fill the signal length
            repeat_factor = signal_len // len(coeff)
            upsampled = np.repeat(coeff, repeat_factor)
            # Trim or pad to exact length
            if len(upsampled) < signal_len:
                upsampled = np.pad(upsampled, (0, signal_len - len(upsampled)))
            else:
                upsampled = upsampled[:signal_len]
        else:
            upsampled = coeff[:signal_len]

        scalogram[i, :] = np.abs(upsampled) ** 2

    return scalogram


def dwt_correlation(wf1, wf2, sample_rate, wavelet='db4', max_level=None):
    """
    Compute correlation between two signals using DWT.

    Returns dict with:
        - level_correlations: correlation at each decomposition level
        - energy_profile_corr: correlation of energy profiles
        - dwt_score: combined DWT-based correlation score
        - coeffs1, coeffs2: DWT coefficients for visualization
    """
    coeffs1, n_levels, wavelet_name = compute_dwt(wf1, wavelet, max_level)
    coeffs2, _, _ = compute_dwt(wf2, wavelet, max_level)

    # Compute correlation at each level
    level_correlations = []
    level_energies1 = []
    level_energies2 = []

    for c1, c2 in zip(coeffs1, coeffs2):
        min_len = min(len(c1), len(c2))
        c1, c2 = c1[:min_len], c2[:min_len]

        if np.std(c1) > 0 and np.std(c2) > 0:
            corr = np.corrcoef(c1, c2)[0, 1]
        else:
            corr = 0.0
        level_correlations.append(float(corr))

        level_energies1.append(np.sum(c1 ** 2))
        level_energies2.append(np.sum(c2 ** 2))

    # Energy distribution correlation
    level_energies1 = np.array(level_energies1)
    level_energies2 = np.array(level_energies2)
    level_energies1 = level_energies1 / (np.sum(level_energies1) + 1e-10)
    level_energies2 = level_energies2 / (np.sum(level_energies2) + 1e-10)

    if np.std(level_energies1) > 0 and np.std(level_energies2) > 0:
        energy_dist_corr = np.corrcoef(level_energies1, level_energies2)[0, 1]
    else:
        energy_dist_corr = 0.0

    # Find dominant level (most energy)
    dominant_level1 = int(np.argmax(level_energies1))
    dominant_level2 = int(np.argmax(level_energies2))

    # Compute scalograms for visualization
    scalogram1 = dwt_to_scalogram(coeffs1, len(wf1))
    scalogram2 = dwt_to_scalogram(coeffs2, len(wf2))

    # Average level correlation (weighted by energy)
    weights = (level_energies1 + level_energies2) / 2
    weighted_corr = np.sum(np.array(level_correlations) * weights)

    # Combined DWT score
    dwt_score = (
        0.40 * max(0, weighted_corr) +
        0.30 * max(0, energy_dist_corr) +
        0.30 * (1.0 if dominant_level1 == dominant_level2 else 0.5)
    )

    # Compute approximate frequency bands for each level
    freq_bands = []
    nyquist = sample_rate / 2
    for level in range(n_levels):
        if level == 0:
            # Approximation coefficients (lowest frequencies)
            f_low = 0
            f_high = nyquist / (2 ** n_levels)
        else:
            # Detail coefficients
            detail_level = n_levels - level
            f_low = nyquist / (2 ** (detail_level + 1))
            f_high = nyquist / (2 ** detail_level)
        freq_bands.append((f_low, f_high))

    return {
        'level_correlations': level_correlations,
        'energy_dist_corr': float(energy_dist_corr),
        'weighted_corr': float(weighted_corr),
        'dominant_level1': dominant_level1,
        'dominant_level2': dominant_level2,
        'dwt_score': float(dwt_score),
        'n_levels': n_levels,
        'wavelet': wavelet_name,
        'freq_bands': freq_bands,
        'scalogram1': scalogram1,
        'scalogram2': scalogram2,
        'coeffs1': coeffs1,
        'coeffs2': coeffs2,
    }


class WaveletViewer(QtWidgets.QDialog):
    """Dialog to display DWT decomposition levels."""

    def __init__(self, coeffs, freq_bands, wavelet_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'DWT Decomposition ({wavelet_name})')
        self.resize(1200, 800)

        n_levels = len(coeffs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Info label
        info = QtWidgets.QLabel(
            f'<b>{n_levels} decomposition levels using {wavelet_name} wavelet</b><br>'
            '<span style="color: cyan;">Level 0 = Approximation (lowest freq)</span> · '
            '<span style="color: yellow;">Higher levels = Details (higher freq)</span>'
        )
        info.setStyleSheet('font-size: 13px; padding: 5px;')
        layout.addWidget(info)

        # Graphics widget
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground('#1e1e1e')

        # Plot each level
        for i, (coeff, (f_low, f_high)) in enumerate(zip(coeffs, freq_bands)):
            if i == 0:
                title = f'Approx: {f_low:.0f}-{f_high:.0f} Hz ({len(coeff)} coeffs)'
                pen = pg.mkPen('c', width=1.5)
            else:
                title = f'Detail {i}: {f_low:.0f}-{f_high:.0f} Hz ({len(coeff)} coeffs)'
                pen = pg.mkPen('y', width=1)

            plot = self.graphics.addPlot(row=i, col=0, title=title)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.plot(coeff, pen=pen)

        layout.addWidget(self.graphics)

        # Close button
        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)


class SpinnerOverlay(QtWidgets.QWidget):
    """Semi-transparent overlay with spinning indicator and text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = "Loading..."
        self._angle = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self.hide()

    def start(self, text="Loading..."):
        self._text = text
        self._angle = 0
        self.show()
        self.raise_()
        self._timer.start(50)

    def stop(self):
        self._timer.stop()
        self.hide()

    def _rotate(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 150))

        center = self.rect().center()

        painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100), 4))
        arc_rect = QtCore.QRectF(center.x() - 30, center.y() - 30, 60, 60)
        painter.drawEllipse(arc_rect)

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 150, 255), 4))
        painter.drawArc(arc_rect, self._angle * 16, 90 * 16)

        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.setFont(QtGui.QFont("Arial", 12))
        text_rect = QtCore.QRectF(0, center.y() + 50, self.width(), 30)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self._text)


class PiezoCapture(QtWidgets.QMainWindow):
    def __init__(self, ip=SCOPE_IP):
        super().__init__()
        self.setWindowTitle('Piezo Capture - DWT Analysis')
        self.resize(1400, 900)

        # Settings
        self.ip = ip
        self.ch1_vdiv = 0.07
        self.ch2_vdiv = 0.07
        self.hdiv = 0.02
        self.trigger_level = 0.050
        self.capture_count = 0
        self.running = False

        # Decimation/interpolation settings
        self.decimate_enabled = True
        self.decimate_points = 14000
        self.interpolate_enabled = True
        self.upsample_factor = 4
        self.envelope_smooth = 201
        self.trigger_delay_ms = -40
        self.show_onset_markers = False
        self.current_time_array = None

        # DWT settings
        self.dwt_wavelet = 'db2'  # Shorter wavelet = better time resolution for transients
        self.dwt_max_level = 8
        self.dwt_target_samples = 10000  # Target samples for DWT (higher = better resolution)

        # Create captures directory
        self.captures_dir = Path('captures')
        self.captures_dir.mkdir(exist_ok=True)

        # Setup UI
        self._setup_ui()

        # Connect to scope
        self.sock = None
        self._connect_scope()

        # Polling timer
        self.poll_timer = QtCore.QTimer()
        self.poll_timer.timeout.connect(self._poll_trigger)

        # Signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        self.signal_timer = QtCore.QTimer()
        self.signal_timer.timeout.connect(lambda: None)
        self.signal_timer.start(100)

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Control bar
        controls = QtWidgets.QHBoxLayout()

        self.arm_btn = QtWidgets.QPushButton('Arm Trigger')
        self.arm_btn.clicked.connect(self._toggle_trigger)
        controls.addWidget(self.arm_btn)

        self.capture_btn = QtWidgets.QPushButton('Manual Capture')
        self.capture_btn.clicked.connect(self._manual_capture)
        controls.addWidget(self.capture_btn)

        controls.addSpacing(20)

        controls.addWidget(QtWidgets.QLabel('Captures:'))
        self.capture_label = QtWidgets.QLabel('0')
        self.capture_label.setMinimumWidth(50)
        controls.addWidget(self.capture_label)

        controls.addSpacing(20)

        # Decimation controls
        self.decimate_cb = QtWidgets.QCheckBox('Decimate')
        self.decimate_cb.setChecked(self.decimate_enabled)
        self.decimate_cb.toggled.connect(self._on_decimate_changed)
        controls.addWidget(self.decimate_cb)

        self.decimate_spin = QtWidgets.QSpinBox()
        self.decimate_spin.setRange(1000, 100000)
        self.decimate_spin.setSingleStep(1000)
        self.decimate_spin.setValue(self.decimate_points)
        self.decimate_spin.setSuffix(' pts')
        self.decimate_spin.valueChanged.connect(self._on_decimate_changed)
        controls.addWidget(self.decimate_spin)

        controls.addSpacing(10)

        # Interpolation controls
        self.interp_cb = QtWidgets.QCheckBox('Interpolate')
        self.interp_cb.setChecked(self.interpolate_enabled)
        self.interp_cb.toggled.connect(self._on_interp_changed)
        controls.addWidget(self.interp_cb)

        self.upsample_spin = QtWidgets.QSpinBox()
        self.upsample_spin.setRange(1, 8)
        self.upsample_spin.setValue(self.upsample_factor)
        self.upsample_spin.setSuffix('x')
        self.upsample_spin.valueChanged.connect(self._on_interp_changed)
        controls.addWidget(self.upsample_spin)

        controls.addSpacing(10)

        # Envelope smoothing
        controls.addWidget(QtWidgets.QLabel('Smooth:'))
        self.smooth_spin = QtWidgets.QSpinBox()
        self.smooth_spin.setRange(1, 1001)
        self.smooth_spin.setSingleStep(50)
        self.smooth_spin.setValue(self.envelope_smooth)
        self.smooth_spin.valueChanged.connect(lambda v: setattr(self, 'envelope_smooth', v))
        controls.addWidget(self.smooth_spin)

        # Trigger delay control
        controls.addWidget(QtWidgets.QLabel('Delay:'))
        self.delay_spin = QtWidgets.QSpinBox()
        self.delay_spin.setRange(-500, 500)
        self.delay_spin.setSingleStep(10)
        self.delay_spin.setValue(self.trigger_delay_ms)
        self.delay_spin.setSuffix(' ms')
        self.delay_spin.valueChanged.connect(self._on_delay_changed)
        controls.addWidget(self.delay_spin)

        # Trigger level control
        controls.addWidget(QtWidgets.QLabel('Trig:'))
        self.trig_spin = QtWidgets.QDoubleSpinBox()
        self.trig_spin.setRange(0.001, 1.0)
        self.trig_spin.setSingleStep(0.005)
        self.trig_spin.setDecimals(3)
        self.trig_spin.setValue(self.trigger_level)
        self.trig_spin.setSuffix(' V')
        self.trig_spin.valueChanged.connect(self._on_trigger_level_changed)
        controls.addWidget(self.trig_spin)

        controls.addSpacing(10)

        # Onset markers toggle
        self.onset_markers_cb = QtWidgets.QCheckBox('Onset')
        self.onset_markers_cb.setChecked(False)
        self.onset_markers_cb.stateChanged.connect(self._on_onset_markers_changed)
        controls.addWidget(self.onset_markers_cb)

        controls.addSpacing(10)

        # DWT wavelet selector
        controls.addWidget(QtWidgets.QLabel('Wavelet:'))
        self.wavelet_combo = QtWidgets.QComboBox()
        self.wavelet_combo.addItems(['haar', 'db2', 'db4', 'db8', 'sym4', 'sym8', 'coif4'])
        self.wavelet_combo.setCurrentText(self.dwt_wavelet)
        self.wavelet_combo.currentTextChanged.connect(lambda w: setattr(self, 'dwt_wavelet', w))
        controls.addWidget(self.wavelet_combo)

        # DWT levels
        controls.addWidget(QtWidgets.QLabel('Levels:'))
        self.levels_spin = QtWidgets.QSpinBox()
        self.levels_spin.setRange(3, 12)
        self.levels_spin.setValue(self.dwt_max_level)
        self.levels_spin.valueChanged.connect(lambda v: setattr(self, 'dwt_max_level', v))
        controls.addWidget(self.levels_spin)

        # DWT target samples
        controls.addWidget(QtWidgets.QLabel('Samples:'))
        self.dwt_samples_spin = QtWidgets.QSpinBox()
        self.dwt_samples_spin.setRange(2000, 50000)
        self.dwt_samples_spin.setSingleStep(1000)
        self.dwt_samples_spin.setValue(self.dwt_target_samples)
        self.dwt_samples_spin.valueChanged.connect(lambda v: setattr(self, 'dwt_target_samples', v))
        controls.addWidget(self.dwt_samples_spin)

        controls.addStretch()

        # Status label
        self.status_label = QtWidgets.QLabel('Ready')
        self.status_label.setStyleSheet('color: gray; font-weight: bold;')
        controls.addWidget(self.status_label)

        layout.addLayout(controls)

        # Correlation display panel
        corr_group = QtWidgets.QGroupBox("Correlation Analysis")
        corr_group.setStyleSheet('QGroupBox { font-weight: bold; }')
        corr_grid = QtWidgets.QGridLayout(corr_group)

        # Classification and score
        self.corr_class_label = QtWidgets.QLabel('--')
        self.corr_class_label.setStyleSheet('font-size: 18px; font-weight: bold;')
        corr_grid.addWidget(QtWidgets.QLabel('Classification:'), 0, 0)
        corr_grid.addWidget(self.corr_class_label, 0, 1)

        self.corr_score_label = QtWidgets.QLabel('--')
        self.corr_score_label.setStyleSheet('font-size: 18px; font-weight: bold;')
        corr_grid.addWidget(QtWidgets.QLabel('Score:'), 0, 2)
        corr_grid.addWidget(self.corr_score_label, 0, 3)

        # Correlation parameters
        self.pearson_env_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Pearson (env):'), 1, 0)
        corr_grid.addWidget(self.pearson_env_label, 1, 1)

        self.cosine_sim_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Cosine sim:'), 1, 2)
        corr_grid.addWidget(self.cosine_sim_label, 1, 3)

        self.onset_lag_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Onset lag:'), 1, 4)
        corr_grid.addWidget(self.onset_lag_label, 1, 5)

        self.amp_ratio_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Amp ratio:'), 2, 0)
        corr_grid.addWidget(self.amp_ratio_label, 2, 1)

        self.peak_lag_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Peak lag:'), 2, 2)
        corr_grid.addWidget(self.peak_lag_label, 2, 3)

        self.ncc_lag_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('NCC lag:'), 2, 4)
        corr_grid.addWidget(self.ncc_lag_label, 2, 5)

        # DWT metrics
        self.dwt_score_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('DWT score:'), 3, 0)
        corr_grid.addWidget(self.dwt_score_label, 3, 1)

        self.dwt_weighted_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Weighted corr:'), 3, 2)
        corr_grid.addWidget(self.dwt_weighted_label, 3, 3)

        self.energy_dist_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Energy dist:'), 3, 4)
        corr_grid.addWidget(self.energy_dist_label, 3, 5)

        # Dominant levels
        self.dom_level1_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('CH1 dom level:'), 4, 0)
        corr_grid.addWidget(self.dom_level1_label, 4, 1)

        self.dom_level2_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('CH2 dom level:'), 4, 2)
        corr_grid.addWidget(self.dom_level2_label, 4, 3)

        layout.addWidget(corr_group)

        # Plot area
        graphics_container = QtWidgets.QWidget()
        graphics_layout = QtWidgets.QVBoxLayout(graphics_container)
        graphics_layout.setContentsMargins(0, 0, 0, 0)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground('#1e1e1e')
        graphics_layout.addWidget(self.graphics)

        self.graphics_container = graphics_container
        self.spinner = SpinnerOverlay(graphics_container)
        layout.addWidget(graphics_container)

        # CH1 waveform + envelope
        self.plot_ch1 = self.graphics.addPlot(row=0, col=0, title='CH1 Waveform + Envelope')
        self.plot_ch1.setLabel('left', 'Voltage', units='mV')
        self.plot_ch1.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ch1 = self.plot_ch1.plot(pen=pg.mkPen(color=(255, 255, 0, 180), width=0.5))
        self.curve_ch1_env_upper = self.plot_ch1.plot(pen=pg.mkPen('r', width=1.5))
        self.curve_ch1_env_lower = self.plot_ch1.plot(pen=pg.mkPen('r', width=1.5))
        self.onset_marker_ch1 = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color=(0, 255, 0), width=3))
        self.onset_marker_ch1.setZValue(1000)
        self.onset_marker_ch1.setVisible(False)
        self.plot_ch1.addItem(self.onset_marker_ch1)
        self.onset_label_ch1 = pg.TextItem(text='', color=(0, 255, 0), anchor=(0, 0))
        self.onset_label_ch1.setZValue(1001)
        self.onset_label_ch1.setVisible(False)
        self.plot_ch1.addItem(self.onset_label_ch1, ignoreBounds=True)

        # CH2 waveform + envelope
        self.plot_ch2 = self.graphics.addPlot(row=1, col=0, title='CH2 Waveform + Envelope')
        self.plot_ch2.setLabel('left', 'Voltage', units='mV')
        self.plot_ch2.setLabel('bottom', 'Time', units='ms')
        self.plot_ch2.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ch2 = self.plot_ch2.plot(pen=pg.mkPen(color=(0, 255, 255, 180), width=0.5))
        self.curve_ch2_env_upper = self.plot_ch2.plot(pen=pg.mkPen('m', width=1.5))
        self.curve_ch2_env_lower = self.plot_ch2.plot(pen=pg.mkPen('m', width=1.5))
        self.onset_marker_ch2 = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color=(0, 255, 0), width=3))
        self.onset_marker_ch2.setZValue(1000)
        self.onset_marker_ch2.setVisible(False)
        self.plot_ch2.addItem(self.onset_marker_ch2)
        self.onset_label_ch2 = pg.TextItem(text='', color=(0, 255, 0), anchor=(0, 0))
        self.onset_label_ch2.setZValue(1001)
        self.onset_label_ch2.setVisible(False)
        self.plot_ch2.addItem(self.onset_label_ch2, ignoreBounds=True)

        # DWT scalogram plots
        self.plot_dwt1 = self.graphics.addPlot(row=0, col=1, title='CH1 DWT Scalogram')
        self.plot_dwt1.setLabel('left', 'Level')
        self.plot_dwt1.showGrid(x=True, y=True, alpha=0.3)
        self.img_dwt1 = pg.ImageItem()
        self.plot_dwt1.addItem(self.img_dwt1)

        self.plot_dwt2 = self.graphics.addPlot(row=1, col=1, title='CH2 DWT Scalogram')
        self.plot_dwt2.setLabel('left', 'Level')
        self.plot_dwt2.setLabel('bottom', 'Time', units='ms')
        self.plot_dwt2.showGrid(x=True, y=True, alpha=0.3)
        self.img_dwt2 = pg.ImageItem()
        self.plot_dwt2.addItem(self.img_dwt2)

        # Colormap
        self.dwt_cmap = pg.colormap.get('viridis')

        # Gradient legends
        self.gradient_legend1 = pg.GradientLegend((10, 150), (-20, -20))
        self.gradient_legend1.setParentItem(self.plot_dwt1.getViewBox())
        self.gradient_legend1.setGradient(self.dwt_cmap.getGradient())

        self.gradient_legend2 = pg.GradientLegend((10, 150), (-20, -20))
        self.gradient_legend2.setParentItem(self.plot_dwt2.getViewBox())
        self.gradient_legend2.setGradient(self.dwt_cmap.getGradient())

        # Click handler for scalograms
        self.current_dwt_result = None
        self.img_dwt1.mouseClickEvent = self._on_scalogram_click
        self.img_dwt2.mouseClickEvent = self._on_scalogram_click

        # Link X axes
        self.plot_ch2.setXLink(self.plot_ch1)
        self.plot_dwt2.setXLink(self.plot_dwt1)

        # Keyboard shortcuts
        close_shortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+W'), self)
        close_shortcut.activated.connect(self.close)

    def _connect_scope(self):
        QtCore.QTimer.singleShot(0, self._show_startup_spinner)

        try:
            print(f"Connecting to {self.ip}:{PORT}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.ip, PORT))
            self.sock.settimeout(5)

            idn = query(self.sock, '*IDN?')
            print(f"Connected to: {idn}")
            self.status_label.setText(f'Connected: {idn[:40]}...')
            self.status_label.setStyleSheet('color: green; font-weight: bold;')

            self._configure_scope()
            QtCore.QTimer.singleShot(100, self._do_initial_capture)

        except socket.error as e:
            print(f"Connection failed: {e}")
            self.status_label.setText(f'Connection failed: {e}')
            self.status_label.setStyleSheet('color: red; font-weight: bold;')
            self.spinner.stop()

    def _show_startup_spinner(self):
        self.spinner.setGeometry(self.graphics_container.rect())
        self.spinner.start("Loading...")

    def _configure_scope(self):
        send_command(self.sock, 'SCSV OFF')
        send_command(self.sock, 'MSIZ 14M')
        self._apply_waveform_settings()

        print(f"Setting CH1: {self.ch1_vdiv*1000:.0f}mV/div")
        send_command(self.sock, 'C1:TRA ON')
        send_command(self.sock, f'C1:VDIV {self.ch1_vdiv}')
        send_command(self.sock, 'C1:OFST 0')

        print(f"Setting CH2: {self.ch2_vdiv*1000:.0f}mV/div")
        send_command(self.sock, 'C2:TRA ON')
        send_command(self.sock, f'C2:VDIV {self.ch2_vdiv}')
        send_command(self.sock, 'C2:OFST 0')

        print(f"Setting horizontal: {self.hdiv*1000:.0f}ms/div")
        send_command(self.sock, f'TDIV {self.hdiv}')
        delay_sec = self.trigger_delay_ms / 1000.0
        send_command(self.sock, f'TRDL {delay_sec}')

        print(f"Setting trigger: C1 @ {self.trigger_level}V")
        send_command(self.sock, 'TRSE EDGE,SR,C1,HT,OFF')
        send_command(self.sock, f'C1:TRLV {self.trigger_level}V')

    def _do_initial_capture(self):
        if not self.sock:
            self.spinner.stop()
            return

        print("Initial capture...")
        query(self.sock, 'INR?', timeout=0.1)
        send_command(self.sock, 'TRMD AUTO', wait=False)

        for _ in range(30):
            QtWidgets.QApplication.processEvents()
            try:
                resp = query(self.sock, 'INR?', timeout=0.1)
                if resp and int(resp.split()[-1]) & 1:
                    break
            except:
                pass
            time.sleep(0.05)

        send_command(self.sock, 'STOP', wait=False)
        time.sleep(0.1)

        upsample = self.upsample_factor if self.interpolate_enabled else 1
        wf1, _ = get_waveform(self.sock, 'C1', self.ch1_vdiv, upsample=upsample)
        wf2, _ = get_waveform(self.sock, 'C2', self.ch2_vdiv, upsample=upsample)

        print(f"  CH1: {len(wf1)} samples, CH2: {len(wf2)} samples")

        if len(wf1) > 0 and len(wf2) > 0:
            env1 = compute_envelope(wf1, smooth_window=self.envelope_smooth)
            env2 = compute_envelope(wf2, smooth_window=self.envelope_smooth)
            self._update_plots(wf1, env1, wf2, env2)
            corr = analyze_correlation(env1, env2)
            self._update_correlation_display(corr)

        self.spinner.stop()
        self._start_capture()

    def _update_correlation_display(self, corr):
        score = corr['correlation_score']
        classification = corr['classification']

        self.corr_score_label.setText(f"{score:.3f}")
        self.corr_class_label.setText(classification)

        if classification == "CORRELATED":
            color = 'green'
        elif classification == "WEAKLY_CORRELATED":
            color = 'orange'
        else:
            color = 'red'
        self.corr_class_label.setStyleSheet(f'font-size: 18px; font-weight: bold; color: {color};')
        self.corr_score_label.setStyleSheet(f'font-size: 18px; font-weight: bold; color: {color};')

        self.pearson_env_label.setText(f"{corr['pearson_envelope']:.3f}")
        self.cosine_sim_label.setText(f"{corr['cosine_similarity']:.3f}")
        self.ncc_lag_label.setText(f"{corr['ncc_lag_samples']} samples")
        self.onset_lag_label.setText(f"{corr['onset_lag_samples']} samples")
        self.peak_lag_label.setText(f"{corr['peak_lag_samples']} samples")
        self.amp_ratio_label.setText(f"{corr['amplitude_ratio']:.3f}")

        if self.current_time_array is not None:
            onset1_idx = corr['onset1_idx']
            onset2_idx = corr['onset2_idx']
            if onset1_idx < len(self.current_time_array):
                t1 = self.current_time_array[onset1_idx]
                self.onset_marker_ch1.setPos(t1)
                self.onset_label_ch1.setText(f't={t1:.1f}ms')
                self.onset_label_ch1.setPos(t1, getattr(self, 'env1_max_mv', 50) * 0.9)
            if onset2_idx < len(self.current_time_array):
                t2 = self.current_time_array[onset2_idx]
                self.onset_marker_ch2.setPos(t2)
                self.onset_label_ch2.setText(f't={t2:.1f}ms')
                self.onset_label_ch2.setPos(t2, getattr(self, 'env2_max_mv', 50) * 0.9)

    def _arm_trigger(self, force_mode=False):
        if self.sock:
            if force_mode:
                self.sock.sendall(b'TRMD NORM\n')
                time.sleep(0.05)
            self.sock.sendall(b'INR?\n')
            self.sock.settimeout(0.2)
            try:
                self.sock.recv(256)
            except:
                pass
            print("Waiting for trigger...")
            self.status_label.setText('Trigger armed - waiting...')
            self.status_label.setStyleSheet('color: orange; font-weight: bold;')

    def _on_decimate_changed(self):
        self.decimate_enabled = self.decimate_cb.isChecked()
        self.decimate_points = self.decimate_spin.value()
        self.decimate_spin.setEnabled(self.decimate_enabled)
        if self.sock:
            self._apply_waveform_settings()

    def _on_interp_changed(self):
        self.interpolate_enabled = self.interp_cb.isChecked()
        self.upsample_factor = self.upsample_spin.value()
        self.upsample_spin.setEnabled(self.interpolate_enabled)

    def _on_delay_changed(self, value):
        self.trigger_delay_ms = value
        if self.sock:
            delay_sec = value / 1000.0
            self.sock.sendall(f'TRDL {delay_sec}\n'.encode())

    def _on_trigger_level_changed(self, value):
        self.trigger_level = value
        if self.sock:
            self.sock.sendall(f'C1:TRLV {value}V\n'.encode())

    def _on_onset_markers_changed(self, state):
        self.show_onset_markers = bool(state)
        self.onset_marker_ch1.setVisible(self.show_onset_markers)
        self.onset_marker_ch2.setVisible(self.show_onset_markers)
        self.onset_label_ch1.setVisible(self.show_onset_markers)
        self.onset_label_ch2.setVisible(self.show_onset_markers)

    def _on_scalogram_click(self, event):
        if self.current_dwt_result is not None:
            viewer = WaveletViewer(
                self.current_dwt_result['coeffs1'],
                self.current_dwt_result['freq_bands'],
                self.current_dwt_result['wavelet'],
                self
            )
            viewer.show()

    def _apply_waveform_settings(self):
        if self.decimate_enabled:
            sparsing = 14000000 // self.decimate_points
            send_command(self.sock, f'WFSU SP,{sparsing},NP,{self.decimate_points},FP,0')
            print(f"Decimation: {self.decimate_points} points (SP={sparsing})")
        else:
            send_command(self.sock, 'WFSU SP,1,NP,0,FP,0')
            print("Decimation: OFF")

    def _toggle_trigger(self):
        if self.running:
            self._disarm_trigger()
        else:
            self._start_capture()

    def _start_capture(self):
        self.running = True
        self.arm_btn.setText('Disarm Trigger')
        self._arm_trigger(force_mode=True)
        self.poll_timer.start(50)

    def _disarm_trigger(self):
        self.running = False
        self.poll_timer.stop()
        self.arm_btn.setText('Arm Trigger')
        self.status_label.setText('Disarmed')
        self.status_label.setStyleSheet('color: gray; font-weight: bold;')
        if self.sock:
            send_command(self.sock, 'TRMD AUTO', wait=False)

    def _manual_capture(self):
        if not self.sock:
            return

        was_running = self.running
        if was_running:
            self.poll_timer.stop()

        self.status_label.setText('Capturing...')
        self.status_label.setStyleSheet('color: blue; font-weight: bold;')
        QtWidgets.QApplication.processEvents()

        query(self.sock, 'INR?', timeout=0.1)
        send_command(self.sock, 'TRMD AUTO', wait=False)

        for _ in range(30):
            QtWidgets.QApplication.processEvents()
            try:
                resp = query(self.sock, 'INR?', timeout=0.1)
                if resp and int(resp.split()[-1]) & 1:
                    break
            except:
                pass
            time.sleep(0.05)

        send_command(self.sock, 'STOP', wait=False)
        time.sleep(0.1)

        upsample = self.upsample_factor if self.interpolate_enabled else 1
        wf1, _ = get_waveform(self.sock, 'C1', self.ch1_vdiv, upsample=upsample)
        wf2, _ = get_waveform(self.sock, 'C2', self.ch2_vdiv, upsample=upsample)

        if len(wf1) > 0 and len(wf2) > 0:
            env1 = compute_envelope(wf1, smooth_window=self.envelope_smooth)
            env2 = compute_envelope(wf2, smooth_window=self.envelope_smooth)
            self._update_plots(wf1, env1, wf2, env2)
            corr = analyze_correlation(env1, env2)
            self._update_correlation_display(corr)
            self.capture_count += 1
            self.capture_label.setText(str(self.capture_count))
            self._save_capture(wf1, env1, wf2, env2, corr)

        self.sock.sendall(b'TRMD NORM\n')
        time.sleep(0.05)

        if was_running:
            self._arm_trigger()
            self.poll_timer.start(50)
        else:
            self.status_label.setText('Disarmed')
            self.status_label.setStyleSheet('color: gray; font-weight: bold;')

    def _poll_trigger(self):
        if not self.sock:
            return

        try:
            resp = query(self.sock, 'INR?', timeout=0.1)
            parts = resp.split()
            if parts:
                inr_val = int(parts[-1])
                if inr_val & 1:
                    self._on_trigger()
        except (ValueError, socket.timeout):
            pass

    def _on_trigger(self):
        print("Trigger fired!")
        self.status_label.setText('Capturing...')
        self.status_label.setStyleSheet('color: blue; font-weight: bold;')
        QtWidgets.QApplication.processEvents()

        upsample = self.upsample_factor if self.interpolate_enabled else 1
        wf1, _ = get_waveform(self.sock, 'C1', self.ch1_vdiv, upsample=upsample)
        wf2, _ = get_waveform(self.sock, 'C2', self.ch2_vdiv, upsample=upsample)
        print(f"  CH1: {len(wf1)} samples, CH2: {len(wf2)} samples")

        if len(wf1) == 0 or len(wf2) == 0:
            print("  No data received, re-arming...")
            self._arm_trigger()
            return

        env1 = compute_envelope(wf1, smooth_window=self.envelope_smooth)
        env2 = compute_envelope(wf2, smooth_window=self.envelope_smooth)

        corr = analyze_correlation(env1, env2)
        print(f"Correlation: {corr['classification']} (score: {corr['correlation_score']:.3f})")

        self.capture_count += 1
        self.capture_label.setText(str(self.capture_count))

        self._update_correlation_display(corr)
        self._update_plots(wf1, env1, wf2, env2)
        self._save_capture(wf1, env1, wf2, env2, corr)

        if self.running:
            self._arm_trigger()

    def _update_plots(self, wf1, env1, wf2, env2):
        total_time = self.hdiv * 14 * 1000  # ms
        time_axis = np.linspace(0, total_time, len(wf1))
        self.current_time_array = time_axis

        self.env1_max_mv = np.max(env1) * 1000
        self.env2_max_mv = np.max(env2) * 1000

        # Waveform plots
        self.curve_ch1.setData(time_axis, wf1 * 1000)
        self.curve_ch1_env_upper.setData(time_axis[:len(env1)], env1 * 1000)
        self.curve_ch1_env_lower.setData(time_axis[:len(env1)], -env1 * 1000)
        self.onset_label_ch1.setPos(self.onset_marker_ch1.value(), self.env1_max_mv * 0.9)

        time_axis2 = np.linspace(0, total_time, len(wf2))
        self.curve_ch2.setData(time_axis2, wf2 * 1000)
        self.curve_ch2_env_upper.setData(time_axis2[:len(env2)], env2 * 1000)
        self.curve_ch2_env_lower.setData(time_axis2[:len(env2)], -env2 * 1000)
        self.onset_label_ch2.setPos(self.onset_marker_ch2.value(), self.env2_max_mv * 0.9)

        # DWT analysis
        sample_rate = len(wf1) / (self.hdiv * 14)

        # Downsample for faster computation
        ds_factor = max(1, len(wf1) // self.dwt_target_samples)
        wf1_ds = wf1[::ds_factor]
        wf2_ds = wf2[::ds_factor]
        sample_rate_ds = sample_rate / ds_factor
        print(f"DWT: {len(wf1)} -> {len(wf1_ds)} samples (ds={ds_factor})")

        dwt_result = dwt_correlation(wf1_ds, wf2_ds, sample_rate_ds,
                                     wavelet=self.dwt_wavelet,
                                     max_level=self.dwt_max_level)
        self.current_dwt_result = dwt_result

        # Update scalogram displays
        scalogram1 = dwt_result['scalogram1']
        scalogram2 = dwt_result['scalogram2']

        # Convert to dB
        power1_db = 10 * np.log10(scalogram1 + 1e-10)
        power2_db = 10 * np.log10(scalogram2 + 1e-10)

        vmin = min(np.percentile(power1_db, 5), np.percentile(power2_db, 5))
        vmax = max(np.percentile(power1_db, 95), np.percentile(power2_db, 95))

        # Set image data (transpose for X=time, Y=level)
        self.img_dwt1.setImage(power1_db.T, levels=(vmin, vmax))
        self.img_dwt2.setImage(power2_db.T, levels=(vmin, vmax))

        self.img_dwt1.setLookupTable(self.dwt_cmap.getLookupTable())
        self.img_dwt2.setLookupTable(self.dwt_cmap.getLookupTable())

        # Scale images
        n_levels = dwt_result['n_levels']
        self.img_dwt1.setRect(0, 0, total_time, n_levels)
        self.img_dwt2.setRect(0, 0, total_time, n_levels)

        self.plot_dwt1.setXRange(0, total_time, padding=0)
        self.plot_dwt1.setYRange(0, n_levels, padding=0)
        self.plot_dwt2.setXRange(0, total_time, padding=0)
        self.plot_dwt2.setYRange(0, n_levels, padding=0)

        # Update legend
        labels = {f'{vmax:.0f} dB': 1.0, f'{vmin:.0f} dB': 0.0}
        self.gradient_legend1.setLabels(labels)
        self.gradient_legend2.setLabels(labels)

        # Update DWT metrics
        self.dwt_score_label.setText(f"{dwt_result['dwt_score']:.3f}")
        self.dwt_weighted_label.setText(f"{dwt_result['weighted_corr']:.3f}")
        self.energy_dist_label.setText(f"{dwt_result['energy_dist_corr']:.3f}")
        self.dom_level1_label.setText(f"Level {dwt_result['dominant_level1']}")
        self.dom_level2_label.setText(f"Level {dwt_result['dominant_level2']}")

    def _save_capture(self, wf1, env1, wf2, env2, corr):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        np.save(self.captures_dir / f"ch1_wf_{timestamp}.npy", wf1.astype(np.float32))
        np.save(self.captures_dir / f"ch2_wf_{timestamp}.npy", wf2.astype(np.float32))
        np.save(self.captures_dir / f"ch1_env_{timestamp}.npy", env1.astype(np.float32))
        np.save(self.captures_dir / f"ch2_env_{timestamp}.npy", env2.astype(np.float32))

        with open(self.captures_dir / f"corr_{timestamp}.json", 'w') as f:
            json.dump(corr, f, indent=2)

        try:
            exporter = pyqtgraph.exporters.ImageExporter(self.graphics.scene())
            exporter.parameters()['width'] = 1400
            exporter.export(str(self.captures_dir / f"plot_{timestamp}.png"))
        except Exception as e:
            print(f"  Warning: Could not save plot image: {e}")

        print(f"Saved capture: {timestamp}")

    def _signal_handler(self, signum, frame):
        print("\nReceived SIGINT, closing gracefully...")
        self.close()

    def closeEvent(self, event):
        self.poll_timer.stop()
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Piezo capture with DWT analysis')
    parser.add_argument('--ip', default=SCOPE_IP, help=f'Scope IP address (default: {SCOPE_IP})')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = PiezoCapture(ip=args.ip)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
