#!/usr/bin/env python
"""
Continuous piezo capture with pyqtgraph - processes each trigger event.
"""
import json
import socket
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
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
        # Wait for operation complete by querying *OPC?
        # The scope returns "1\n" when the previous command has finished
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
                # Look for "1" response (may have extra whitespace/newlines)
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

    # Decode with error handling for any stray binary data
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
        print(f"  DEBUG {channel}: got {len(data_raw)} bytes, starts with: {data_raw[:50] if data_raw else 'empty'}")

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

    # Remove trailing garbage bytes
    if len(waveform_data) > 2:
        waveform_data = waveform_data[:-2]

    # Convert bytes to voltage
    values = np.frombuffer(waveform_data, dtype=np.int8)
    voltage = (values.astype(float) * vdiv / 25.0) - offset

    # Upsample using cubic spline for smooth curves
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

    # Smooth the envelope with a moving average
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


def analyze_correlation(env1, env2, wf1, wf2):
    """Analyze correlation between two channels."""
    min_len = min(len(env1), len(env2))
    env1, env2 = env1[:min_len], env2[:min_len]
    wf1, wf2 = wf1[:min_len], wf2[:min_len]

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

    peak1_amp = np.max(env1)
    peak2_amp = np.max(env2)
    amplitude_ratio = min(peak1_amp, peak2_amp) / (max(peak1_amp, peak2_amp) + 1e-10)

    cosine_sim = np.dot(env1_norm, env2_norm) / (np.linalg.norm(env1_norm) * np.linalg.norm(env2_norm) + 1e-10)

    correlation_score = (
        0.35 * pearson_env +
        0.30 * cosine_sim +
        0.20 * amplitude_ratio +
        0.15 * (1 - min(abs(peak_lag_samples), 100) / 100)
    )

    if correlation_score > 0.7:
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
        'amplitude_ratio': float(amplitude_ratio),
        'cosine_similarity': float(cosine_sim),
        'correlation_score': float(correlation_score),
        'classification': classification,
    }


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
        """Show spinner with given text."""
        self._text = text
        self._angle = 0
        self.show()
        self.raise_()
        self._timer.start(50)

    def stop(self):
        """Hide spinner."""
        self._timer.stop()
        self.hide()

    def _rotate(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Semi-transparent background
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 150))

        center = self.rect().center()

        # Draw spinning arc
        painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100), 4))
        arc_rect = QtCore.QRectF(center.x() - 30, center.y() - 30, 60, 60)
        painter.drawEllipse(arc_rect)

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 150, 255), 4))
        painter.drawArc(arc_rect, self._angle * 16, 90 * 16)

        # Draw text below spinner
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.setFont(QtGui.QFont("Arial", 12))
        text_rect = QtCore.QRectF(0, center.y() + 50, self.width(), 30)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self._text)


class PiezoCapture(QtWidgets.QMainWindow):
    def __init__(self, ip=SCOPE_IP):
        super().__init__()
        self.setWindowTitle('Piezo Capture - Dual Channel Correlation')
        self.resize(1400, 900)

        # Settings
        self.ip = ip
        self.ch1_vdiv = 0.07
        self.ch2_vdiv = 0.07
        self.hdiv = 0.02
        self.trigger_level = 0.005
        self.capture_count = 0
        self.running = False

        # Decimation/interpolation settings
        self.decimate_enabled = True
        self.decimate_points = 14000
        self.interpolate_enabled = True
        self.upsample_factor = 4
        self.envelope_smooth = 201  # Larger window for smoother envelope

        # Create captures directory
        self.captures_dir = Path('captures')
        self.captures_dir.mkdir(exist_ok=True)

        # Setup UI
        self._setup_ui()

        # Connect to scope
        self.sock = None
        self._connect_scope()

        # Polling timer for trigger detection
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

        self.capture_btn = QtWidgets.QPushButton('Capture')
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

        # Classification and score (prominent)
        self.corr_class_label = QtWidgets.QLabel('--')
        self.corr_class_label.setStyleSheet('font-size: 18px; font-weight: bold;')
        corr_grid.addWidget(QtWidgets.QLabel('Classification:'), 0, 0)
        corr_grid.addWidget(self.corr_class_label, 0, 1)

        self.corr_score_label = QtWidgets.QLabel('--')
        self.corr_score_label.setStyleSheet('font-size: 18px; font-weight: bold;')
        corr_grid.addWidget(QtWidgets.QLabel('Score:'), 0, 2)
        corr_grid.addWidget(self.corr_score_label, 0, 3)

        # All correlation parameters
        self.pearson_env_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Pearson (env):'), 1, 0)
        corr_grid.addWidget(self.pearson_env_label, 1, 1)

        self.cosine_sim_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Cosine sim:'), 1, 2)
        corr_grid.addWidget(self.cosine_sim_label, 1, 3)

        self.ncc_peak_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('NCC peak:'), 1, 4)
        corr_grid.addWidget(self.ncc_peak_label, 1, 5)

        self.ncc_lag_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('NCC lag:'), 2, 0)
        corr_grid.addWidget(self.ncc_lag_label, 2, 1)

        self.peak_lag_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Peak lag:'), 2, 2)
        corr_grid.addWidget(self.peak_lag_label, 2, 3)

        self.amp_ratio_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('Amp ratio:'), 2, 4)
        corr_grid.addWidget(self.amp_ratio_label, 2, 5)

        # Spectral centroids
        self.centroid1_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('CH1 centroid:'), 3, 0)
        corr_grid.addWidget(self.centroid1_label, 3, 1)

        self.centroid2_label = QtWidgets.QLabel('--')
        corr_grid.addWidget(QtWidgets.QLabel('CH2 centroid:'), 3, 2)
        corr_grid.addWidget(self.centroid2_label, 3, 3)

        layout.addWidget(corr_group)

        # Plot area with spinner overlay
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

        # CH2 waveform + envelope
        self.plot_ch2 = self.graphics.addPlot(row=1, col=0, title='CH2 Waveform + Envelope')
        self.plot_ch2.setLabel('left', 'Voltage', units='mV')
        self.plot_ch2.setLabel('bottom', 'Time', units='ms')
        self.plot_ch2.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ch2 = self.plot_ch2.plot(pen=pg.mkPen(color=(0, 255, 255, 180), width=0.5))
        self.curve_ch2_env_upper = self.plot_ch2.plot(pen=pg.mkPen('m', width=1.5))
        self.curve_ch2_env_lower = self.plot_ch2.plot(pen=pg.mkPen('m', width=1.5))

        # FFT plots
        self.plot_fft1 = self.graphics.addPlot(row=0, col=1, title='CH1 FFT')
        self.plot_fft1.setLabel('left', 'Magnitude', units='dB')
        self.plot_fft1.showGrid(x=True, y=True, alpha=0.3)
        self.curve_fft1 = self.plot_fft1.plot(pen=pg.mkPen('y', width=1))

        self.plot_fft2 = self.graphics.addPlot(row=1, col=1, title='CH2 FFT')
        self.plot_fft2.setLabel('left', 'Magnitude', units='dB')
        self.plot_fft2.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_fft2.showGrid(x=True, y=True, alpha=0.3)
        self.curve_fft2 = self.plot_fft2.plot(pen=pg.mkPen('c', width=1))

        # Link X axes
        self.plot_ch2.setXLink(self.plot_ch1)
        self.plot_fft2.setXLink(self.plot_fft1)

        # Keyboard shortcuts
        close_shortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+W'), self)
        close_shortcut.activated.connect(self.close)

    def _connect_scope(self):
        # Show spinner immediately
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

            # Configure scope
            self._configure_scope()

            # Do initial capture after window is shown (will start capture when done)
            QtCore.QTimer.singleShot(100, self._do_initial_capture)

        except socket.error as e:
            print(f"Connection failed: {e}")
            self.status_label.setText(f'Connection failed: {e}')
            self.status_label.setStyleSheet('color: red; font-weight: bold;')
            self.spinner.stop()

    def _show_startup_spinner(self):
        """Show spinner immediately on startup."""
        self.spinner.setGeometry(self.graphics_container.rect())
        self.spinner.start("Loading...")

    def _configure_scope(self):
        """Configure scope settings."""
        send_command(self.sock, 'SCSV OFF')

        # Set memory depth and decimation
        send_command(self.sock, 'MSIZ 14M')
        self._apply_waveform_settings()

        # Configure CH1
        print(f"Setting CH1: {self.ch1_vdiv*1000:.0f}mV/div, centered at 0V")
        send_command(self.sock, 'C1:TRA ON')
        send_command(self.sock, f'C1:VDIV {self.ch1_vdiv}')
        send_command(self.sock, 'C1:OFST 0')

        # Configure CH2
        print(f"Setting CH2: {self.ch2_vdiv*1000:.0f}mV/div, centered at 0V")
        send_command(self.sock, 'C2:TRA ON')
        send_command(self.sock, f'C2:VDIV {self.ch2_vdiv}')
        send_command(self.sock, 'C2:OFST 0')

        # Horizontal
        print(f"Setting horizontal: {self.hdiv*1000:.0f}ms/div")
        send_command(self.sock, f'TDIV {self.hdiv}')
        print("Setting horizontal delay: -120ms (left shift)")
        send_command(self.sock, 'TRDL -0.12')

        # Trigger
        print(f"Setting trigger: C1 @ {self.trigger_level}V")
        send_command(self.sock, 'TRSE EDGE,SR,C1,HT,OFF')
        send_command(self.sock, f'C1:TRLV {self.trigger_level}V')

        # Verify settings
        print("\nVerifying settings:")
        print(f"  CH1 V/div: {query(self.sock, 'C1:VDIV?')}")
        print(f"  CH2 V/div: {query(self.sock, 'C2:VDIV?')}")
        print(f"  Time/div: {query(self.sock, 'TDIV?')}")
        print()

    def _do_initial_capture(self):
        """Capture current waveform to fill plots on startup."""
        if not self.sock:
            self.spinner.stop()
            return

        print("Initial capture to fill plots...")

        # Clear INR, put in AUTO mode, poll for acquisition complete
        query(self.sock, 'INR?', timeout=0.1)
        send_command(self.sock, 'TRMD AUTO', wait=False)

        # Poll INR for acquisition complete (bit 0)
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

        # Now read waveforms
        upsample = self.upsample_factor if self.interpolate_enabled else 1
        wf1, _ = get_waveform(self.sock, 'C1', self.ch1_vdiv, upsample=upsample)
        wf2, _ = get_waveform(self.sock, 'C2', self.ch2_vdiv, upsample=upsample)

        print(f"  CH1: {len(wf1)} samples, CH2: {len(wf2)} samples")

        if len(wf1) > 0 and len(wf2) > 0:
            env1 = compute_envelope(wf1, smooth_window=self.envelope_smooth)
            env2 = compute_envelope(wf2, smooth_window=self.envelope_smooth)
            self._update_plots(wf1, env1, wf2, env2)
            corr = analyze_correlation(env1, env2, wf1, wf2)
            self._update_correlation_display(corr)
            print("  Initial capture successful")
        else:
            print("  No data - plots will fill on first trigger")

        # Hide spinner and arm trigger by default
        self.spinner.stop()
        self._start_capture()

    def _update_correlation_display(self, corr):
        """Update the correlation panel labels."""
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
        self.ncc_peak_label.setText(f"{corr['ncc_peak']:.1f}")
        self.ncc_lag_label.setText(f"{corr['ncc_lag_samples']} samples")
        self.peak_lag_label.setText(f"{corr['peak_lag_samples']} samples")
        self.amp_ratio_label.setText(f"{corr['amplitude_ratio']:.3f}")

    def _arm_trigger(self, force_mode=False):
        """Arm the trigger. In NORMAL mode, scope auto-rearms so we just clear INR."""
        if self.sock:
            if force_mode:
                self.sock.sendall(b'TRMD NORM\n')
                time.sleep(0.05)
            # Just clear INR to detect next trigger
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
        """Handle decimation settings change."""
        self.decimate_enabled = self.decimate_cb.isChecked()
        self.decimate_points = self.decimate_spin.value()
        self.decimate_spin.setEnabled(self.decimate_enabled)
        if self.sock:
            self._apply_waveform_settings()

    def _on_interp_changed(self):
        """Handle interpolation settings change."""
        self.interpolate_enabled = self.interp_cb.isChecked()
        self.upsample_factor = self.upsample_spin.value()
        self.upsample_spin.setEnabled(self.interpolate_enabled)

    def _apply_waveform_settings(self):
        """Apply current decimation settings to scope."""
        if self.decimate_enabled:
            sparsing = 14000000 // self.decimate_points
            send_command(self.sock, f'WFSU SP,{sparsing},NP,{self.decimate_points},FP,0')
            print(f"Decimation: {self.decimate_points} points (SP={sparsing})")
        else:
            send_command(self.sock, 'WFSU SP,1,NP,0,FP,0')
            print("Decimation: OFF (full resolution)")

    def _toggle_trigger(self):
        if self.running:
            self._disarm_trigger()
        else:
            self._start_capture()

    def _start_capture(self):
        self.running = True
        self.arm_btn.setText('Disarm Trigger')
        self._arm_trigger(force_mode=True)
        self.poll_timer.start(50)  # Poll every 50ms

    def _disarm_trigger(self):
        self.running = False
        self.poll_timer.stop()
        self.arm_btn.setText('Arm Trigger')
        self.status_label.setText('Disarmed')
        self.status_label.setStyleSheet('color: gray; font-weight: bold;')
        # Put scope back in AUTO mode
        if self.sock:
            send_command(self.sock, 'TRMD AUTO', wait=False)

    def _manual_capture(self):
        """Take a manual capture without waiting for trigger."""
        if not self.sock:
            return

        # Stop polling if running
        was_running = self.running
        if was_running:
            self.poll_timer.stop()

        self.status_label.setText('Capturing...')
        self.status_label.setStyleSheet('color: blue; font-weight: bold;')
        QtWidgets.QApplication.processEvents()

        # Put in AUTO mode and wait for acquisition
        query(self.sock, 'INR?', timeout=0.1)
        send_command(self.sock, 'TRMD AUTO', wait=False)

        # Poll for acquisition complete
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

        # Capture waveforms
        upsample = self.upsample_factor if self.interpolate_enabled else 1
        wf1, _ = get_waveform(self.sock, 'C1', self.ch1_vdiv, upsample=upsample)
        wf2, _ = get_waveform(self.sock, 'C2', self.ch2_vdiv, upsample=upsample)

        print(f"Manual capture: CH1 {len(wf1)} samples, CH2 {len(wf2)} samples")

        if len(wf1) > 0 and len(wf2) > 0:
            env1 = compute_envelope(wf1, smooth_window=self.envelope_smooth)
            env2 = compute_envelope(wf2, smooth_window=self.envelope_smooth)
            self._update_plots(wf1, env1, wf2, env2)
            corr = analyze_correlation(env1, env2, wf1, wf2)
            self._update_correlation_display(corr)
            self.capture_count += 1
            self.capture_label.setText(str(self.capture_count))
            self._save_capture(wf1, env1, wf2, env2, corr)

        # Resume trigger if was running
        if was_running:
            self._arm_trigger()
            self.poll_timer.start(50)
        else:
            self.status_label.setText('Disarmed')
            self.status_label.setStyleSheet('color: gray; font-weight: bold;')

    def _poll_trigger(self):
        """Check if trigger has fired."""
        if not self.sock:
            return

        try:
            resp = query(self.sock, 'INR?', timeout=0.1)
            parts = resp.split()
            if parts:
                inr_val = int(parts[-1])
                if inr_val & 1:  # Trigger fired
                    self._on_trigger()
        except (ValueError, socket.timeout):
            pass

    def _on_trigger(self):
        """Handle trigger event - capture and process."""
        print("Trigger fired!")
        self.status_label.setText('Capturing...')
        self.status_label.setStyleSheet('color: blue; font-weight: bold;')
        QtWidgets.QApplication.processEvents()

        # Capture both channels
        print("Capturing waveforms...")
        upsample = self.upsample_factor if self.interpolate_enabled else 1
        wf1, _ = get_waveform(self.sock, 'C1', self.ch1_vdiv, upsample=upsample)
        wf2, _ = get_waveform(self.sock, 'C2', self.ch2_vdiv, upsample=upsample)
        print(f"  CH1: {len(wf1)} samples, CH2: {len(wf2)} samples")

        if len(wf1) == 0 or len(wf2) == 0:
            print("  No data received, re-arming...")
            self._arm_trigger()
            return

        # Compute envelopes
        env1 = compute_envelope(wf1, smooth_window=self.envelope_smooth)
        env2 = compute_envelope(wf2, smooth_window=self.envelope_smooth)

        # Analyze correlation
        corr = analyze_correlation(env1, env2, wf1, wf2)
        print(f"Correlation: {corr['classification']} (score: {corr['correlation_score']:.3f})")

        # Update capture count
        self.capture_count += 1
        self.capture_label.setText(str(self.capture_count))

        # Update correlation display
        self._update_correlation_display(corr)

        # Update plots
        self._update_plots(wf1, env1, wf2, env2)

        # Save data
        self._save_capture(wf1, env1, wf2, env2, corr)

        # Re-arm trigger
        if self.running:
            self._arm_trigger()

    def _update_plots(self, wf1, env1, wf2, env2):
        """Update all plots with new data."""
        # Time axis
        total_time = self.hdiv * 14 * 1000  # ms
        time_axis = np.linspace(0, total_time, len(wf1))

        # CH1 waveform + envelope
        self.curve_ch1.setData(time_axis, wf1 * 1000)
        self.curve_ch1_env_upper.setData(time_axis[:len(env1)], env1 * 1000)
        self.curve_ch1_env_lower.setData(time_axis[:len(env1)], -env1 * 1000)

        # CH2 waveform + envelope
        time_axis2 = np.linspace(0, total_time, len(wf2))
        self.curve_ch2.setData(time_axis2, wf2 * 1000)
        self.curve_ch2_env_upper.setData(time_axis2[:len(env2)], env2 * 1000)
        self.curve_ch2_env_lower.setData(time_axis2[:len(env2)], -env2 * 1000)

        # FFT and spectral centroids
        sample_rate = len(wf1) / (self.hdiv * 14)
        centroids = []

        for wf, curve in [(wf1, self.curve_fft1), (wf2, self.curve_fft2)]:
            n = len(wf)
            fft_vals = np.fft.fft(wf)
            fft_mag = np.abs(fft_vals[:n//2]) * 2 / n
            freqs = np.fft.fftfreq(n, 1/sample_rate)[:n//2]
            fft_db = 20 * np.log10(fft_mag + 1e-10)
            curve.setData(freqs, fft_db)

            # Compute spectral centroid: sum(freq * mag) / sum(mag)
            mag_sum = np.sum(fft_mag)
            if mag_sum > 0:
                centroid = np.sum(freqs * fft_mag) / mag_sum
            else:
                centroid = 0
            centroids.append(centroid)

        # Update centroid labels
        self.centroid1_label.setText(f"{centroids[0]:.1f} Hz")
        self.centroid2_label.setText(f"{centroids[1]:.1f} Hz")

        # Set FFT x limit
        self.plot_fft1.setXRange(0, min(sample_rate/2, 5000))
        self.plot_fft2.setXRange(0, min(sample_rate/2, 5000))

    def _save_capture(self, wf1, env1, wf2, env2, corr):
        """Save capture data to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Save waveforms
        np.save(self.captures_dir / f"ch1_wf_{timestamp}.npy", wf1.astype(np.float32))
        np.save(self.captures_dir / f"ch2_wf_{timestamp}.npy", wf2.astype(np.float32))

        # Save envelopes
        np.save(self.captures_dir / f"ch1_env_{timestamp}.npy", env1.astype(np.float32))
        np.save(self.captures_dir / f"ch2_env_{timestamp}.npy", env2.astype(np.float32))

        # Save correlation
        with open(self.captures_dir / f"corr_{timestamp}.json", 'w') as f:
            json.dump(corr, f, indent=2)

        # Save plot image
        try:
            exporter = pyqtgraph.exporters.ImageExporter(self.graphics.scene())
            exporter.parameters()['width'] = 1400
            exporter.export(str(self.captures_dir / f"plot_{timestamp}.png"))
        except Exception as e:
            print(f"  Warning: Could not save plot image: {e}")

        print(f"Saved capture: {timestamp}")

    def _signal_handler(self, signum, frame):
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
    parser = argparse.ArgumentParser(description='Continuous piezo capture with correlation analysis')
    parser.add_argument('--ip', default=SCOPE_IP, help=f'Scope IP address (default: {SCOPE_IP})')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = PiezoCapture(ip=args.ip)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
