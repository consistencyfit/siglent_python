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
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

SCOPE_IP = '192.168.1.142'
PORT = 5025


def send_command(sock, cmd):
    """Send a command to the scope."""
    sock.sendall((cmd + '\n').encode())
    time.sleep(0.05)


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

    return response.decode().strip()


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


def get_waveform(sock, channel='C1', vdiv=None):
    """Get waveform data from the specified channel."""
    if vdiv is None:
        vdiv_resp = query(sock, f'{channel}:VDIV?')
        try:
            vdiv = float(vdiv_resp.split()[-1].replace('V', ''))
        except:
            vdiv = 1.0

    offset_resp = query(sock, f'{channel}:OFST?')
    try:
        offset = float(offset_resp.split()[-1].replace('V', ''))
    except:
        offset = 0.0

    data_raw = query_binary(sock, f'{channel}:WF? DAT2', timeout=2)

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

    values = np.frombuffer(waveform_data, dtype=np.int8)
    voltage = (values.astype(float) * vdiv / 25.0) - offset

    return voltage, vdiv


def compute_envelope(signal):
    """Compute the envelope of a signal using Hilbert transform."""
    # Remove DC offset before computing envelope
    signal_centered = signal - np.mean(signal)
    analytic_signal = hilbert(signal_centered)
    envelope = np.abs(analytic_signal)
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

        self.start_btn = QtWidgets.QPushButton('Start Capture')
        self.start_btn.clicked.connect(self._toggle_capture)
        controls.addWidget(self.start_btn)

        self.arm_btn = QtWidgets.QPushButton('Re-arm Trigger')
        self.arm_btn.clicked.connect(self._arm_trigger)
        controls.addWidget(self.arm_btn)

        controls.addSpacing(20)

        controls.addWidget(QtWidgets.QLabel('Captures:'))
        self.capture_label = QtWidgets.QLabel('0')
        self.capture_label.setMinimumWidth(50)
        controls.addWidget(self.capture_label)

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

        layout.addWidget(corr_group)

        # Plot area
        pg.setConfigOptions(antialias=False, useOpenGL=True)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground('#1e1e1e')
        layout.addWidget(self.graphics)

        # CH1 waveform + envelope
        self.plot_ch1 = self.graphics.addPlot(row=0, col=0, title='CH1 Waveform + Envelope')
        self.plot_ch1.setLabel('left', 'Voltage', units='mV')
        self.plot_ch1.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ch1 = self.plot_ch1.plot(pen=pg.mkPen('y', width=1))
        self.curve_ch1_env_upper = self.plot_ch1.plot(pen=pg.mkPen('r', width=2))
        self.curve_ch1_env_lower = self.plot_ch1.plot(pen=pg.mkPen('r', width=2))

        # CH2 waveform + envelope
        self.plot_ch2 = self.graphics.addPlot(row=1, col=0, title='CH2 Waveform + Envelope')
        self.plot_ch2.setLabel('left', 'Voltage', units='mV')
        self.plot_ch2.setLabel('bottom', 'Time', units='ms')
        self.plot_ch2.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ch2 = self.plot_ch2.plot(pen=pg.mkPen('c', width=1))
        self.curve_ch2_env_upper = self.plot_ch2.plot(pen=pg.mkPen('m', width=2))
        self.curve_ch2_env_lower = self.plot_ch2.plot(pen=pg.mkPen('m', width=2))

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

    def _connect_scope(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.ip, PORT))
            self.sock.settimeout(5)

            idn = query(self.sock, '*IDN?')
            self.status_label.setText(f'Connected: {idn[:40]}...')
            self.status_label.setStyleSheet('color: green; font-weight: bold;')

            # Configure scope
            self._configure_scope()

        except socket.error as e:
            self.status_label.setText(f'Connection failed: {e}')
            self.status_label.setStyleSheet('color: red; font-weight: bold;')

    def _configure_scope(self):
        """Configure scope settings."""
        send_command(self.sock, 'SCSV OFF')

        # Configure CH1
        send_command(self.sock, 'C1:TRA ON')
        time.sleep(0.1)
        send_command(self.sock, f'C1:VDIV {self.ch1_vdiv}')
        time.sleep(0.1)
        send_command(self.sock, 'C1:OFST 0')
        time.sleep(0.1)
        query(self.sock, 'C1:VDIV?')  # Force processing

        # Configure CH2
        send_command(self.sock, 'C2:TRA ON')
        time.sleep(0.1)
        send_command(self.sock, f'C2:VDIV {self.ch2_vdiv}')
        time.sleep(0.1)
        send_command(self.sock, 'C2:OFST 0')
        time.sleep(0.1)
        query(self.sock, 'C2:VDIV?')  # Force processing

        # Horizontal
        send_command(self.sock, f'TDIV {self.hdiv}')
        time.sleep(0.1)
        send_command(self.sock, 'TRDL -0.12')
        time.sleep(0.1)

        # Trigger
        send_command(self.sock, 'TRSE EDGE,SR,C1,HT,OFF')
        send_command(self.sock, f'C1:TRLV {self.trigger_level}V')
        time.sleep(0.1)

        # Verify settings
        ch1_vdiv = query(self.sock, 'C1:VDIV?')
        ch2_vdiv = query(self.sock, 'C2:VDIV?')
        print(f"Configured: CH1={ch1_vdiv}, CH2={ch2_vdiv}")

    def _arm_trigger(self):
        """Arm the trigger in NORMAL mode."""
        if self.sock:
            send_command(self.sock, 'TRMD NORM')
            query(self.sock, 'INR?')  # Clear INR
            self.status_label.setText('Trigger armed - waiting...')
            self.status_label.setStyleSheet('color: orange; font-weight: bold;')

    def _toggle_capture(self):
        if self.running:
            self._stop_capture()
        else:
            self._start_capture()

    def _start_capture(self):
        self.running = True
        self.start_btn.setText('Stop Capture')
        self._arm_trigger()
        self.poll_timer.start(50)  # Poll every 50ms

    def _stop_capture(self):
        self.running = False
        self.poll_timer.stop()
        self.start_btn.setText('Start Capture')
        self.status_label.setText('Stopped')
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
        self.status_label.setText('Capturing...')
        self.status_label.setStyleSheet('color: blue; font-weight: bold;')
        QtWidgets.QApplication.processEvents()

        # Capture both channels
        wf1, _ = get_waveform(self.sock, 'C1', self.ch1_vdiv)
        wf2, _ = get_waveform(self.sock, 'C2', self.ch2_vdiv)

        if len(wf1) == 0 or len(wf2) == 0:
            self._arm_trigger()
            return

        # Compute envelopes
        env1 = compute_envelope(wf1)
        env2 = compute_envelope(wf2)

        # Analyze correlation
        corr = analyze_correlation(env1, env2, wf1, wf2)

        # Update capture count
        self.capture_count += 1
        self.capture_label.setText(str(self.capture_count))

        # Update correlation display
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

        # Update all correlation parameters
        self.pearson_env_label.setText(f"{corr['pearson_envelope']:.3f}")
        self.cosine_sim_label.setText(f"{corr['cosine_similarity']:.3f}")
        self.ncc_peak_label.setText(f"{corr['ncc_peak']:.1f}")
        self.ncc_lag_label.setText(f"{corr['ncc_lag_samples']} samples")
        self.peak_lag_label.setText(f"{corr['peak_lag_samples']} samples")
        self.amp_ratio_label.setText(f"{corr['amplitude_ratio']:.3f}")

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

        # FFT
        sample_rate = len(wf1) / (self.hdiv * 14)

        for wf, curve in [(wf1, self.curve_fft1), (wf2, self.curve_fft2)]:
            n = len(wf)
            fft_vals = np.fft.fft(wf)
            fft_mag = np.abs(fft_vals[:n//2]) * 2 / n
            freqs = np.fft.fftfreq(n, 1/sample_rate)[:n//2]
            fft_db = 20 * np.log10(fft_mag + 1e-10)
            curve.setData(freqs, fft_db)

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

    def _signal_handler(self, signum, frame):
        self.close()

    def closeEvent(self, event):
        self.poll_timer.stop()
        if self.sock:
            try:
                send_command(self.sock, 'TRMD AUTO')
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
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
