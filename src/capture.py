import atexit
import logging
import socket
import signal
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

scope_ip = '192.168.1.142'
port = 5025  # SCPI port for Siglent scopes

# Cache for scaling parameters (to avoid querying every frame)
scale_cache = {'vdiv': None, 'offset': None, 'tdiv': None}

def send_command(sock, cmd):
    """Send a command to the scope."""
    sock.sendall((cmd + '\n').encode())
    time.sleep(0.05)

def query(sock, cmd, timeout=2, delay=0.05):
    """Send a query and return the response."""
    sock.sendall((cmd + '\n').encode())
    sock.settimeout(timeout)
    time.sleep(delay)

    response = b''
    expected_length = None
    start_time = time.time()

    try:
        while True:
            # Check if we've exceeded timeout
            if time.time() - start_time > timeout:
                break

            chunk = sock.recv(65536)
            if not chunk:
                break
            response += chunk

            # Look for IEEE 488.2 header near the START of response (within first 50 bytes)
            if expected_length is None:
                header_portion = response[:50]
                if b'#' in header_portion:
                    hash_pos = response.find(b'#')
                    if len(response) > hash_pos + 2:
                        try:
                            num_digits = int(chr(response[hash_pos + 1]))
                            if num_digits > 0 and len(response) > hash_pos + 2 + num_digits:
                                length_str = response[hash_pos + 2:hash_pos + 2 + num_digits]
                                data_length = int(length_str)
                                expected_length = hash_pos + 2 + num_digits + data_length + 2
                                log.debug(f"Expecting {expected_length} total bytes ({data_length} data bytes)")
                        except (ValueError, IndexError):
                            pass

            # If we know expected length, check if we have enough
            if expected_length and len(response) >= expected_length:
                break

    except socket.timeout:
        # Keep trying if we know we need more data
        if expected_length and len(response) < expected_length:
            remaining = expected_length - len(response)
            log.debug(f"Timeout but need {remaining} more bytes, retrying...")
            try:
                sock.settimeout(0.5)  # Shorter timeout for retry
                while len(response) < expected_length:
                    chunk = sock.recv(min(65536, expected_length - len(response)))
                    if not chunk:
                        break
                    response += chunk
            except socket.timeout:
                pass

    return response

def parse_value(response):
    """Parse a numeric value from scope response."""
    try:
        text = response.decode().strip()
        # Handle responses like "C1:VDIV 1.00E+00V" or just "1.00E+00"
        parts = text.replace('V', '').replace('S', '').split()
        return float(parts[-1])
    except:
        return None

def get_scale_params(sock, channel='C1'):
    """Get voltage scale parameters from scope."""
    global scale_cache

    vdiv_resp = query(sock, f'{channel}:VDIV?')
    vdiv = parse_value(vdiv_resp)
    if vdiv:
        scale_cache['vdiv'] = vdiv

    offset_resp = query(sock, f'{channel}:OFST?')
    offset = parse_value(offset_resp)
    if offset is not None:
        scale_cache['offset'] = offset
    else:
        scale_cache['offset'] = 0.0

    return scale_cache['vdiv'], scale_cache['offset']

def setup_waveform_transfer(sock, channel='C1', num_points=1400, memory_depth=7000000):
    """Configure waveform transfer parameters for faster streaming."""
    # Calculate sparsing to get approximately num_points from memory_depth
    sparsing = max(1, memory_depth // num_points)
    # WFSU: SP=sparsing, NP=num points, FP=first point, SN=sequence number
    send_command(sock, f'WFSU SP,{sparsing},NP,{num_points},FP,0,SN,0')
    log.info(f"Configured waveform transfer: {num_points} points, sparsing {sparsing}")


def get_waveform(sock, channel='C1'):
    """Retrieve waveform data from the specified channel."""
    t0 = time.time()

    # Clear the INR register first
    sock.sendall(b'INR?\n')
    time.sleep(0.01)
    try:
        sock.recv(1024)  # Discard response
    except:
        pass

    # Wait for new acquisition (bit 0 of INR = new signal acquired)
    for _ in range(50):  # Max 500ms wait
        sock.sendall(b'INR?\n')
        time.sleep(0.01)
        try:
            resp = sock.recv(1024)
            inr_val = int(resp.decode().strip().split()[-1])
            if inr_val & 1:  # Bit 0 set = new data
                break
        except:
            pass

    data_raw = query(sock, f'{channel}:WF? DAT2', timeout=1.0, delay=0.02)
    log.debug(f"Query took {time.time()-t0:.3f}s, got {len(data_raw)} bytes")

    # Parse the binary block data (IEEE 488.2 format: #NXXXXXXXX<data>)
    if b'#' not in data_raw:
        log.warning(f"No '#' in response, first 50 bytes: {data_raw[:50]}")
        return np.array([])

    hash_pos = data_raw.find(b'#')
    if hash_pos == -1 or len(data_raw) <= hash_pos + 2:
        return np.array([])

    # Get number of digits describing the length
    try:
        num_digits = int(chr(data_raw[hash_pos + 1]))
    except ValueError:
        log.warning(f"Invalid num_digits at pos {hash_pos+1}")
        return np.array([])

    if len(data_raw) <= hash_pos + 2 + num_digits:
        return np.array([])

    # Get data length and start position
    length_str = data_raw[hash_pos + 2:hash_pos + 2 + num_digits]
    try:
        data_length = int(length_str)
    except ValueError:
        log.warning(f"Invalid data_length: {length_str}")
        return np.array([])

    data_start = hash_pos + 2 + num_digits
    waveform_data = data_raw[data_start:data_start + data_length]

    if len(waveform_data) < data_length:
        log.warning(f"Incomplete data: got {len(waveform_data)}, expected {data_length}")
        return np.array([])

    if len(waveform_data) < 2:
        return np.array([])

    log.debug(f"Parsed waveform: {len(waveform_data)} bytes")

    # Remove trailing garbage bytes (usually 2)
    if len(waveform_data) > 2:
        waveform_data = waveform_data[:-2]

    # Get scale parameters
    vdiv = scale_cache.get('vdiv') or 1.0
    offset = scale_cache.get('offset') or 0.0

    # Convert bytes to voltage: data * vDiv / 25 - vOffset
    values = np.frombuffer(waveform_data, dtype=np.int8)
    voltage = (values.astype(float) * vdiv / 25.0) - offset

    # Log first 10 raw values to check if data is actually changing
    log.debug(f"First 10 raw: {list(values[:10])} | Range: {voltage.min():.4f} to {voltage.max():.4f}")
    return voltage


class CaptureWorker(QtCore.QObject):
    """Worker that runs network operations in a background thread."""
    waveform_ready = QtCore.pyqtSignal(np.ndarray)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, sock, channel):
        super().__init__()
        self.sock = sock
        self.channel = channel
        self._stop_event = threading.Event()
        self._busy = False  # Guard against overlapping queries

    def stop(self):
        """Signal the worker to stop. This is thread-safe and returns immediately."""
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

    def reset(self):
        """Reset the stop event for a new capture session."""
        self._stop_event.clear()

    @QtCore.pyqtSlot()
    def run_capture(self):
        """Fetch a single waveform. Called by timer in the worker thread."""
        if self._stop_event.is_set() or self._busy:
            return
        self._busy = True
        try:
            waveform = get_waveform(self.sock, self.channel)
            if not self._stop_event.is_set() and len(waveform) > 0:
                self.waveform_ready.emit(waveform)
        except Exception as e:
            if not self._stop_event.is_set():
                self.error_occurred.emit(str(e))
        finally:
            self._busy = False


class ScopeStreamer:
    def __init__(self, channel='C1', interval=50, ip=None):
        global scope_ip
        if ip:
            scope_ip = ip

        self.channel = channel
        self.interval = interval
        self.streaming = False  # Display only mode
        self.capturing = False  # Display + save mode
        self.sock = None
        self._closing = False
        self.capture_dir = Path('captures')
        self.capture_dir.mkdir(exist_ok=True)

        # Connect to scope
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((scope_ip, port))
        self.sock.settimeout(5)

        # Query scope ID
        idn = query(self.sock, '*IDN?')
        log.info(f"Connected to: {idn.decode().strip()}")

        # Get initial scale parameters
        vdiv, offset = get_scale_params(self.sock, channel)
        log.info(f"V/div: {vdiv}, Offset: {offset}")

        # Ensure scope is running in auto trigger mode
        send_command(self.sock, 'TRMD AUTO')
        log.info("Scope set to AUTO trigger mode")



        # Set up pyqtgraph
        pg.setConfigOptions(antialias=False, useOpenGL=True)

        self.app = QtWidgets.QApplication([])

        # Enable Ctrl+C to work
        signal.signal(signal.SIGINT, self._signal_handler)
        # Process events periodically so signals are handled
        self.signal_timer = QtCore.QTimer()
        self.signal_timer.timeout.connect(lambda: None)
        self.signal_timer.start(100)
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle(f'Siglent Oscilloscope - {channel} Live Stream')
        self.win.resize(1200, 600)
        self.win.closeEvent = self._handle_close_event

        main_layout = QtWidgets.QVBoxLayout(self.win)
        controls_layout = QtWidgets.QHBoxLayout()

        # Stream button (display only, no saving)
        self.stream_btn = QtWidgets.QPushButton('Start Stream')
        self.stream_btn.clicked.connect(self.toggle_stream)
        controls_layout.addWidget(self.stream_btn)

        # Capture button (display + save to disk)
        self.capture_btn = QtWidgets.QPushButton('Start Capture')
        self.capture_btn.clicked.connect(self.toggle_capture)
        controls_layout.addWidget(self.capture_btn)

        controls_layout.addSpacing(20)

        # Frequency slider
        freq_label = QtWidgets.QLabel('Update freq:')
        controls_layout.addWidget(freq_label)

        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.freq_slider.setMinimum(1)    # 1 Hz
        self.freq_slider.setMaximum(50)   # 50 Hz
        self.freq_slider.setValue(1000 // interval)  # Convert ms to Hz
        self.freq_slider.setFixedWidth(150)
        self.freq_slider.valueChanged.connect(self._on_freq_changed)
        controls_layout.addWidget(self.freq_slider)

        self.freq_value_label = QtWidgets.QLabel(f'{1000 // interval} Hz')
        self.freq_value_label.setFixedWidth(50)
        controls_layout.addWidget(self.freq_value_label)

        controls_layout.addStretch(1)
        main_layout.addLayout(controls_layout)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('#2b2b2b')
        main_layout.addWidget(self.plot_widget)

        # Create plot
        self.plot = self.plot_widget.addPlot(title=f'{channel} Live Stream')
        self.plot.setLabel('left', 'Voltage', units='V')
        self.plot.setLabel('bottom', 'Sample')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.getViewBox().setBackgroundColor('k')

        # Create curve with yellow color (oscilloscope style)
        self.curve = self.plot.plot(pen=pg.mkPen(color='y', width=1))

        # Set Y range based on V/div
        if scale_cache['vdiv']:
            yrange = scale_cache['vdiv'] * 4
            center = -(scale_cache['offset'] or 0)
            self.plot.setYRange(center - yrange, center + yrange)

        # Set up worker thread for non-blocking captures
        self.worker_thread = QtCore.QThread()
        self.worker = CaptureWorker(self.sock, self.channel)
        self.worker.moveToThread(self.worker_thread)

        # Timer runs in worker thread to trigger captures
        self.capture_timer = QtCore.QTimer()
        self.capture_timer.moveToThread(self.worker_thread)
        self.capture_timer.timeout.connect(self.worker.run_capture)

        # Connect worker signals to UI slots
        self.worker.waveform_ready.connect(self._on_waveform_ready)
        self.worker.error_occurred.connect(self._on_error)

        # Start worker thread
        self.worker_thread.start()

        # Register cleanup on exit
        atexit.register(self.cleanup)

        self.close_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Close), self.win
        )
        self.close_shortcut.activated.connect(self.win.close)

        # Frame rate tracking
        self.last_time = time.time()
        self.fps_counter = 0
        self.fps = 0

        self.win.show()

    def _signal_handler(self, signum, frame):
        log.info("Exiting...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        if self._closing:
            return
        self._closing = True

        # Stop the worker and timer
        if hasattr(self, 'worker'):
            self.worker.stop()
        if hasattr(self, 'capture_timer'):
            QtCore.QMetaObject.invokeMethod(
                self.capture_timer, 'stop', QtCore.Qt.ConnectionType.BlockingQueuedConnection
            )

        # Quit and wait for worker thread
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(2000)  # Wait up to 2 seconds

        if hasattr(self, 'signal_timer'):
            self.signal_timer.stop()

        # Properly close the socket connection
        if self.sock:
            try:
                # Clear any pending data and close gracefully
                self.sock.settimeout(0.1)
                try:
                    while self.sock.recv(4096):
                        pass
                except:
                    pass
                self.sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            finally:
                self.sock.close()
                self.sock = None

        self.app.quit()

    def _handle_close_event(self, event):
        self.cleanup()
        event.accept()

    def _on_freq_changed(self, value):
        """Handle frequency slider change."""
        self.interval = 1000 // value  # Convert Hz to ms
        self.freq_value_label.setText(f'{value} Hz')

        # If currently running, update the timer interval
        if self.streaming or self.capturing:
            QtCore.QMetaObject.invokeMethod(
                self.capture_timer, 'setInterval', QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(int, self.interval)
            )

    def _start_timer(self):
        """Start the capture timer in the worker thread."""
        self.worker.reset()
        QtCore.QMetaObject.invokeMethod(
            self.capture_timer, 'start', QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(int, self.interval)
        )

    def _stop_timer(self):
        """Stop the capture timer."""
        self.worker.stop()
        QtCore.QMetaObject.invokeMethod(
            self.capture_timer, 'stop', QtCore.Qt.ConnectionType.QueuedConnection
        )

    def toggle_stream(self):
        """Toggle streaming mode (display only, no saving)."""
        if self.streaming:
            self._stop_timer()
            self.stream_btn.setText('Start Stream')
            self.capture_btn.setEnabled(True)
            self.streaming = False
        else:
            # Stop capture if running
            if self.capturing:
                self.toggle_capture()

            self._start_timer()
            self.stream_btn.setText('Stop Stream')
            self.capture_btn.setEnabled(False)
            self.streaming = True

    def toggle_capture(self):
        """Toggle capture mode (display + save to disk)."""
        if self.capturing:
            self._stop_timer()
            self.capture_btn.setText('Start Capture')
            self.stream_btn.setEnabled(True)
            log.info(f"Capture stopped. Saved to {self.current_capture_dir}")
            self.capturing = False
        else:
            # Stop streaming if running
            if self.streaming:
                self.toggle_stream()

            # Create new capture directory
            timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            self.current_capture_dir = self.capture_dir / f"cap_{timestamp}"
            self.current_capture_dir.mkdir(exist_ok=True)
            log.info(f"Starting capture to {self.current_capture_dir}")

            self._start_timer()
            self.capture_btn.setText('Stop Capture')
            self.stream_btn.setEnabled(False)
            self.capturing = True

    def _on_waveform_ready(self, waveform):
        """Handle waveform data from worker thread (runs on main thread)."""
        if not self.streaming and not self.capturing:
            return

        # Update display
        self.curve.setData(waveform)
        self.plot.setXRange(0, len(waveform), padding=0)

        # Only save when capturing
        if self.capturing:
            self._save_capture(waveform)

        # FPS tracking
        self.fps_counter += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.last_time = now
            mode = 'Capturing' if self.capturing else 'Streaming'
            self.win.setWindowTitle(
                f'Siglent Oscilloscope - {self.channel} [{mode}] ({self.fps} fps)'
            )

    def _on_error(self, error_msg):
        """Handle errors from worker thread."""
        log.error(f"Error reading waveform: {error_msg}")

    def _save_capture(self, waveform):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = self.current_capture_dir / f"{self.channel}_{timestamp}.npy"
        np.save(filename, waveform.astype(np.float32))

    def run(self):
        try:
            self.app.exec()
        finally:
            self.cleanup()


def stream_waveform(channel='C1', interval=50, ip=None):
    """Stream waveform data with live plot updates using pyqtgraph."""
    streamer = ScopeStreamer(channel=channel, interval=interval, ip=ip)
    streamer.run()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stream waveform from Siglent oscilloscope')
    parser.add_argument('-c', '--channel', default='C1', help='Channel to stream (default: C1)')
    parser.add_argument('-i', '--interval', type=int, default=50, help='Update interval in ms (default: 50)')
    parser.add_argument('--ip', default=scope_ip, help=f'Scope IP address (default: {scope_ip})')
    args = parser.parse_args()

    log.info(f"Connecting to Siglent scope at {args.ip}...")
    stream_waveform(channel=args.channel, interval=args.interval, ip=args.ip)

if __name__ == '__main__':
    main()
