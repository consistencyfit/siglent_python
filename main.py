import socket
import signal
import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

scope_ip = '192.168.1.142'
port = 5025  # SCPI port for Siglent scopes

# Cache for scaling parameters (to avoid querying every frame)
scale_cache = {'vdiv': None, 'offset': None, 'tdiv': None}

def send_command(sock, cmd):
    """Send a command to the scope."""
    sock.sendall((cmd + '\n').encode())
    time.sleep(0.05)

def query(sock, cmd, timeout=2):
    """Send a query and return the response."""
    sock.sendall((cmd + '\n').encode())
    sock.settimeout(timeout)
    time.sleep(0.1)
    response = b''
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            response += chunk
            if len(chunk) < 65536:
                break
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

def get_waveform(sock, channel='C1'):
    """Retrieve waveform data from the specified channel."""
    # Get the actual waveform data
    data_raw = query(sock, f'{channel}:WF? DAT2', timeout=3)

    # Parse the binary block data (IEEE 488.2 format: #NXXXXXXXX<data>)
    if b'#' not in data_raw:
        return np.array([])

    hash_pos = data_raw.find(b'#')
    if hash_pos == -1 or len(data_raw) <= hash_pos + 2:
        return np.array([])

    # Get number of digits describing the length
    try:
        num_digits = int(chr(data_raw[hash_pos + 1]))
    except ValueError:
        return np.array([])

    if len(data_raw) <= hash_pos + 2 + num_digits:
        return np.array([])

    # Get data length and start position
    length_str = data_raw[hash_pos + 2:hash_pos + 2 + num_digits]
    try:
        data_length = int(length_str)
    except ValueError:
        return np.array([])

    data_start = hash_pos + 2 + num_digits
    waveform_data = data_raw[data_start:data_start + data_length]

    if len(waveform_data) < 2:
        return np.array([])

    # Remove trailing garbage bytes (usually 2)
    if len(waveform_data) > 2:
        waveform_data = waveform_data[:-2]

    # Get scale parameters
    vdiv = scale_cache.get('vdiv') or 1.0
    offset = scale_cache.get('offset') or 0.0

    # Convert bytes to voltage: data * vDiv / 25 - vOffset
    values = np.frombuffer(waveform_data, dtype=np.int8)
    voltage = (values.astype(float) * vdiv / 25.0) - offset

    return voltage


class ScopeStreamer:
    def __init__(self, channel='C1', interval=50, ip=None):
        global scope_ip
        if ip:
            scope_ip = ip

        self.channel = channel
        self.interval = interval
        self.sock = None
        self._closing = False

        # Connect to scope
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((scope_ip, port))
        self.sock.settimeout(5)

        # Query scope ID
        idn = query(self.sock, '*IDN?')
        print(f"Connected to: {idn.decode().strip()}")

        # Get initial scale parameters
        vdiv, offset = get_scale_params(self.sock, channel)
        print(f"V/div: {vdiv}, Offset: {offset}")

        # Set up pyqtgraph
        pg.setConfigOptions(antialias=False, useOpenGL=True)

        self.app = QtWidgets.QApplication([])

        # Enable Ctrl+C to work
        signal.signal(signal.SIGINT, self._signal_handler)
        # Process events periodically so signals are handled
        self.signal_timer = QtCore.QTimer()
        self.signal_timer.timeout.connect(lambda: None)
        self.signal_timer.start(100)
        self.win = pg.GraphicsLayoutWidget(title=f'Siglent Oscilloscope - {channel} Live Stream')
        self.win.resize(1200, 600)
        self.win.setBackground('#2b2b2b')
        self.win.closeEvent = self._handle_close_event

        # Create plot
        self.plot = self.win.addPlot(title=f'{channel} Live Stream')
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

        # Set up timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(interval)
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
        print("\nExiting...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        if self._closing:
            return
        self._closing = True
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'signal_timer'):
            self.signal_timer.stop()
        if self.sock:
            self.sock.close()
            self.sock = None
        self.app.quit()

    def _handle_close_event(self, event):
        self.cleanup()
        event.accept()

    def update(self):
        try:
            waveform = get_waveform(self.sock, self.channel)
            if len(waveform) > 0:
                self.curve.setData(waveform)

                # Update X range to match data length
                self.plot.setXRange(0, len(waveform), padding=0)

            # FPS tracking
            self.fps_counter += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_time = now
                self.win.setWindowTitle(
                    f'Siglent Oscilloscope - {self.channel} Live Stream ({self.fps} fps)'
                )

        except Exception as e:
            print(f"Error reading waveform: {e}")

    def run(self):
        try:
            self.app.exec()
        finally:
            self.cleanup()


def stream_waveform(channel='C1', interval=50, ip=None):
    """Stream waveform data with live plot updates using pyqtgraph."""
    streamer = ScopeStreamer(channel=channel, interval=interval, ip=ip)
    streamer.run()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stream waveform from Siglent oscilloscope')
    parser.add_argument('-c', '--channel', default='C1', help='Channel to stream (default: C1)')
    parser.add_argument('-i', '--interval', type=int, default=50, help='Update interval in ms (default: 50)')
    parser.add_argument('--ip', default=scope_ip, help=f'Scope IP address (default: {scope_ip})')
    args = parser.parse_args()

    print(f"Connecting to Siglent scope at {args.ip}...")
    stream_waveform(channel=args.channel, interval=args.interval, ip=args.ip)
