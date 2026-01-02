#!/usr/bin/env python
"""
Trigger capture script for Siglent oscilloscopes.

Configures the scope with specified settings and saves a capture when the trigger fires.
"""
import argparse
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

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
            # For simple text queries, check for newline
            if response.endswith(b'\n'):
                break
    except socket.timeout:
        pass

    return response.decode().strip()


def get_screen_capture(sock, timeout=5):
    """Get screen capture (BMP) from the scope."""
    sock.sendall(b'SCDP\n')
    sock.settimeout(timeout)
    time.sleep(1)

    data = b''
    while True:
        try:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
        except socket.timeout:
            break

    return data


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

            # Parse IEEE 488.2 header to get expected length
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


def get_waveform(sock, channel='C1', vdiv=None, offset=0.0):
    """Get waveform data from the specified channel."""
    # Query vdiv if not provided
    if vdiv is None:
        vdiv_resp = query(sock, f'{channel}:VDIV?')
        try:
            vdiv = float(vdiv_resp.split()[-1].replace('V', ''))
        except:
            vdiv = 1.0

    # Get waveform data
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

    # Convert bytes to voltage
    values = np.frombuffer(waveform_data, dtype=np.int8)
    voltage = (values.astype(float) * vdiv / 25.0) - offset

    return voltage, vdiv


def compute_envelope(signal):
    """Compute the envelope of a signal using Hilbert transform."""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope


def plot_waveform_with_envelope(waveform, envelope, tdiv, channel='C1', save_path=None):
    """Plot waveform with its envelope overlay."""
    num_points = len(waveform)
    # 14 divisions on screen, calculate time axis
    total_time = tdiv * 14
    time_axis = np.linspace(0, total_time * 1000, num_points)  # in ms

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_axis, waveform * 1000, 'y-', linewidth=0.5, alpha=0.7, label=f'{channel} Signal')
    ax.plot(time_axis, envelope * 1000, 'r-', linewidth=1.5, label='Envelope (upper)')
    ax.plot(time_axis, -envelope * 1000, 'r-', linewidth=1.5, label='Envelope (lower)')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title(f'{channel} Waveform with Envelope')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('#2b2b2b')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved plot to: {save_path}")

    plt.show()


def plot_dual_channel_envelopes(wf1, env1, wf2, env2, tdiv, save_path=None):
    """Plot both channels with their envelopes."""
    num_points = max(len(wf1), len(wf2))
    total_time = tdiv * 14
    time_axis1 = np.linspace(0, total_time * 1000, len(wf1))
    time_axis2 = np.linspace(0, total_time * 1000, len(wf2))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # CH1
    ax1.plot(time_axis1, wf1 * 1000, 'y-', linewidth=0.5, alpha=0.7, label='CH1 Signal')
    ax1.plot(time_axis1, env1 * 1000, 'r-', linewidth=1.5, label='Envelope')
    ax1.plot(time_axis1, -env1 * 1000, 'r-', linewidth=1.5)
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_title('CH1 Waveform with Envelope')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('black')

    # CH2
    ax2.plot(time_axis2, wf2 * 1000, 'c-', linewidth=0.5, alpha=0.7, label='CH2 Signal')
    ax2.plot(time_axis2, env2 * 1000, 'm-', linewidth=1.5, label='Envelope')
    ax2.plot(time_axis2, -env2 * 1000, 'm-', linewidth=1.5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.set_title('CH2 Waveform with Envelope')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('black')

    fig.patch.set_facecolor('#2b2b2b')
    for ax in (ax1, ax2):
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved plot to: {save_path}")

    plt.show()


def wait_for_trigger(sock, timeout=30):
    """Wait for the trigger to fire. Returns True if triggered, False if timeout."""
    print(f"Waiting for trigger (timeout: {timeout}s)...")
    start_time = time.time()

    # Clear INR register
    query(sock, 'INR?')

    while time.time() - start_time < timeout:
        try:
            resp = query(sock, 'INR?', timeout=0.5)
            # Parse the INR value - bit 0 indicates new acquisition
            parts = resp.split()
            if parts:
                inr_val = int(parts[-1])
                if inr_val & 1:  # Bit 0 set = new data acquired
                    print("Trigger fired!")
                    return True
        except (ValueError, socket.timeout):
            pass
        time.sleep(0.1)

    print("Timeout waiting for trigger")
    return False


def main():
    parser = argparse.ArgumentParser(description='Configure scope and capture on trigger')
    parser.add_argument('--ip', default=SCOPE_IP, help=f'Scope IP address (default: {SCOPE_IP})')
    parser.add_argument('--trigger-level', type=float, default=0.005, help='Trigger level in volts (default: 0.005 = 5mV)')
    parser.add_argument('--trigger-source', default='C1', help='Trigger source channel (default: C1)')
    parser.add_argument('--hdiv', type=float, default=0.02, help='Horizontal time/div in seconds (default: 0.02 = 20ms)')
    parser.add_argument('--ch1-vdiv', type=float, default=0.07, help='CH1 voltage/div in volts (default: 0.07 = 70mV)')
    parser.add_argument('--ch2-vdiv', type=float, default=0.07, help='CH2 voltage/div in volts (default: 0.07 = 70mV)')
    parser.add_argument('--timeout', type=int, default=30, help='Trigger timeout in seconds (default: 30)')
    parser.add_argument('--output', '-o', default=None, help='Output filename (default: capture_TIMESTAMP.bmp)')
    args = parser.parse_args()

    # Connect to scope
    print(f"Connecting to {args.ip}:{PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.ip, PORT))
        sock.settimeout(5)
    except socket.error as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    try:
        # Get scope ID
        idn = query(sock, '*IDN?')
        print(f"Connected to: {idn}")

        # Disable screensaver
        send_command(sock, 'SCSV OFF')

        # Configure CH1
        print(f"Setting CH1: {args.ch1_vdiv*1000:.0f}mV/div, centered at 0V")
        send_command(sock, 'C1:TRA ON')  # Enable CH1 trace
        time.sleep(0.1)
        send_command(sock, f'C1:VDIV {args.ch1_vdiv}')
        time.sleep(0.1)
        send_command(sock, 'C1:OFST 0')  # Center at 0V
        time.sleep(0.1)
        query(sock, 'C1:VDIV?')  # Force processing

        # Configure CH2
        print(f"Setting CH2: {args.ch2_vdiv*1000:.0f}mV/div, centered at 0V")
        send_command(sock, 'C2:TRA ON')  # Enable CH2 trace
        time.sleep(0.1)
        send_command(sock, f'C2:VDIV {args.ch2_vdiv}')
        time.sleep(0.1)
        send_command(sock, 'C2:OFST 0')  # Center at 0V
        time.sleep(0.1)
        query(sock, 'C2:VDIV?')  # Force processing

        # Configure horizontal (time/div)
        print(f"Setting horizontal: {args.hdiv*1000:.0f}ms/div")
        send_command(sock, f'TDIV {args.hdiv}')
        time.sleep(0.1)

        # Set horizontal delay (negative = left shift)
        print("Setting horizontal delay: -120ms (left shift)")
        send_command(sock, 'TRDL -0.12')
        time.sleep(0.1)

        # Configure trigger
        print(f"Setting trigger: {args.trigger_source} @ {args.trigger_level}V")
        send_command(sock, f'TRSE EDGE,SR,{args.trigger_source},HT,OFF')  # Edge trigger on source
        send_command(sock, f'{args.trigger_source}:TRLV {args.trigger_level}V')  # Trigger level
        time.sleep(0.1)

        # Set trigger mode to NORMAL (do this last!)
        print("Setting trigger mode: NORMAL")
        send_command(sock, 'TRMD NORM')
        time.sleep(0.3)
        # Verify it took effect
        mode = query(sock, 'TRMD?')
        print(f"  Confirmed trigger mode: {mode}")

        # Verify settings
        print("\nVerifying settings:")
        print(f"  CH1 V/div: {query(sock, 'C1:VDIV?')}")
        print(f"  CH1 Offset: {query(sock, 'C1:OFST?')}")
        print(f"  CH2 V/div: {query(sock, 'C2:VDIV?')}")
        print(f"  CH2 Offset: {query(sock, 'C2:OFST?')}")
        print(f"  Time/div: {query(sock, 'TDIV?')}")
        print(f"  Horiz delay: {query(sock, 'TRDL?')}")
        print(f"  Trigger mode: {query(sock, 'TRMD?')}")
        print(f"  Trigger level: {query(sock, f'{args.trigger_source}:TRLV?')}")
        print()

        # Wait for trigger
        if wait_for_trigger(sock, timeout=args.timeout):
            # Small delay to let display update
            time.sleep(0.2)

            # Capture waveform data from both channels
            print("Capturing CH1 waveform...")
            wf1, vdiv1 = get_waveform(sock, 'C1', args.ch1_vdiv)
            print(f"  Got {len(wf1)} samples")

            print("Capturing CH2 waveform...")
            wf2, vdiv2 = get_waveform(sock, 'C2', args.ch2_vdiv)
            print(f"  Got {len(wf2)} samples")

            if len(wf1) > 0 and len(wf2) > 0:
                # Compute envelopes
                print("Computing envelopes...")
                env1 = compute_envelope(wf1)
                env2 = compute_envelope(wf2)

                # Determine output path
                captures_dir = Path('captures')
                captures_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save waveform data
                np.save(captures_dir / f"ch1_waveform_{timestamp}.npy", wf1.astype(np.float32))
                np.save(captures_dir / f"ch2_waveform_{timestamp}.npy", wf2.astype(np.float32))
                print(f"Saved waveforms to: captures/ch*_waveform_{timestamp}.npy")

                # Save envelope data
                np.save(captures_dir / f"ch1_envelope_{timestamp}.npy", env1.astype(np.float32))
                np.save(captures_dir / f"ch2_envelope_{timestamp}.npy", env2.astype(np.float32))
                print(f"Saved envelopes to: captures/ch*_envelope_{timestamp}.npy")

                # Plot both channels with envelopes
                plot_path = captures_dir / f"plot_{timestamp}.png"
                plot_dual_channel_envelopes(wf1, env1, wf2, env2, args.hdiv, save_path=plot_path)
            else:
                print("Error: No waveform data received")

            # Also capture screen
            print("Capturing screen...")
            bmp_data = get_screen_capture(sock)

            if len(bmp_data) > 0:
                if args.output:
                    output_file = Path(args.output)
                else:
                    output_file = captures_dir / f"capture_{timestamp}.bmp"
                output_file.write_bytes(bmp_data)
                print(f"Saved screenshot to: {output_file} ({len(bmp_data)} bytes)")
        else:
            print("No trigger occurred within timeout period")

    finally:
        sock.close()
        print("Connection closed")


if __name__ == '__main__':
    main()
