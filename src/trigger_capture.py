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
    parser.add_argument('--trigger-level', type=float, default=0.02, help='Trigger level in volts (default: 0.02 = 20mV)')
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

        # Set trigger mode to NORMAL
        print("Setting trigger mode: NORMAL")
        send_command(sock, 'TRMD NORM')
        time.sleep(0.1)

        # Configure trigger
        print(f"Setting trigger: {args.trigger_source} @ {args.trigger_level}V")
        send_command(sock, f'TRSE EDGE,SR,{args.trigger_source},HT,OFF')  # Edge trigger on source
        send_command(sock, f'{args.trigger_source}:TRLV {args.trigger_level}V')  # Trigger level

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

            # Capture screen
            print("Capturing screen...")
            bmp_data = get_screen_capture(sock)

            if len(bmp_data) > 0:
                # Determine output filename
                if args.output:
                    output_file = Path(args.output)
                else:
                    captures_dir = Path('captures')
                    captures_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = captures_dir / f"capture_{timestamp}.bmp"

                # Save the capture
                output_file.write_bytes(bmp_data)
                print(f"Saved capture to: {output_file} ({len(bmp_data)} bytes)")
            else:
                print("Error: No data received from screen capture")
        else:
            print("No trigger occurred within timeout period")
            # Switch back to AUTO mode
            send_command(sock, 'TRMD AUTO')

    finally:
        # Restore AUTO trigger mode and close socket
        try:
            send_command(sock, 'TRMD AUTO')
        except:
            pass
        sock.close()
        print("Connection closed")


if __name__ == '__main__':
    main()
