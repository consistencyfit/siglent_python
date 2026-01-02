#!/usr/bin/env python3
"""Reset Siglent scope connection."""
import socket
import sys

scope_ip = sys.argv[1] if len(sys.argv) > 1 else '192.168.1.142'
port = 5025

print(f"Attempting to reset scope at {scope_ip}...")

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    sock.connect((scope_ip, port))

    # Send reset command
    sock.sendall(b'*RST\n')
    sock.sendall(b'*CLS\n')  # Clear status

    # Properly close
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except:
        pass
    sock.close()
    print("Scope reset successfully")
except ConnectionRefusedError:
    print("Connection refused - scope may need power cycle")
except socket.timeout:
    print("Connection timed out - scope may need power cycle")
except Exception as e:
    print(f"Error: {e}")
