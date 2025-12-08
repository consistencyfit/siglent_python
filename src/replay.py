import sys
import argparse
from pathlib import Path
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

class ReplayPlayer:
    def __init__(self, capture_dir, fps=30):
        self.capture_dir = Path(capture_dir)
        if not self.capture_dir.exists():
            print(f"Error: Directory {capture_dir} does not exist.")
            sys.exit(1)

        self.files = sorted(list(self.capture_dir.glob('*.npy')))
        if not self.files:
            print(f"No .npy files found in {capture_dir}")
            sys.exit(1)
        
        print(f"Found {len(self.files)} frames.")
        
        self.fps = fps
        self.current_frame_idx = 0
        self.playing = True
        self.loop = True

        # Set up UI
        self.app = QtWidgets.QApplication([])
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle(f'Replay - {self.capture_dir.name}')
        self.win.resize(1000, 600)
        
        layout = QtWidgets.QVBoxLayout(self.win)
        
        # Plot
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('#2b2b2b')
        self.plot = self.plot_widget.addPlot(title=f'Replay: {self.capture_dir.name}')
        self.plot.setLabel('left', 'Voltage', units='V')
        self.plot.setLabel('bottom', 'Sample')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=pg.mkPen(color='y', width=1))
        
        layout.addWidget(self.plot_widget)
        
        # Controls
        controls = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton('Pause')
        self.play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_btn)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.files) - 1)
        self.slider.valueChanged.connect(self.set_frame)
        controls.addWidget(self.slider)
        
        self.label = QtWidgets.QLabel(f"Frame: 0/{len(self.files)}")
        controls.addWidget(self.label)
        
        layout.addLayout(controls)

        self.win.show()
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000 / self.fps))

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText('Pause' if self.playing else 'Play')

    def set_frame(self, index):
        self.current_frame_idx = index
        self.display_frame()

    def update(self):
        if not self.playing:
            return
            
        self.current_frame_idx += 1
        if self.current_frame_idx >= len(self.files):
            if self.loop:
                self.current_frame_idx = 0
            else:
                self.current_frame_idx = len(self.files) - 1
                self.playing = False
                self.play_btn.setText('Play')
        
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)
        self.display_frame()

    def display_frame(self):
        if 0 <= self.current_frame_idx < len(self.files):
            file = self.files[self.current_frame_idx]
            try:
                data = np.load(file)
                self.curve.setData(data)
                self.label.setText(f"Frame: {self.current_frame_idx + 1}/{len(self.files)}")
            except Exception as e:
                print(f"Error loading frame {file}: {e}")

    def run(self):
        self.app.exec()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay captured waveforms')
    parser.add_argument('directory', help='Directory containing .npy capture files')
    parser.add_argument('--fps', type=int, default=30, help='Playback FPS')
    
    args = parser.parse_args()
    
    player = ReplayPlayer(args.directory, fps=args.fps)
    player.run()
