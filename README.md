# Gesture Block Creator 🎯

A real-time hand gesture-based system using OpenCV and MediaPipe that allows you to create, select, move, and delete virtual blocks in 2D space through intuitive hand gestures.

## Features ✨

- **Create Blocks**: Use an open palm gesture to spawn colorful blocks
- **Select Blocks**: Pinch gesture to select blocks
- **Move Blocks**: Maintain pinch and move your hand to drag blocks around
- **Delete Blocks**: Make a fist over a block to delete it
- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand landmark detection
- **Visual Feedback**: Clear UI showing current mode and block count
- **Multiple Blocks**: Create and manage multiple blocks simultaneously

## Installation 🚀

### Prerequisites

- Python 3.8 or higher
- Webcam

### Setup

1. Clone the repository:
```bash
git clone https://github.com/realarpan/gesture-block-creator.git
cd gesture-block-creator
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🎮

1. Run the application:
```bash
python main.py
```

2. Make sure your webcam is enabled and your hand is visible in the frame.

### Gesture Controls

| Gesture | Action |
|---------|--------|
| 🖐️ **Open Palm** | Create a new block at cursor position |
| 🤏 **Pinch** | Select a block (hold pinch over block) |
| 🤏➡️ **Pinch + Move** | Move the selected block |
| ✊ **Fist** | Delete block (make fist over block) |

### Keyboard Shortcuts

- **Q**: Quit the application
- **C**: Clear all blocks

## How It Works 🔧

### System Architecture

The application uses a state machine with four modes:

1. **IDLE**: Default state, waiting for gesture input
2. **CREATING**: Detected open palm, ready to create block
3. **SELECTING**: Pinch detected over a block, selecting it
4. **MOVING**: Moving a selected block with pinch gesture

### Technical Stack

- **OpenCV**: Video capture and image processing
- **MediaPipe**: Hand landmark detection and tracking
- **NumPy**: Mathematical operations and distance calculations

### Core Components

#### Block Class
Manages individual block properties:
- Position (x, y coordinates)
- Size (width, height)
- Color (random RGB)
- Selection state

#### GestureBlockCreator Class
Main application logic:
- Hand tracking initialization
- Gesture recognition (pinch, fist, open palm)
- Block management (create, select, move, delete)
- UI rendering

### Gesture Detection

- **Pinch Detection**: Calculates distance between thumb and index finger tips
- **Fist Detection**: Checks if finger tips are below their corresponding joints
- **Open Palm Detection**: Verifies fingers are extended above their joints

## Configuration ⚙️

You can modify these parameters in the code:

```python
# In GestureBlockCreator.__init__()
self.pinch_threshold = 30  # Distance for pinch detection (pixels)
self.selection_threshold = 10  # Frames to hold pinch for selection

# In Block.__init__()
width = 80  # Block width
height = 80  # Block height

# In run() method
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Camera resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

## Requirements Details 📋

- **opencv-python**: Computer vision and video processing
- **mediapipe**: Hand tracking and gesture recognition
- **numpy**: Numerical computations

## Troubleshooting 🔍

### Common Issues

1. **Webcam not detected**
   - Check if webcam is connected
   - Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or higher

2. **Poor hand detection**
   - Ensure good lighting conditions
   - Keep hand within camera frame
   - Adjust `min_detection_confidence` in MediaPipe Hands initialization

3. **Laggy performance**
   - Reduce camera resolution
   - Close other applications using webcam
   - Lower `max_num_hands` to 1

## Future Enhancements 🚀

- [ ] Save and load block configurations
- [ ] Different block shapes (circles, triangles)
- [ ] Resize gesture for blocks
- [ ] Rotation gesture
- [ ] Color picker gesture
- [ ] Export block positions to JSON
- [ ] Multi-hand collaboration mode
- [ ] 3D block visualization

## Contributing 🤝

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments 🙏

- [MediaPipe](https://mediapipe.dev/) by Google for hand tracking
- [OpenCV](https://opencv.org/) for computer vision capabilities

## Author ✍️

**Arpan**
- GitHub: [@realarpan](https://github.com/realarpan)

---

Made with ❤️ using Python, OpenCV, and MediaPipe
