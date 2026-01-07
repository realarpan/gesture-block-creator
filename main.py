import cv2
import mediapipe as mp
import numpy as np
import random
import json
import time
from enum import Enum
from pathlib import Path
from collections import deque

class GestureMode(Enum):
    IDLE = 0
    CREATING = 1
    SELECTING = 2
    MOVING = 3
    RESIZING = 4
    ROTATING = 5

class BlockShape(Enum):
    RECTANGLE = 0
    CIRCLE = 1
    TRIANGLE = 2

class Block:
    def __init__(self, x, y, width=80, height=80, shape=BlockShape.RECTANGLE):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.initial_width = width
        self.initial_height = height
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.selected = False
        self.rotation = 0  # Rotation angle in degrees
        self.shape = shape
        self.creation_time = time.time()
        self.opacity = 1.0
        
    def contains_point(self, x, y):
        if self.shape == BlockShape.RECTANGLE:
            return (self.x <= x <= self.x + self.width and 
                    self.y <= y <= self.y + self.height)
        elif self.shape == BlockShape.CIRCLE:
            center_x = self.x + self.width // 2
            center_y = self.y + self.height // 2
            radius = self.width // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            return distance <= radius
        return False
    
    def get_center(self):
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def draw(self, frame):
        color = self.color if not self.selected else tuple([min(c + 80, 255) for c in self.color])
        
        if self.shape == BlockShape.RECTANGLE:
            overlay = frame.copy()
            cv2.rectangle(overlay, (self.x, self.y), 
                         (self.x + self.width, self.y + self.height), 
                         color, -1)
            cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0, frame)
            cv2.rectangle(frame, (self.x, self.y), 
                         (self.x + self.width, self.y + self.height), 
                         (255, 255, 255), 2 if self.selected else 1)
            
        elif self.shape == BlockShape.CIRCLE:
            center = self.get_center()
            radius = self.width // 2
            overlay = frame.copy()
            cv2.circle(overlay, center, radius, color, -1)
            cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0, frame)
            cv2.circle(frame, center, radius, (255, 255, 255), 2 if self.selected else 1)
            
        elif self.shape == BlockShape.TRIANGLE:
            center = self.get_center()
            pts = np.array([
                [center[0], self.y],
                [self.x + self.width, self.y + self.height],
                [self.x, self.y + self.height]
            ], dtype=np.int32)
            overlay = frame.copy()
            cv2.drawContours(overlay, [pts], 0, color, -1)
            cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0, frame)
            cv2.drawContours(frame, [pts], 0, (255, 255, 255), 2 if self.selected else 1)
        
        # Draw rotation indicator if rotated
        if self.rotation != 0 and self.selected:
            cv2.putText(frame, f"Rot: {self.rotation}°", 
                       (self.x, self.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 255, 255), 1)

class GestureBlockCreator:
    def __init__(self, config_path='config.json'):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        self.blocks = []
        self.mode = GestureMode.IDLE
        self.selected_block = None
        self.prev_x, self.prev_y = 0, 0
        self.prev_distance = 0
        
        # Gesture detection parameters
        self.pinch_threshold = self.config['pinch_threshold']
        self.selection_frames = 0
        self.selection_threshold = self.config['selection_threshold']
        self.pinch_spread_threshold = self.config['pinch_spread_threshold']
        
        # Performance metrics
        self.fps_deque = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = time.time()
        
        # Hand gesture history for swipe detection
        self.hand_history = deque(maxlen=10)
        self.gesture_history = deque(maxlen=5)
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        default_config = {
            'pinch_threshold': 30,
            'selection_threshold': 10,
            'pinch_spread_threshold': 80,
            'block_size': 80,
            'camera_width': 1280,
            'camera_height': 720,
            'show_hand_landmarks': True,
            'auto_save': True,
            'save_path': 'blocks_save.json'
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def detect_pinch(self, landmarks, frame_shape):
        """Detect pinch gesture"""
        h, w = frame_shape[:2]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        
        distance = self.calculate_distance((thumb_x, thumb_y), (index_x, index_y))
        return distance < self.pinch_threshold, (index_x, index_y), distance
    
    def detect_pinch_spread(self, landmarks, frame_shape):
        """Detect pinch spread gesture (for resizing)"""
        h, w = frame_shape[:2]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
        
        dist1 = self.calculate_distance((thumb_x, thumb_y), (index_x, index_y))
        dist2 = self.calculate_distance((thumb_x, thumb_y), (middle_x, middle_y))
        
        return (dist1 > 50 and dist2 > 50), max(dist1, dist2)
    
    def detect_fist(self, landmarks):
        """Detect fist gesture"""
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        folded_count = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y > landmarks[pip].y:
                folded_count += 1
        
        return folded_count >= 3
    
    def detect_open_palm(self, landmarks):
        """Detect open palm gesture"""
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        extended_count = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                extended_count += 1
        
        return extended_count >= 3
    
    def detect_thumbs_up(self, landmarks):
        """Detect thumbs up gesture"""
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        index_pip = landmarks[6]
        
        return (thumb_tip.y < thumb_mcp.y and 
                thumb_tip.y < index_pip.y)
    
    def detect_peace_sign(self, landmarks, frame_shape):
        """Detect peace sign (for color cycling)"""
        h, w = frame_shape[:2]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        
        index_y = index_tip.y
        middle_y = middle_tip.y
        ring_y = ring_tip.y
        
        index_pip = landmarks[6].y
        middle_pip = landmarks[10].y
        ring_pip = landmarks[14].y
        
        return (index_y < index_pip and 
                middle_y < middle_pip and 
                ring_y > ring_pip)
    
    def detect_swipe(self, hand_pos):
        """Detect swipe gesture"""
        self.hand_history.append(hand_pos)
        
        if len(self.hand_history) < 5:
            return None
        
        first = self.hand_history[0]
        last = self.hand_history[-1]
        
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        
        if abs(dx) > 100 and abs(dy) < 50:
            return "right" if dx > 0 else "left"
        elif abs(dy) > 100 and abs(dx) < 50:
            return "down" if dy > 0 else "up"
        
        return None
    
    def save_blocks(self, filepath=None):
        """Save block positions and properties to JSON"""
        if filepath is None:
            filepath = self.config['save_path']
        
        blocks_data = []
        for block in self.blocks:
            blocks_data.append({
                'x': block.x,
                'y': block.y,
                'width': block.width,
                'height': block.height,
                'color': block.color,
                'shape': block.shape.name,
                'rotation': block.rotation
            })
        
        with open(filepath, 'w') as f:
            json.dump(blocks_data, f, indent=2)
        
        print(f"Blocks saved to {filepath}")
    
    def load_blocks(self, filepath=None):
        """Load block positions and properties from JSON"""
        if filepath is None:
            filepath = self.config['save_path']
        
        if not Path(filepath).exists():
            print(f"No save file found at {filepath}")
            return
        
        try:
            with open(filepath, 'r') as f:
                blocks_data = json.load(f)
            
            self.blocks.clear()
            for block_data in blocks_data:
                block = Block(
                    block_data['x'],
                    block_data['y'],
                    block_data['width'],
                    block_data['height'],
                    BlockShape[block_data['shape']]
                )
                block.color = tuple(block_data['color'])
                block.rotation = block_data.get('rotation', 0)
                self.blocks.append(block)
            
            print(f"Blocks loaded from {filepath}")
        except Exception as e:
            print(f"Error loading blocks: {e}")
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        h, w = frame.shape[:2]
        
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if self.config['show_hand_landmarks']:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                
                # Detect gestures
                is_pinching, pinch_pos, pinch_dist = self.detect_pinch(landmarks, frame.shape)
                is_pinch_spread, spread_dist = self.detect_pinch_spread(landmarks, frame.shape)
                is_fist = self.detect_fist(landmarks)
                is_open_palm = self.detect_open_palm(landmarks)
                is_thumbs_up = self.detect_thumbs_up(landmarks)
                is_peace = self.detect_peace_sign(landmarks, frame.shape)
                
                # Get hand center
                hand_center_x = int(landmarks[9].x * w)
                hand_center_y = int(landmarks[9].y * h)
                
                # Get index finger tip position
                index_tip = landmarks[8]
                cursor_x, cursor_y = int(index_tip.x * w), int(index_tip.y * h)
                
                # Draw cursor
                cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), -1)
                
                # Multi-hand operations
                if hand_count >= 2 and hand_idx == 1:
                    if is_pinching and self.selected_block:
                        # Two-hand rotation
                        self.mode = GestureMode.ROTATING
                
                # State machine
                if self.mode == GestureMode.IDLE:
                    if is_open_palm:
                        self.mode = GestureMode.CREATING
                        self.selection_frames = 0
                    elif is_thumbs_up and self.selected_block:
                        # Cycle shape
                        shapes = list(BlockShape)
