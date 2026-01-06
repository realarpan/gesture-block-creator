import cv2
import mediapipe as mp
import numpy as np
import random
from enum import Enum

class GestureMode(Enum):
    IDLE = 0
    CREATING = 1
    SELECTING = 2
    MOVING = 3

class Block:
    def __init__(self, x, y, width=80, height=80):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.selected = False
    
    def contains_point(self, x, y):
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def draw(self, frame):
        color = self.color if not self.selected else tuple([min(c + 50, 255) for c in self.color])
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     color, -1)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     (255, 255, 255), 2)

class GestureBlockCreator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.blocks = []
        self.mode = GestureMode.IDLE
        self.selected_block = None
        self.prev_x, self.prev_y = 0, 0
        
        # Gesture detection parameters
        self.pinch_threshold = 30
        self.selection_frames = 0
        self.selection_threshold = 10
        
    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def detect_pinch(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        
        distance = self.calculate_distance((thumb_x, thumb_y), (index_x, index_y))
        
        return distance < self.pinch_threshold, (index_x, index_y)
    
    def detect_fist(self, landmarks):
        # Check if all fingers are folded
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_pips = [6, 10, 14, 18]  # Corresponding PIP joints
        
        folded_count = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y > landmarks[pip].y:
                folded_count += 1
        
        return folded_count >= 3
    
    def detect_open_palm(self, landmarks):
        # Check if all fingers are extended
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        extended_count = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                extended_count += 1
        
        return extended_count >= 3
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                h, w = frame.shape[:2]
                
                # Detect gestures
                is_pinching, pinch_pos = self.detect_pinch(landmarks, frame.shape)
                is_fist = self.detect_fist(landmarks)
                is_open_palm = self.detect_open_palm(landmarks)
                
                # Get index finger tip position
                index_tip = landmarks[8]
                cursor_x, cursor_y = int(index_tip.x * w), int(index_tip.y * h)
                
                # Draw cursor
                cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), -1)
                
                # State machine for gesture control
                if self.mode == GestureMode.IDLE:
                    if is_open_palm:
                        self.mode = GestureMode.CREATING
                        self.selection_frames = 0
                    elif is_pinching:
                        # Check if pinching over a block
                        for block in self.blocks:
                            if block.contains_point(pinch_pos[0], pinch_pos[1]):
                                self.selection_frames += 1
                                if self.selection_frames > self.selection_threshold:
                                    self.mode = GestureMode.SELECTING
                                    self.selected_block = block
                                    block.selected = True
                                    self.prev_x, self.prev_y = pinch_pos
                                break
                        else:
                            self.selection_frames = 0
                
                elif self.mode == GestureMode.CREATING:
                    if not is_open_palm:
                        # Create block at current position
                        new_block = Block(cursor_x - 40, cursor_y - 40)
                        self.blocks.append(new_block)
                        self.mode = GestureMode.IDLE
                
                elif self.mode == GestureMode.SELECTING:
                    if is_pinching:
                        self.mode = GestureMode.MOVING
                    else:
                        if self.selected_block:
                            self.selected_block.selected = False
                        self.selected_block = None
                        self.mode = GestureMode.IDLE
                
                elif self.mode == GestureMode.MOVING:
                    if is_pinching and self.selected_block:
                        # Move the selected block
                        dx = pinch_pos[0] - self.prev_x
                        dy = pinch_pos[1] - self.prev_y
                        
                        self.selected_block.x += dx
                        self.selected_block.y += dy
                        
                        # Keep block within bounds
                        self.selected_block.x = max(0, min(self.selected_block.x, w - self.selected_block.width))
                        self.selected_block.y = max(0, min(self.selected_block.y, h - self.selected_block.height))
                        
                        self.prev_x, self.prev_y = pinch_pos
                    else:
                        # Release block
                        if self.selected_block:
                            self.selected_block.selected = False
                        self.selected_block = None
                        self.mode = GestureMode.IDLE
                
                # Delete block with fist gesture
                if is_fist and self.mode == GestureMode.IDLE:
                    for i, block in enumerate(self.blocks):
                        if block.contains_point(cursor_x, cursor_y):
                            self.blocks.pop(i)
                            break
        
        # Draw all blocks
        for block in self.blocks:
            block.draw(frame)
        
        # Display instructions and mode
        self.draw_ui(frame)
        
        return frame
    
    def draw_ui(self, frame):
        # Background for instructions
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Instructions
        instructions = [
            "GESTURE CONTROLS:",
            "Open Palm -> Create Block",
            "Pinch -> Select Block",
            "Pinch + Move -> Move Block",
            "Fist over Block -> Delete Block",
            f"Mode: {self.mode.name} | Blocks: {len(self.blocks)}"
        ]
        
        y_offset = 30
        for i, text in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (20, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Gesture Block Creator Started!")
        print("Press 'q' to quit, 'c' to clear all blocks")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            cv2.imshow('Gesture Block Creator', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.blocks.clear()
                print("All blocks cleared!")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    app = GestureBlockCreator()
    app.run()
