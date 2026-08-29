# Bring a selected block to the front while preserving the selection.
        block = self.blocks.pop(hit)
        self.blocks.append(block)
        self.selected_index = len(self.blocks) - 1
        self.set_message("Block selected — move your pinch")

    def update_gesture(self, hand_landmarks: object, width: int, height: int) -> tuple[Optional[tuple[int, int]], bool]:
        landmarks = hand_landmarks.landmark
        thumb = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        point = self.landmark_to_point(index, width, height)
        pinching = self.is_pinching(thumb, index)

        if pinching and not self.was_pinching:
            self.begin_pinch(point, width, height)
        elif pinching and self.selected_index is not None:
            self.blocks[self.selected_index].move_to(point, width, height)
        elif not pinching and self.was_pinching:
            self.selected_index = None
            self.set_message("Released")

        self.was_pinching = pinching
        return point, pinching

    def draw_blocks(self, frame: object) -> None:
        for index, block in enumerate(self.blocks):
            half = block.size // 2
            left, top = block.x - half, block.y - half
            right, bottom = block.x + half, block.y + half
            is_selected = index == self.selected_index

            shadow_offset = 6
            cv2.rectangle(frame, (left + shadow_offset, top + shadow_offset), (right + shadow_offset, bottom + shadow_offset), (28, 28, 28), -1)
            cv2.rectangle(frame, (left, top), (right, bottom), block.color, -1)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255) if is_selected else (35, 35, 35), 3 if is_selected else 2)
            cv2.rectangle(frame, (left + 10, top + 10), (right - 10, top + 16), (255, 255, 255), -1)

    def draw_overlay(self, frame: object, cursor: Optional[tuple[int, int]], pinching: bool) -> None:
        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 92), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
        cv2.putText(frame, "GESTURE BLOCK CREATOR", (22, 34), cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Pinch empty space: create  |  Pinch block: select & move  |  C: clear  |  X: delete selected  |  Q: quit", (22, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (218, 218, 218), 1, cv2.LINE_AA)

        if cursor is not None:
            color = (92, 244, 151) if pinching else (255, 255, 255)
            cv2.circle(frame, cursor, 15 if pinching else 10, color, 2)
            if pinching:
                cv2.circle(frame, cursor, 5, color, -1)

        if time.monotonic() < self.message_until:
            (message_width, _), _ = cv2.getTextSize(self.message, cv2.FONT_HERSHEY_SIMPLEX, 0.57, 2)
            x = max(18, (width - message_width) // 2)
            cv2.rectangle(frame, (x - 12, height - 52), (x + message_width + 12, height - 18), (25, 25, 25), -1)
            cv2.putText(frame, self.message, (x, height - 29), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 2, cv2.LINE_AA)

    def run(self) -> None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not camera.isOpened():
            raise RuntimeError("Could not open the webcam. Check camera access and try again.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    self.set_message("Could not read from the webcam")
                    continue

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cursor: Optional[tuple[int, int]] = None
                pinching = False

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    cursor, pinching = self.update_gesture(hand_landmarks, width, height)
                    self.drawer.draw_landmarks(frame, hand_landmarks, self.hand_connections)
                elif self.was_pinching:
                    self.was_pinching = False
                    self.selected_index = None
                    self.set_message("Hand lost — released")

                self.draw_blocks(frame)
                self.draw_overlay(frame, cursor, pinching)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("c"), ord("C")):
                    self.blocks.clear()
                    self.selected_index = None
                    self.set_message("Canvas cleared")
                if key in (ord("x"), ord("X"), 8, 127) and self.selected_index is not None:
                    self.blocks.pop(self.selected_index)
                    self.selected_index = None
                    self.set_message("Selected block deleted")
        finally:
            camera.release()
            self.hands.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    GestureBlockCreator().run()
