import cv2
import mediapipe as mp
import time
import math

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

FINGER_TIPS = [8, 12, 16, 20]
FINGER_MCPS = [5,  9, 13, 17]


class GestureEngine:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,                  # ← detect both hands
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        self.dir_buffer        = []
        self.buffer_size       = 7
        self.threshold         = 0.10
        self.last_dir_time     = 0
        self.dir_cooldown      = 0.25         # seconds between direction changes

        self.last_special_time = 0
        self.special_cooldown  = 1.0          # seconds between pause/restart

        self.right_hand_detected = False
        self.left_hand_detected  = False

    # ─────────────────────────────────────────────────────────────────────────
    def get_gesture(self, frame):
        """
        Returns (direction, special, annotated_frame)

        Two-hand logic:
          Right hand (x > 0.5 on mirrored frame) → direction control
          Left  hand (x < 0.5 on mirrored frame) → PALM=restart, FIST=pause
        """
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        direction  = None
        special    = None
        annotated  = frame.copy()

        self.right_hand_detected = False
        self.left_hand_detected  = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm    = hand_landmarks.landmark
                wrist = lm[0]

                # ── Classify hand side by wrist x position ────────────────────
                # Frame is mirrored → wrist.x > 0.5 = right side of screen = right hand
                is_right_hand = wrist.x > 0.5

                # Draw landmarks — blue for right (direction), orange for left (control)
                dot_color  = (255, 80,  0) if is_right_hand else (0, 180, 255)
                line_color = (200, 200, 200)
                mp_draw.draw_landmarks(
                    annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=dot_color,  thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=line_color, thickness=1)
                )

                if is_right_hand:
                    # ── RIGHT HAND → direction ────────────────────────────────
                    self.right_hand_detected = True
                    direction = self._get_direction(lm)

                    # Draw direction arrow
                    h, w = frame.shape[:2]
                    p1 = (int(lm[0].x * w), int(lm[0].y * h))
                    p2 = (int(lm[8].x * w), int(lm[8].y * h))
                    cv2.arrowedLine(annotated, p1, p2,
                                    (0, 255, 80) if direction else (80, 80, 80),
                                    3, tipLength=0.3)

                    # Label above right hand
                    label_x = int(lm[0].x * frame.shape[1])
                    label_y = int(lm[0].y * frame.shape[0]) - 20
                    cv2.putText(annotated, "MOVE HAND",
                                (label_x - 50, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

                else:
                    # ── LEFT HAND → special gestures ──────────────────────────
                    self.left_hand_detected = True
                    now = time.time()

                    if now - self.last_special_time > self.special_cooldown:
                        gesture = self._classify_hand(lm)
                        if gesture == 'FIST':
                            special = 'FIST'
                            self.last_special_time = now
                        elif gesture == 'PALM':
                            special = 'PALM'
                            self.last_special_time = now

                    # Label above left hand
                    label_x = int(lm[0].x * frame.shape[1])
                    label_y = int(lm[0].y * frame.shape[0]) - 20
                    cv2.putText(annotated, "CTRL HAND",
                                (label_x - 50, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

        # ── HUD overlay ───────────────────────────────────────────────────────
        self._draw_hud(annotated, direction, special)

        # ── Border ────────────────────────────────────────────────────────────
        # Green = both hands, Yellow = one hand, Red = no hands
        if self.right_hand_detected and self.left_hand_detected:
            bc = (0, 220, 0)
        elif self.right_hand_detected or self.left_hand_detected:
            bc = (0, 220, 220)
        else:
            bc = (0, 0, 220)
        cv2.rectangle(annotated, (0, 0),
                      (annotated.shape[1]-1, annotated.shape[0]-1), bc, 4)

        # ── Divider line in middle of frame ───────────────────────────────────
        mid_x = annotated.shape[1] // 2
        cv2.line(annotated, (mid_x, 0), (mid_x, annotated.shape[0]),
                 (60, 60, 60), 1)

        # ── Side labels ───────────────────────────────────────────────────────
        h_frame = annotated.shape[0]
        cv2.putText(annotated, "CTRL (Pause/Restart)",
                    (8, h_frame - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 180, 255), 1)
        cv2.putText(annotated, "MOVE (Direction)",
                    (mid_x + 8, h_frame - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 255, 80), 1)

        return self._smooth_direction(direction), special, annotated

    # ─────────────────────────────────────────────────────────────────────────
    def _classify_hand(self, lm):
        """
        Finger-counting method — works regardless of hand orientation.
        A finger is extended if its tip is farther from the wrist
        than its MCP knuckle is (by a safe margin).
          0 fingers extended → FIST
          3+ fingers extended → PALM
        """
        extended = 0
        for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS):
            wrist_to_tip = self._dist(lm[0], lm[tip])
            wrist_to_mcp = self._dist(lm[0], lm[mcp])
            if wrist_to_mcp < 0.001:
                continue
            # Tip is extended if it is significantly farther from wrist than knuckle
            if wrist_to_tip > wrist_to_mcp * 1.15:
                extended += 1

        print(f"[DEBUG] Extended fingers: {extended}  (FIST=0 | PALM>=3)")

        if extended == 0:
            return 'FIST'
        elif extended >= 3:
            return 'PALM'
        return None

    # ─────────────────────────────────────────────────────────────────────────
    def _dist(self, a, b):
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

    # ─────────────────────────────────────────────────────────────────────────
    def _get_direction(self, lm):
        dx = lm[8].x - lm[0].x
        dy = lm[8].y - lm[0].y

        direction = None
        if abs(dx) > abs(dy):
            if dx >  self.threshold: direction = 'RIGHT'
            elif dx < -self.threshold: direction = 'LEFT'
        else:
            if dy < -self.threshold: direction = 'UP'
            elif dy >  self.threshold: direction = 'DOWN'

        if direction:
            now = time.time()
            if now - self.last_dir_time >= self.dir_cooldown:
                self.last_dir_time = now
                return direction
        return None

    # ─────────────────────────────────────────────────────────────────────────
    def _smooth_direction(self, direction):
        if direction:
            self.dir_buffer.append(direction)
        if len(self.dir_buffer) > self.buffer_size:
            self.dir_buffer.pop(0)
        if not self.dir_buffer:
            return None
        weighted = {}
        for i, d in enumerate(self.dir_buffer):
            weighted[d] = weighted.get(d, 0) + (i + 1)
        return max(weighted, key=weighted.get)

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_hud(self, frame, direction, special):
        h = frame.shape[0]

        # Direction status (top left)
        if direction:
            d_label = f"Move: {direction}"
            d_color = (0, 255, 80)
        elif self.right_hand_detected:
            d_label = "Move: (waiting)"
            d_color = (160, 160, 160)
        else:
            d_label = "Move hand: not detected"
            d_color = (0, 80, 220)
        cv2.putText(frame, d_label, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, d_color, 2)

        # Special gesture status (top right area)
        if special:
            s_label = f"Ctrl: {special}"
            s_color = (0, 255, 255) if special == 'PALM' else (255, 140, 0)
        elif self.left_hand_detected:
            s_label = "Ctrl: (show fist/palm)"
            s_color = (160, 160, 160)
        else:
            s_label = "Ctrl hand: not detected"
            s_color = (0, 80, 220)

        mid_x = frame.shape[1] // 2
        cv2.putText(frame, s_label, (mid_x + 5, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)

    # ─────────────────────────────────────────────────────────────────────────
    def release(self):
        self.hands.close()


# ─── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = GestureEngine()
    cap    = cv2.VideoCapture(0)
    print("Two-Hand Gesture Engine — Q to quit")
    print("  RIGHT side of camera → point to move")
    print("  LEFT  side of camera → ✋ palm=restart, ✊ fist=pause")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        direction, special, annotated = engine.get_gesture(frame)
        if special:     print(f"Special:   {special}")
        elif direction: print(f"Direction: {direction}")
        cv2.imshow("Two-Hand Gesture Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    engine.release()
    cv2.destroyAllWindows()
