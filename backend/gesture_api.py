import cv2
import mediapipe as mp
import numpy as np
import math
import time
import base64

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

FINGER_TIPS = [8, 12, 16, 20]
FINGER_MCPS = [5,  9, 13, 17]



def _raw_dir(lm, threshold=0.10):
    """Quick direction label for overlay — no cooldown/smoothing."""
    dx = lm[8].x - lm[0].x
    dy = lm[8].y - lm[0].y
    if abs(dx) > abs(dy):
        if dx >  threshold: return 'RIGHT'
        if dx < -threshold: return 'LEFT'
    else:
        if dy < -threshold: return 'UP'
        if dy >  threshold: return 'DOWN'
    return None

class GestureEngine:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        self.dir_buffer        = []
        self.buffer_size       = 7
        self.threshold         = 0.10
        self.last_dir_time     = 0
        self.dir_cooldown      = 0.25
        self.last_special_time = 0
        self.special_cooldown  = 1.0

    def process_frame(self, frame_bytes: bytes) -> dict:
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return self._empty_result("invalid_frame")

        frame = cv2.flip(frame, 1)
        annotated = frame.copy()

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        direction      = None
        special        = None
        right_detected = False
        left_detected  = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm    = hand_landmarks.landmark
                wrist = lm[0]
                is_right_hand = wrist.x > 0.5  # after cv2.flip, right hand appears on RIGHT side (x>0.5)

                # ── Draw landmarks for RIGHT hand only (green) ───────────
                if is_right_hand:
                    mp_draw.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 80),   thickness=2, circle_radius=5),
                        mp_draw.DrawingSpec(color=(180, 255, 180), thickness=2)
                    )
                    h, w = frame.shape[:2]
                    p1 = (int(lm[0].x * w), int(lm[0].y * h))
                    p2 = (int(lm[8].x * w), int(lm[8].y * h))
                    cv2.arrowedLine(annotated, p1, p2, (0, 255, 80), 3, tipLength=0.35)
                    raw_dir = _raw_dir(lm, self.threshold)
                    if raw_dir:
                        cv2.putText(annotated, raw_dir,
                                    (p2[0] + 8, p2[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
                # ── Left hand: no drawing ─────────────────────────────────────

                # ── Gesture logic ────────────────────────────────────────────
                if is_right_hand:
                    right_detected = True
                    direction = self._get_direction(lm)
                else:
                    left_detected = True
                    now = time.time()
                    if now - self.last_special_time > self.special_cooldown:
                        gesture = self._classify_hand(lm)
                        if gesture in ('FIST', 'PALM'):
                            special = gesture
                            self.last_special_time = now

        # ── Subtle border when hand detected ────────────────────────────────
        if right_detected:
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0,0), (w-1, h-1), (0, 220, 80), 2)

        # ── Encode annotated frame → base64 JPEG ─────────────────────────────
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buf).decode('utf-8')

        return {
            "direction":   self._smooth_direction(direction),
            "special":     special,
            "right_hand":  right_detected,
            "left_hand":   left_detected,
            "frame":       frame_b64       # ← annotated frame sent back
        }

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

    def _classify_hand(self, lm):
        extended = 0
        for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS):
            wrist_to_tip = self._dist(lm[0], lm[tip])
            wrist_to_mcp = self._dist(lm[0], lm[mcp])
            if wrist_to_mcp < 0.001:
                continue
            if wrist_to_tip > wrist_to_mcp * 1.15:
                extended += 1
        if extended == 0:   return 'FIST'
        elif extended >= 3: return 'PALM'
        return None

    def _dist(self, a, b):
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

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

    def _empty_result(self, reason=""):
        return {
            "direction":  None,
            "special":    None,
            "right_hand": False,
            "left_hand":  False,
            "frame":      None,
            "error":      reason
        }

    def release(self):
        self.hands.close()