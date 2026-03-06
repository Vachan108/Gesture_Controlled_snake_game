import cv2
import pygame
import sys
import time

from gesture_engine import GestureEngine
from snake_game import SnakeGame

SHOW_CAMERA  = True
CAMERA_INDEX = 0
GRACE_PERIOD = 2.0  # seconds before snake starts moving (fixes instant death)


def main():
    print("=" * 52)
    print("  🐍 Gesture Snake — Phase 2")
    print("  👆 Point finger     → Move snake")
    print("  ✊ Make a fist      → Pause / Unpause")
    print("  ✋ Open palm        → Restart game")
    print("  Keyboard: R=Restart | P=Pause | Q=Quit")
    print("=" * 52)

    engine = GestureEngine()
    game   = SnakeGame()
    cap    = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        sys.exit(1)

    print("✅ Webcam opened. Get ready...")

    start_time        = time.time()
    current_direction = None

    while True:
        # ── Grace period countdown ────────────────────────────────────────────
        elapsed     = time.time() - start_time
        in_grace    = elapsed < GRACE_PERIOD
        countdown   = max(0, GRACE_PERIOD - elapsed)

        # ── Webcam frame ──────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # ── Gesture detection ─────────────────────────────────────────────────
        direction, special, annotated = engine.get_gesture(frame)

        # Overlay countdown on camera during grace period
        if in_grace:
            cv2.putText(annotated, f"Starting in {countdown:.1f}s",
                        (10, annotated.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

        if SHOW_CAMERA:
            cv2.imshow("Gesture Cam", annotated)
            cv2.waitKey(1)

        # ── Handle special gestures ───────────────────────────────────────────
        if special == 'PALM':
            game.reset()
            start_time = time.time()  # Reset grace period on restart
            current_direction = None
            print("✋ PALM detected → Game Restarted")

        elif special == 'FIST':
            game.paused = not game.paused
            print(f"✊ FIST detected → {'Paused' if game.paused else 'Resumed'}")

        # ── Apply direction (skip during grace period) ────────────────────────
        if direction and not in_grace:
            current_direction = direction

        if not in_grace:
            game.set_direction(current_direction)

        # ── Pygame events ─────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _shutdown(cap, engine)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    _shutdown(cap, engine)
                else:
                    game.handle_key(event.key)
                    if event.key == pygame.K_r:
                        start_time = time.time()

        # ── Game update + render ──────────────────────────────────────────────
        if not in_grace:
            game.update()

        game.draw(gesture_label=current_direction or ("READY..." if in_grace else "---"))
        game.tick()


def _shutdown(cap, engine):
    print("\n👋 Shutting down...")
    cap.release()
    engine.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()