import pygame
import random
import sys

# ─── Constants ────────────────────────────────────────────────────────────────
WINDOW_W = 600
WINDOW_H = 600
CELL_SIZE = 20
GRID_W = WINDOW_W // CELL_SIZE
GRID_H = WINDOW_H // CELL_SIZE
FPS = 6

# Colors
BLACK      = (0,   0,   0)
WHITE      = (255, 255, 255)
GREEN      = (0,   200, 80)
DARK_GREEN = (0,   140, 50)
RED        = (220, 50,  50)
GRAY       = (40,  40,  40)
YELLOW     = (255, 220, 0)
CYAN       = (0,   220, 220)


class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("🐍 Gesture Snake AI")
        self.clock = pygame.time.Clock()
        self.font_big   = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_med   = pygame.font.SysFont("Arial", 28)
        self.font_small = pygame.font.SysFont("Arial", 20)
        self.reset()

    def reset(self):
        """Reset game to initial state."""
        cx, cy = GRID_W // 2, GRID_H // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = (1, 0)   # Moving RIGHT initially
        self.pending_dir = (1, 0)
        self.food = self._spawn_food()
        self.score = 0
        self.game_over = False
        self.paused = False

    def _spawn_food(self):
        """Spawn food at a random cell not occupied by the snake."""
        while True:
            pos = (random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))
            if pos not in self.snake:
                return pos

    def set_direction(self, gesture_dir):
        """
        Update direction from gesture input.
        Ignores reverse direction to prevent self-collision.
        """
        if gesture_dir is None or self.game_over:
            return

        mapping = {
            'UP':    (0, -1),
            'DOWN':  (0,  1),
            'LEFT':  (-1, 0),
            'RIGHT': (1,  0),
        }

        new_dir = mapping.get(gesture_dir)
        if not new_dir:
            return

        # Prevent reversing into itself
        opposite = (-self.direction[0], -self.direction[1])
        if new_dir != opposite:
            self.pending_dir = new_dir

    def update(self):
        """Advance game state by one tick."""
        if self.game_over or self.paused:
            return

        self.direction = self.pending_dir
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Wall collision
        if not (0 <= new_head[0] < GRID_W and 0 <= new_head[1] < GRID_H):
            self.game_over = True
            return

        # Self collision
        if new_head in self.snake:
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        # Food eaten
        if new_head == self.food:
            self.score += 10
            self.food = self._spawn_food()
        else:
            self.snake.pop()

    def draw(self, gesture_label=None):
        """Render the full game screen."""
        self.screen.fill(BLACK)
        self._draw_grid()
        self._draw_food()
        self._draw_snake()
        self._draw_hud(gesture_label)

        if self.game_over:
            self._draw_overlay("GAME OVER", f"Score: {self.score}", "Press R to restart")

        if self.paused:
            self._draw_overlay("PAUSED", "", "Press P to resume")

        pygame.display.flip()

    def _draw_grid(self):
        for x in range(0, WINDOW_W, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, WINDOW_H))
        for y in range(0, WINDOW_H, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (WINDOW_W, y))

    def _draw_snake(self):
        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i > 0 else DARK_GREEN
            rect = pygame.Rect(x * CELL_SIZE + 1, y * CELL_SIZE + 1,
                               CELL_SIZE - 2, CELL_SIZE - 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            # Eyes on head
            if i == 0:
                ex = x * CELL_SIZE + CELL_SIZE // 2
                ey = y * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.circle(self.screen, WHITE, (ex - 4, ey - 4), 3)
                pygame.draw.circle(self.screen, WHITE, (ex + 4, ey - 4), 3)

    def _draw_food(self):
        fx, fy = self.food
        rect = pygame.Rect(fx * CELL_SIZE + 2, fy * CELL_SIZE + 2,
                           CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(self.screen, RED, rect, border_radius=6)

    def _draw_hud(self, gesture_label):
        score_surf = self.font_med.render(f"Score: {self.score}", True, YELLOW)
        self.screen.blit(score_surf, (10, 8))

        if gesture_label:
            dir_surf = self.font_small.render(f"Gesture: {gesture_label}", True, CYAN)
            self.screen.blit(dir_surf, (WINDOW_W - 180, 8))

        hint = self.font_small.render("P=Pause  R=Restart  Q=Quit", True, GRAY)
        self.screen.blit(hint, (10, WINDOW_H - 26))

    def _draw_overlay(self, title, subtitle, hint):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        t = self.font_big.render(title, True, WHITE)
        self.screen.blit(t, t.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 - 50)))

        if subtitle:
            s = self.font_med.render(subtitle, True, YELLOW)
            self.screen.blit(s, s.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 10)))

        h = self.font_small.render(hint, True, CYAN)
        self.screen.blit(h, h.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 55)))

    def handle_key(self, key):
        """Handle keyboard fallback controls."""
        if key == pygame.K_r:
            self.reset()
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_q:
            pygame.quit()
            sys.exit()
        # Arrow key fallbacks
        elif key == pygame.K_UP:
            self.set_direction('UP')
        elif key == pygame.K_DOWN:
            self.set_direction('DOWN')
        elif key == pygame.K_LEFT:
            self.set_direction('LEFT')
        elif key == pygame.K_RIGHT:
            self.set_direction('RIGHT')

    def tick(self):
        self.clock.tick(FPS)


# ─── Standalone test (keyboard only) ─────────────────────────────────────────
if __name__ == "__main__":
    game = SnakeGame()
    print("Snake Game running (keyboard mode) — press Q to quit")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                game.handle_key(event.key)

        game.update()
        game.draw(gesture_label="KEYBOARD")
        game.tick()