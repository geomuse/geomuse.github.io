import pygame
import random

# 初始化 Pygame
pygame.init()

# 颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

# 游戏板尺寸
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 30

# 屏幕尺寸
SCREEN_WIDTH = BOARD_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = BOARD_HEIGHT * BLOCK_SIZE

# 创建屏幕
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("俄罗斯方块")

# 定义方块形状
SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]]
]

COLORS = [CYAN, YELLOW, MAGENTA, RED, GREEN, BLUE, ORANGE]

class Tetromino:
    def __init__(self):
        self.shape = random.choice(SHAPES)
        self.color = random.choice(COLORS)
        self.x = BOARD_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0

    def rotate(self):
        self.shape = list(zip(*self.shape[::-1]))

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

class TetrisGame:
    def __init__(self):
        self.board = [[BLACK for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.current_piece = Tetromino()
        self.game_over = False
        self.score = 0

    def draw(self):
        screen.fill(BLACK)
        for y, row in enumerate(self.board):
            for x, color in enumerate(row):
                pygame.draw.rect(screen, color, (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE-1, BLOCK_SIZE-1))
        
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, self.current_piece.color,
                                     ((self.current_piece.x + x) * BLOCK_SIZE,
                                      (self.current_piece.y + y) * BLOCK_SIZE,
                                      BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()

    def move_piece(self, dx, dy):
        self.current_piece.move(dx, dy)
        if self.check_collision():
            self.current_piece.move(-dx, -dy)
            if dy > 0:
                self.lock_piece()

    def check_collision(self):
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    if (self.current_piece.x + x < 0 or
                        self.current_piece.x + x >= BOARD_WIDTH or
                        self.current_piece.y + y >= BOARD_HEIGHT or
                        self.board[self.current_piece.y + y][self.current_piece.x + x] != BLACK):
                        return True
        return False

    def lock_piece(self):
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    self.board[self.current_piece.y + y][self.current_piece.x + x] = self.current_piece.color
        self.clear_lines()
        self.current_piece = Tetromino()
        if self.check_collision():
            self.game_over = True

    def clear_lines(self):
        lines_cleared = 0
        for y in range(BOARD_HEIGHT):
            if all(cell != BLACK for cell in self.board[y]):
                del self.board[y]
                self.board.insert(0, [BLACK for _ in range(BOARD_WIDTH)])
                lines_cleared += 1
        self.score += lines_cleared ** 2 * 100

    def rotate_piece(self):
        self.current_piece.rotate()
        if self.check_collision():
            self.current_piece.rotate()
            self.current_piece.rotate()
            self.current_piece.rotate()

def main():
    game = TetrisGame()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.5  # 每0.5秒下落一次
    
    while not game.game_over:
        fall_time += clock.get_rawtime()
        clock.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.move_piece(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    game.move_piece(1, 0)
                elif event.key == pygame.K_DOWN:
                    game.move_piece(0, 1)
                elif event.key == pygame.K_UP:
                    game.rotate_piece()

        if fall_time / 1000 > fall_speed:
            game.move_piece(0, 1)
            fall_time = 0

        game.draw()

    print(f"Game Over! Final Score: {game.score}")
    pygame.quit()

if __name__ == "__main__":
    
    main()