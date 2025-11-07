import pygame, sys, random, heapq
from collections import deque

# 配置
CELL = 20
COLS = 30
ROWS = 20
WIDTH = CELL * COLS
HEIGHT = CELL * ROWS
FPS = 12

# 方向向量
DIRS = {'UP': (0,-1), 'DOWN': (0,1), 'LEFT': (-1,0), 'RIGHT': (1,0)}
OPPOSITE = {'UP':'DOWN','DOWN':'UP','LEFT':'RIGHT','RIGHT':'LEFT'}

def in_bounds(pos):
    x,y = pos
    return 0 <= x < COLS and 0 <= y < ROWS

def neighbors(pos):
    x,y = pos
    for dx,dy in DIRS.values():
        nx,ny = x+dx, y+dy
        if in_bounds((nx,ny)):
            yield (nx,ny)

# A* shortest path avoiding obstacles
def astar(start, goal, obstacles):
    def h(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_heap = []
    heapq.heappush(open_heap, (h(start,goal), 0, start, None))
    came = {}
    gscore = {start:0}
    while open_heap:
        f,g,curr,prev = heapq.heappop(open_heap)
        if curr in came: continue
        came[curr] = prev
        if curr == goal:
            # reconstruct
            path = []
            node = curr
            while node != start:
                path.append(node)
                node = came[node]
            path.reverse()
            return path
        for n in neighbors(curr):
            if n in obstacles and n != goal: 
                continue
            tentative = g + 1
            if n not in gscore or tentative < gscore[n]:
                gscore[n] = tentative
                heapq.heappush(open_heap, (tentative + h(n,goal), tentative, n, curr))
    return None

# flood fill count reachable cells from pos avoiding obstacles
def flood_count(start, obstacles):
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for n in neighbors(cur):
            if n in seen or n in obstacles:
                continue
            seen.add(n)
            q.append(n)
    return len(seen)

# choose next move for auto snake
def get_auto_move(snake, food):
    head = snake[0]
    body_set = set(snake)  # obstacles
    # try find path to food
    path = astar(head, food, body_set)
    if path:
        # safety check: simulate path and ensure after reaching food there's space (flood fill)
        # simulate snake after following path (assume it will grow by 1 at food)
        sim_snake = list(snake)
        for step in path:
            sim_snake.insert(0, step)
            if step == food:
                # grows: do not pop tail this step
                pass
            else:
                sim_snake.pop()
        # after simulation, check reachability from head to tail (or check flood)
        obstacles_after = set(sim_snake)
        # compute flood from new head; if reachable cells count >= len(sim_snake) it's safe-ish
        reachable = flood_count(sim_snake[0], obstacles_after - {sim_snake[-1]})
        if reachable >= len(sim_snake):
            next_pos = path[0]
            dx = next_pos[0] - head[0]
            dy = next_pos[1] - head[1]
            for k,v in DIRS.items():
                if v == (dx,dy): return k
    # fallback: try move towards tail (follow-tail strategy)
    tail = snake[-1]
    obstacles_no_tail = set(snake[:-1])  # tail will move, so it's free next
    path2 = astar(head, tail, obstacles_no_tail)
    if path2:
        next_pos = path2[0]
        dx = next_pos[0] - head[0]
        dy = next_pos[1] - head[1]
        for k,v in DIRS.items():
            if v == (dx,dy): return k
    # last fallback: any safe direction
    for k,(dx,dy) in DIRS.items():
        nx,ny = head[0]+dx, head[1]+dy
        if in_bounds((nx,ny)) and (nx,ny) not in set(snake[:-1]):
            return k
    return 'UP'  # no choice

# Pygame 主循环与渲染
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # 初始蛇与食物
    snake = [(COLS//2 + i, ROWS//2) for i in range(3)][::-1]  # head at front
    direction = 'LEFT'
    food = None

    def place_food():
        while True:
            p = (random.randrange(COLS), random.randrange(ROWS))
            if p not in snake:
                return p

    food = place_food()
    score = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        # 自动决定方向
        direction = get_auto_move(snake, food)

        dx,dy = DIRS[direction]
        new_head = (snake[0][0] + dx, snake[0][1] + dy)
        # 撞墙或撞自己 => 重新开始
        if not in_bounds(new_head) or new_head in snake[:-1]:
            # game over -> reset
            snake = [(COLS//2 + i, ROWS//2) for i in range(3)][::-1]
            direction = 'LEFT'
            food = place_food()
            score = 0
            continue

        snake.insert(0, new_head)
        if new_head == food:
            score += 1
            food = place_food()
        else:
            snake.pop()

        # 渲染
        screen.fill((10,10,10))
        # grid optional
        for x in range(COLS):
            for y in range(ROWS):
                pygame.draw.rect(screen, (18,18,18), (x*CELL, y*CELL, CELL, CELL), 1)
        # 食物
        pygame.draw.rect(screen, (200,30,30), (food[0]*CELL, food[1]*CELL, CELL, CELL))
        # 蛇
        for i,seg in enumerate(snake):
            color = (50,200,50) if i==0 else (30,150,30)
            pygame.draw.rect(screen, color, (seg[0]*CELL+1, seg[1]*CELL+1, CELL-2, CELL-2))
        # 分数
        txt = font.render(f"Score: {score}", True, (200,200,200))
        screen.blit(txt, (8,8))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == '__main__':
    
    main()
