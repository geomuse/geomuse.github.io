import pygame
import time
import random
import numpy as np
# from dataclasses import dataclass # 移除 dataclass，使用標準類別以修復初始化錯誤
from collections import deque

class snake():
    
    # 核心修復：將類屬性移入 __init__，確保實例化時能夠正確訪問屬性
    def __init__(self):
        pygame.font.init()
        # 遊戲窗口大小和速度設置
        self.window_width = 600
        self.window_height = 600
        self.snake_block = 10 # 調整為 20 確保與格子計算一致 (原代碼中食物生成邏輯使用 / 20.0 * 20.0)
        self.snake_speed = 15 
        
        # 定義顏色
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.purple = (155, 89, 182)
        self.green = (46, 204, 113)
        self.blue = (52, 152, 219)
        self.red = (255, 0, 0) # 新增顏色

        self.game_window = None
        self.snake_list = []
        self.snake_length = 1
        self.snake_x = self.window_width / 2
        self.snake_y = self.window_height / 2
        self.snake_x_change = 0
        self.snake_y_change = 0
        self.food_x, self.food_y = self._random_food_pos() # 初始化食物位置
        self.clock = pygame.time.Clock()
        self.action = None
        self.last_action = None # 新增：記錄上一個動作，防止 180 度轉向

    # 輔助函數：用於初始化和重置食物位置
    def _random_food_pos(self):
        return (round(random.randrange(0, self.window_width - self.snake_block) / self.snake_block) * self.snake_block,
                round(random.randrange(0, self.window_height - self.snake_block) / self.snake_block) * self.snake_block)
    
    def _init_windows(self):
        self.game_window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('snake.')

    def design_snake(self, snake_block, snake_list):
        for x in snake_list:
            pygame.draw.rect(self.game_window, self.green, [x[0], x[1], snake_block, snake_block])

    def design_food(self):
        pygame.draw.rect(self.game_window, self.purple, [self.food_x, self.food_y, self.snake_block, self.snake_block])

    def display_score(self, score):
        font = pygame.font.SysFont(None, 15)
        text = font.render(str(score), True, self.blue)
        self.game_window.blit(text, [10, 10])

    def score_table_show(self):
        font_style = pygame.font.SysFont(None, 15)
        message = font_style.render(str(self.snake_length - 1), True, self.black)
        self.game_window.blit(message, [self.window_width / 2, self.window_height / 2])
        pygame.display.update()
        time.sleep(2)

    def reset(self):
        self._init_windows()
        self.snake_list = []
        self.snake_length = 1
        self.reward = 0 
        self.snake_x = self.window_width / 2
        self.snake_y = self.window_height / 2
        self.snake_x_change = 0
        self.snake_y_change = 0
        self.food_x, self.food_y = self._random_food_pos() # 使用修正後的食物初始化
        self.last_action = None # 重置 last_action

        observation = self.get_observation()
        return observation

    def step(self,action):
        # 修正：防止 180 度轉向，並更新 last_action
        if action is not None:
            if self.last_action == 0 and action == 1: action = self.last_action
            elif self.last_action == 1 and action == 0: action = self.last_action
            elif self.last_action == 2 and action == 3: action = self.last_action
            elif self.last_action == 3 and action == 2: action = self.last_action
            self.last_action = action
            
            if action == 0:
                self.snake_y_change = -self.snake_block
                self.snake_x_change = 0
            elif action == 1:
                self.snake_y_change = self.snake_block
                self.snake_x_change = 0
            elif action == 2:
                self.snake_x_change = -self.snake_block
                self.snake_y_change = 0
            elif action == 3:
                self.snake_x_change = self.snake_block
                self.snake_y_change = 0

        self.snake_x += self.snake_x_change
        self.snake_y += self.snake_y_change

        done = False
        # reward += 0

        if self.snake_x >= self.window_width or self.snake_x < 0 or self.snake_y >= self.window_height or self.snake_y < 0 :
            self.score_table_show()
            done = True
            # reward = -10

        if [self.snake_x, self.snake_y] in self.snake_list[:-1]:
            self.score_table_show()
            done = True
            # reward = -1

        self.game_window.fill(self.white)
        self.design_food()
        snake_head = []
        snake_head.append(self.snake_x)
        snake_head.append(self.snake_y)
        self.snake_list.append(snake_head)
        if len(self.snake_list) > self.snake_length:
            del self.snake_list[0]

        self.design_snake(self.snake_block, self.snake_list)
        
        self.display_score(self.reward)
        pygame.display.update()

        if self.snake_x == self.food_x and self.snake_y == self.food_y:
            self.food_x, self.food_y = self._random_food_pos() # 使用修正後的食物初始化
            self.snake_length = 2 
            self.reward += 1

        observation = self.get_observation()
        return observation, self.reward, done, {}

    def get_observation(self):
        observation = pygame.surfarray.array3d(pygame.display.get_surface())
        observation = np.transpose(observation, axes=(1, 0, 2))
        return observation

    def render(self, mode=None):
        self.game_window.fill(self.white)
        pygame.draw.rect(self.game_window, self.purple, [self.food_x, self.food_y, self.snake_block, self.snake_block])
        self.design_snake(self.snake_block, self.snake_list)
        self.display_score(self.reward - 1)
        pygame.display.update()
        self.clock.tick(self.snake_speed)
        
        # 關鍵修正：確保窗口可移動和關閉
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close() 
            
            if mode in ('human','h'):
                if event.type == pygame.KEYDOWN :
                    if event.key == pygame.K_UP:
                        return 0 
                    elif event.key == pygame.K_DOWN:
                        return 1
                    elif event.key == pygame.K_LEFT:
                        return 2
                    elif event.key == pygame.K_RIGHT:
                        return 3

    def close(self):
        pygame.quit()
        quit()
    
    def run(self):
        # 注意：這裡的 env 應該是 self
        action = self.render(mode='h')
        if action != None :
            print(action)
        return self.step(action)
    
    # --- 高生存率 AI 輔助函數 (新增) ---
    def is_valid(self, x, y, snake_body_set):
        """檢查座標是否在邊界內且不在蛇身體上 (不包括蛇尾)"""
        if (0 <= x < self.window_width and 
            0 <= y < self.window_height and 
            (x, y) not in snake_body_set):
            return True
        return False

    def is_safe(self, start_pos, target_pos, snake_body_set):
        """用 BFS 檢查從 start_pos 是否能安全到達 target_pos (逃生路線檢查)"""
        queue = deque([(start_pos, [start_pos])])
        visited = {start_pos}
        
        while queue:
            (x, y), path = queue.popleft()
            
            if (x, y) == target_pos:
                return True
            
            for dx, dy in [(0, -self.snake_block), (0, self.snake_block), 
                           (-self.snake_block, 0), (self.snake_block, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                
                if self.is_valid(nx, ny, snake_body_set) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
                    
        return False

    def bfs_agent(self,env):
        """用 BFS 找到食物並檢查逃生路線，最大化生存時間"""
        head = (int(env.snake_x), int(env.snake_y))
        food = (int(env.food_x), int(env.food_y))
        
        directions = {
            0: (0, -env.snake_block), 1: (0, env.snake_block), 
            2: (-env.snake_block, 0), 3: (env.snake_block, 0)
        }
        
        # 獲取當前蛇身 (排除當前蛇頭)
        current_snake_body = set(map(tuple, env.snake_list[:-1]))

        # **修復 UnboundLocalError**：定義蛇尾
        if env.snake_list:
            snake_tail = tuple(env.snake_list[0])
        else:
            snake_tail = head 

        # 1. 尋找食物路徑 (BFS)
        queue = deque([(head, [])])
        visited = {head}
        best_path = None
        
        while queue:
            (x, y), path = queue.popleft()

            if (x, y) == food:
                best_path = path 
                break

            for action, (dx, dy) in directions.items():
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                # 檢查 180 度轉向 (如果這是第一個動作)
                if env.last_action is not None and len(path) == 0:
                    if (env.last_action == 0 and action == 1) or \
                       (env.last_action == 1 and action == 0) or \
                       (env.last_action == 2 and action == 3) or \
                       (env.last_action == 3 and action == 2):
                       continue

                if self.is_valid(nx, ny, current_snake_body) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [action]))

        # 2. 決策：有找到食物路徑且安全嗎？
        if best_path:
            next_action = best_path[0]
            dx, dy = directions[next_action]
            next_head = (head[0] + dx, head[1] + dy)

            # 模擬移動後新的蛇身 (蛇頭移動，蛇尾消失)
            simulated_body = current_snake_body.copy()
            simulated_body.add(next_head)
            simulated_body.discard(snake_tail) 

            # **關鍵生存檢查**：移動後是否能到達尾巴 (避免陷入死鎖)
            if self.is_safe(next_head, snake_tail, simulated_body):
                return next_action 

        # 3. 備用策略：如果找不到食物或路徑不安全，則執行保命策略 (追尾)
        safe_actions = []
        for action, (dx, dy) in directions.items():
            next_head = (head[0] + dx, head[1] + dy)
            
            simulated_body = current_snake_body.copy()
            simulated_body.add(next_head)
            simulated_body.discard(snake_tail) 

            # 檢查移動後是否能到達尾巴 (確保不會死鎖)
            if self.is_valid(next_head[0], next_head[1], current_snake_body): 
                if self.is_safe(next_head, snake_tail, simulated_body):
                    safe_actions.append(action)

        if safe_actions:
            # 選擇一個安全的動作
            return random.choice(safe_actions)
        
        # 4. 最後手段：選擇一個不會立即死亡的動作
        for action, (dx, dy) in directions.items():
            next_head = (head[0] + dx, head[1] + dy)
            if self.is_valid(next_head[0], next_head[1], current_snake_body):
                return action
                
        # 無路可走
        return random.choice([0, 1, 2, 3]) 

if __name__ == "__main__":
    try:
        pygame.init() 
    except:
        pass

    env = snake()
    observation = env.reset()
    done = False

    while not done:
        env.render()
        
        action = env.bfs_agent(env)
        
        if action is not None:
             observation, reward, done, _ = env.step(action)
        
    env.close()