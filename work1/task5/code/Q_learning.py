import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GridWorld:
    def __init__(self):
        self.grid = [
            ['S', 0 , 0 , 0 , 0 ],
            [ 0 ,'T', 0 ,'T', 0 ],
            [ 0 , 0 , 0 ,'T', 0 ],
            [ 0 ,'T', 0 , 0 , 0 ],
            [ 0 , 0 , 0 , 0 ,'G']
        ]
        self.rows = 5
        self.cols = 5
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.trap_positions = [(1, 1), (1, 3), (2, 3), (3, 1)]
        # self.trap_positions = [(1, 0), (1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3), (3, 4)]

        # 动作映射：0-上, 1-下, 2-左, 3-右
        self.actions = [0, 1, 2, 3]
        self.action_names = ['↑', '↓', '←', '→']

    def reset(self):
        """重置环境到起始状态"""
        return self.start_pos

    def step(self, state, action):
        """执行动作，返回新状态、奖励和是否结束"""
        row, col = state

        new_row, new_col = 0, 0

        # 根据动作计算新位置
        if action == 0:  # ↑
            new_row, new_col = max(row - 1, 0), col
        elif action == 1:  # ↓
            new_row, new_col = min(row + 1, self.rows - 1), col
        elif action == 2:  # ←
            new_row, new_col = row, max(col - 1, 0)
        elif action == 3:  # →
            new_row, new_col = row, min(col + 1, self.cols - 1)

        # 边界检查
        if (new_row, new_col) == (row, col):                # 试图走出边界，保持原地
            reward = -1
            done = False
            return (row, col), reward, done

        # 检查新位置类型
        if (new_row, new_col) == self.goal_pos:             # 到达终点
            reward = 10
            done = True
        elif (new_row, new_col) in self.trap_positions:     # 掉入陷阱
            reward = -10
            done = True
        else:                                               # 普通空地
            reward = -1
            done = False

        return (new_row, new_col), reward, done

    def render(self, state=None):
        """可视化网格世界"""
        grid_copy = [row.copy() for row in self.grid]

        if state:
            row, col = state
            if grid_copy[row][col] == 0:
                grid_copy[row][col] = 'A'  # A表示智能体位置

        # 当前网格状态
        for row in grid_copy:
            print(' '.join(['|'] + [str(x) for x in row] + ['|']))
        print()

class QLearningAgent:
    def __init__(self, grid_world_env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.env = grid_world_env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # 初始化Q表：状态空间大小 x 动作空间大小
        self.q_table = np.zeros((grid_world_env.rows, grid_world_env.cols, len(grid_world_env.actions)))

    def choose_action(self, state):
        """使用ε-贪婪策略选择动作"""
        row, col = state

        if random.uniform(0, 1) < self.exploration_rate:
            # 探索：随机选择动作
            return random.choice(self.env.actions)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.q_table[row, col])

    def update_q_value(self, state, action, reward, next_state, done):
        """更新Q值"""
        row, col = state
        next_row, next_col = next_state

        current_q = self.q_table[row, col, action]

        if done:
            # 如果是终止状态，没有下一个状态的Q值
            target = reward
        else:
            # 使用下一个状态的最大Q值
            max_next_q = np.max(self.q_table[next_row, next_col])
            target = reward + self.discount_factor * max_next_q

        # Q值更新公式
        self.q_table[row, col, action] = current_q + self.learning_rate * (target - current_q)

    def train(self, episodes=1000, max_steps=100, verbose=False):
        """训练智能体"""
        rewards_per_episode = []
        steps_per_episode = []

        for episode in range(episodes):
            state = self.env.reset()
            train_total_reward = 0
            train_steps = 0

            for train_steps in range(max_steps):
                # 选择动作
                action = self.choose_action(state)

                # 执行动作
                next_state, reward, done = self.env.step(state, action)

                # 更新Q值
                self.update_q_value(state, action, reward, next_state, done)

                train_total_reward += reward
                state = next_state

                if done:
                    train_steps += 1
                    break

            rewards_per_episode.append(train_total_reward)
            steps_per_episode.append(train_steps)

            # 逐渐减少探索率
            self.exploration_rate = max(0.01, self.exploration_rate * 0.995)

            if verbose and (episode + 1) % 100 == 0:
                print(f"回合 {episode + 1:<4d}: 总奖励 = {train_total_reward:<4d}, 步数 = {train_steps:<4d}")

        return rewards_per_episode, steps_per_episode

    def test(self, max_steps=50):
        """测试训练好的智能体"""
        state = self.env.reset()
        test_total_reward = 0
        test_path = [state]

        self.env.render(state)

        for step in range(max_steps):
            # 测试时使用贪婪策略（不探索）
            row, col = state
            action = np.argmax(self.q_table[row, col])

            next_state, reward, done = self.env.step(state, action)

            test_total_reward += reward
            state = next_state
            test_path.append(state)

            print(f"步骤 {step + 1}: {self.env.action_names[action]}")
            self.env.render(state)

            if done:
                if reward > 0:
                    print(f"成功到达终点! 总步数: {step + 1}, 总奖励: {test_total_reward}")
                else:
                    print(f"掉入陷阱! 总步数: {step + 1}, 总奖励: {test_total_reward}")
                break
        else:
            print(f"未在最大步数内完成，总奖励: {test_total_reward}")

        return test_path, test_total_reward

    def get_optimal_policy(self):
        """获取最优策略"""
        policy = np.zeros((self.env.rows, self.env.cols), dtype=int)

        for i in range(self.env.rows):      # pylint: disable=redefined-outer-name
            for j in range(self.env.cols):  # pylint: disable=redefined-outer-name
                if (i, j) == self.env.goal_pos or (i, j) in self.env.trap_positions:
                    policy[i, j] = -1  # 终止状态
                else:
                    policy[i, j] = np.argmax(self.q_table[i, j])

        return policy

    def print_policy(self):
        """打印策略"""
        policy = self.get_optimal_policy()

        print("- 最优策略 -")
        for i in range(self.env.rows):          # pylint: disable=redefined-outer-name
            row_str = []
            for j in range(self.env.cols):      # pylint: disable=redefined-outer-name
                if (i, j) == self.env.start_pos:
                    row_str.append('S')
                elif (i, j) == self.env.goal_pos:
                    row_str.append('G')
                elif (i, j) in self.env.trap_positions:
                    row_str.append('T')
                else:
                    action = policy[i, j]
                    if action == -1:
                        row_str.append('X')
                    else:
                        row_str.append(self.env.action_names[action])
            print(' '.join(['|'] + row_str + ['|']))

def plot_training_progress(rewards, steps):     # pylint: disable=redefined-outer-name
    """绘制训练进度图"""
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    window_size = 50
    smoothed_rewards = [np.mean(rewards[i:i+window_size])
                       for i in range(len(rewards) - window_size)]

    ax1.plot(rewards, alpha=0.3, label='原始奖励')
    ax1.plot(range(window_size, len(rewards)), smoothed_rewards,
             label=f'{window_size}步移动平均', color='red')
    ax1.set_xlabel('回合')
    ax1.set_ylabel('总奖励')
    ax1.set_title('训练过程中的奖励变化')
    ax1.legend()
    ax1.grid(True)

    smoothed_steps = [np.mean(steps[i:i+window_size])
                     for i in range(len(steps) - window_size)]

    ax2.plot(steps, alpha=0.3, label='原始步数')
    ax2.plot(range(window_size, len(steps)), smoothed_steps,
             label=f'{window_size}步移动平均', color='red')
    ax2.set_xlabel('回合')
    ax2.set_ylabel('步数')
    ax2.set_title('训练过程中的步数变化')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 创建环境 & 智能体
    env = GridWorld()
    agent = QLearningAgent(env,
                          learning_rate=0.1,
                          discount_factor=0.9,
                          exploration_rate=0.1)

    print("---------------训练开始---------------")
    print("- 环境布局 -")
    env.render()

    # 训练
    rewards, steps = agent.train(episodes=1000, max_steps=100, verbose=True)

    # 绘制训练进度
    plot_training_progress(rewards, steps)

    # 测试
    print("-----------------测试-----------------")
    path, total_reward = agent.test(max_steps=50)

    # 显示最优策略
    print("\n" + "="*50)
    agent.print_policy()

    # 显示Q表
    print("\n--------------------Q表--------------------")
    print("状态\\动作\t  ↑\t  ↓\t  ←\t  →")
    for i in range(env.rows):
        for j in range(env.cols):
            state_str = f"({i},{j})"
            q_values = [f"{agent.q_table[i, j, a]:.2f}" for a in env.actions]
            print(f"{state_str}\t\t" + "\t".join(q_values))
