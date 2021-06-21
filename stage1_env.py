'''
Jiyao Chen
June 15 2021
Environment for stage 1
'''

import random
import numpy as np
import operator

K = 9                       # number of drones
NUMBER_OF_TASKS = 9         # number of tasks
GRID_SIZE = 300             # size of the grid/environment
ALPHA = 0.5                 # constant coefficient for execution in reward function
C_H = 2                     # energy consumption rate for hovering
C_T = 3                     # energy consumption rate for task execution
C_F = 2.5                   # energy consumption rate for forwarding
EFFICIENCY_THRESHOLD = 0.3  # energy efficiency threshold for calculating reward
B = 10000                   # battery capacity for drone
T = 3000                    # energy required for one single task
Q = -5                      # drone not coming back penalty

# actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
UPPER_LEFT = 4
UPPER_RIGHT = 5
BOTTOM_LEFT = 6
BOTTOM_RIGHT = 7
FREEZE = 8
EXECUTE = 9
STATIONARY = 10

# base station
BASE_X = 300
BASE_Y = 300
class Environment:
    def __init__(self, args, current_path):
        self.num_agents = K
        self.num_tasks = NUMBER_OF_TASKS
        self.grid_size = GRID_SIZE
        self.state_size = self.num_agents * 3 + self.num_tasks * 3 # not used
        self.agents_positions = []
        self.tasks_positions = []
        self.cell = []

        self.y_ik = np.zeros((self.num_tasks, self.num_agents))
        self.B_k = np.array((B for i in range(self.num_agents)))
        self.T_i = np.array((T for in range(self.num_tasks)))
        # self.tasks_positions_idx = np.random.choice(len(self.cells) - 1, size=self.num_tasks,
        #                                     replace=False)
        self.tasks_positions_idx = []

        self.action_space = [UP, DOWN, LEFT, RIGHT, UPPER_LEFT, UPPER_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, FREEZE, EXECUTE, STATIONARY]
        self.action_diff = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0), (0, 0), (0, 0)]


        self.num_episodes = 0
        self.terminal = False

    def set_positions_idx(self):

        cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)]
        tasks_positions_idx = [15050, 15150, 15250, 45050, 45150, 45250, 75050, 75150, 75250]
        return [cells, tasks_positions_idx]

    def reset(self):  # initialize the world

        self.terminal = False
        [self.cells, self.tasks_positions_idx] = self.set_positions_idx()
        self.y_ik = np.zeros((self.num_tasks, self.num_agents))
        self.B_k = np.array((B for i in range (self.num_agents)))
        self.T_i = np.array((T for in range(self.num_tasks)))
        

        # map generated position indices to positions
        self.tasks_positions = [self.cells[pos] for pos in self.tasks_positions_idx]
        self.agents_positions = [(BASE_X, BASE_Y) for i in range(K)]
        initial_action = [10 for i in range(self.num_agents)]
        initial_state = list(sum(self.tasks_positions + self.agents_positions + initial_action + self.T_i, ()))
        return initial_state

    def step(self, agents_actions):
        # check whether to go back
        if (np.array(self.B_k) - np.array(self.energy_required_back(self.agents_positions))).all() < 0:
            reward = Q
            self.terminal = True
        
        else:
            # update the position of agents
            self.agents_positions = self.update_positions(self.agents_positions, agents_actions)

            # update drones energy state
            self.B_k = self.update_agents_energy(agents_actions)
            # update task energy required 
            self.T_i = self.update_tasks_energy(self.agents_positions, agents_actions)
            # update 2D array (the portion of task at location i executed by drone k)
            self.y_ik = self.update_2D_array(self.agents_positions, agents_actions)
            # calculate reward based on the energy efficiency and the energy required to finsih the remining tasks
            reward = ALPHA * (np.sum(self.y_ik)/B*self.num_agents-np.sum(self.B_k)-EFFICIENCY_THRESHOLD)-np.sum(self.T_i)/T*self.num_tasks

            if np.sum(self.T_i)==0:
                self.terminal=True

        new_state = list(sum(self.tasks_positions + self.agents_positions + agents_actions + self.T_i, ()))

        return [new_state, reward, self.terminal]

    def energy_required_back(self, pos_list):
        energy_required = []
        for pos in pos_list:
            forward_energy = min(abs(pos[0] - BASE_X), abs(pos[1] - BASE_Y)) * C_F
            howard_energy = abs(abs(pos[0]) - abs(pos[1])) * C_F*(2**(1/2)/2) + C_H*(1-(2**(1/2)/2))
            energy_required.append(forward_energy + howard_energy)
        return energy_required

    def update_agents_energy(self, act_list):
        B_k = self.B_k
        for i in range (len(act_list)):
            if 4 <= act_list[i] <= 7:
                B_k[i] = B_k[i]-C_F
            elif act_list[i]<=3:
                B_k[i] = B_k[i]-C_F*(2**(1/2)/2)-C_H*(1-(2**(1/2)/2))
            elif act_list[i]==9:
                B_k[i] = B_k[i]-C_T-C_H
            else:
                B_k[i] = B_k[i]
        return B_k

    def update_tasks_energy(self, pos_list, act_list):
        T_i = self.T_i
        execute_idx = [i for i in range (len(act_list)) if act_list[i] == 9]
        agents_execute_pos = list(np.array(pos_list)[execute_idx])
        task_idx = []
        for agent_execute_pos in agents_execute_pos:
            for j in range (len(self.tasks_positions)):
                if (self.tasks_positions[j] == agent_execute_pos):
                    task_idx.append(j)
        for idx in task_idx:
            T_i[idx] = T_i[idx]-C_T
        
        return T_i
            
    def update_2D_array(self, pos_list, act_list):
        y_ik = self.y_ik
        execute_idx = [i for i in range (len(act_list)) if act_list[i] == 9]
        agents_execute_pos = list(np.array(pos_list)[execute_idx])
        for p in range (len(agents_execute_pos)):
            for q in range (len(self.tasks_positions)):
                if (self.tasks_positions[q] == agents_execute_pos[p]):
                    y_ik[q][p] = y_ik[q][p] + C_T
        return y_ik

    def update_positions(self, pos_list, act_list):
        positions_action_applied = []
        for idx in range(len(pos_list)):
            if act_list[idx] != 8 or act_list[idx] != 9:
                pos_act_applied = map(operator.add, pos_list[idx], self.action_diff[act_list[idx]])
                # checks to make sure the new pos in inside the grid
                for i in range(0, 2):
                    if pos_act_applied[i] < 0:
                        pos_act_applied[i] = 0
                    if pos_act_applied[i] >= self.grid_size:
                        pos_act_applied[i] = self.grid_size - 1
                positions_action_applied.append(tuple(pos_act_applied))
            else:
                positions_action_applied.append(pos_list[idx])

        final_positions = []

        for pos_idx in range(len(pos_list)):
            if positions_action_applied[pos_idx] == pos_list[pos_idx]:
                final_positions.append(pos_list[pos_idx])
            elif positions_action_applied[pos_idx] not in pos_list and positions_action_applied[
                pos_idx] not in positions_action_applied[
                                0:pos_idx] + positions_action_applied[
                                             pos_idx + 1:]:
                final_positions.append(positions_action_applied[pos_idx])
            else:
                final_positions.append(pos_list[pos_idx])

        return final_positions

    def action_space(self):
        return len(self.action_space)

    # def render(self):

    #     pygame.time.delay(500)
    #     pygame.display.flip()

    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             sys.exit()

    #     self.screen.fill(BLACK)
    #     text = self.my_font.render("Step: {0}".format(self.step_num), 1, WHITE)
    #     self.screen.blit(text, (5, 15))

    #     for row in range(self.grid_size):
    #         for column in range(self.grid_size):
    #             pos = (row, column)

    #             frequency = self.find_frequency(pos, self.agents_positions)

    #             if pos in self.landmarks_positions and frequency >= 1:
    #                 if frequency == 1:
    #                     self.screen.blit(self.img_agent_landmark,
    #                                      ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
    #                 else:
    #                     self.screen.blit(self.img_agent_agent_landmark,
    #                                      ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))

    #             elif pos in self.landmarks_positions:
    #                 self.screen.blit(self.img_landmark,
    #                                  ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))

    #             elif frequency >= 1:
    #                 if frequency == 1:
    #                     self.screen.blit(self.img_agent,
    #                                      ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
    #                 elif frequency > 1:
    #                     self.screen.blit(self.img_agent_agent,
    #                                      ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
    #                 else:
    #                     print('Error!')
    #             else:
    #                 pygame.draw.rect(self.screen, WHITE,
    #                                  [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50, WIDTH,
    #                                   HEIGHT])

    #     if self.recorder_flag:
    #         file_name = "%04d.png" % self.step_num
    #         pygame.image.save(self.screen, os.path.join(self.snaps_path, file_name))

    #     if not self.terminal:
    #         self.step_num += 1

    # def gui_setup(self):

    #     # Initialize pygame
    #     pygame.init()

    #     # Set the HEIGHT and WIDTH of the screen
    #     board_size_x = (WIDTH + MARGIN) * self.grid_size
    #     board_size_y = (HEIGHT + MARGIN) * self.grid_size

    #     window_size_x = int(board_size_x)
    #     window_size_y = int(board_size_y * 1.2)

    #     window_size = [window_size_x, window_size_y]
    #     screen = pygame.display.set_mode(window_size)

    #     # Set title of screen
    #     pygame.display.set_caption("Agents-and-Landmarks Game")

    #     myfont = pygame.font.SysFont("monospace", 30)

    #     return [screen, myfont]

    def find_frequency(self, a, items):
        freq = 0
        for item in items:
            if item == a:
                freq += 1

        return freq