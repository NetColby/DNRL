'''
Jiyao Chen
June 15 2021
Environment for stage 1
'''

import random
import numpy as np
import operator

K = 9  # number of drones
NUMBER_OF_TASKS = 9  # number of tasks
GRID_SIZE = 7  # size of the grid/environment
ALPHA = 0.5  # constant coefficient for execution in reward function
C_H = 2  # energy consumption rate for hovering
C_T = 3  # energy consumption rate for task execution
C_F = 2.5  # energy consumption rate for forwarding
EFFICIENCY_THRESHOLD = 0.3  # energy efficiency threshold for calculating reward
B = 15000  # battery capacity for drone
T = 2100  # energy required for one single task
Q = -50  # drone not coming back penalty
TRAVEL_ENERGY_THRESHOLD = 8000

# actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
UPPER_LEFT = 4
UPPER_RIGHT = 5
BOTTOM_LEFT = 6
BOTTOM_RIGHT = 7
STATIONARY = 8
EXECUTE = 9

# base station
BASE_X = 6
BASE_Y = 6


class Environment:
    def __init__(self, args, current_path):
        self.num_agents = K
        self.num_tasks = NUMBER_OF_TASKS
        self.grid_size = GRID_SIZE
        self.state_size = self.num_agents * 4 + self.num_tasks * 3  # not used
        self.agents_positions = []
        self.tasks_positions = []
        self.cells = []

        self.y_ik = np.zeros((self.num_tasks, self.num_agents))
        self.B_k = np.array([B for i in range(self.num_agents)])
        self.T_i = np.array([T for i in range(self.num_tasks)])
        # self.tasks_positions_idx = np.random.choice(len(self.cells) - 1, size=self.num_tasks,
        #                                     replace=False)
        self.tasks_positions_idx = []

        self.action_space = [UP, DOWN, LEFT, RIGHT, UPPER_LEFT, UPPER_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, STATIONARY, EXECUTE]
        self.action_diff = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0), (0, 0)]

        self.num_episodes = 0
        self.terminal = False

    def set_positions_idx(self):

        cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)]
        tasks_positions_idx = [8, 10, 12, 22, 24, 26, 36, 38, 40]
        return [cells, tasks_positions_idx]

    def reset(self):  # initialize the world

        self.terminal = False
        [self.cells, self.tasks_positions_idx] = self.set_positions_idx()
        self.y_ik = np.zeros((self.num_tasks, self.num_agents))
        self.B_k = np.array([B for i in range(self.num_agents)])
        self.T_i = np.array([T for i in range(self.num_tasks)])

        # map generated position indices to positions
        self.tasks_positions = [self.cells[pos] for pos in self.tasks_positions_idx]
        self.agents_positions = [(BASE_X, BASE_Y) for i in range(K)]
        initial_actions = [8 for i in range(self.num_agents)]
        initial_pos_state = list(sum(self.tasks_positions + self.agents_positions, ()))
        initial_state = initial_pos_state + initial_actions + list(self.T_i)
        initial_state = initial_state + list(self.B_k)
        return initial_state

    def step(self, agents_actions):
        # task finisihed & drones all go back to base station
        # update the position of agents
        self.agents_positions = self.update_positions(self.agents_positions, agents_actions)
        #
        # update drones energy state
        self.B_k = self.update_agents_energy(agents_actions)
        # update task energy required
        self.T_i = self.update_tasks_energy(self.agents_positions, agents_actions)
        # update 2D array (the portion of task at location i executed by drone k)
        self.y_ik = self.update_2D_array(self.agents_positions, agents_actions)

        if self.T_i.all()==0 and np.all(np.asarray(self.agents_positions)==(BASE_X,BASE_Y)):
            # reward = np.sum(self.y_ik) / (T * self.num_tasks) + np.sum(self.y_ik) / (B * self.num_agents - np.sum(self.B_k))
            reward = 1 - ((B*self.num_agents-np.sum(self.B_k))-(T*self.num_tasks-np.sum(self.T_i)))/(B*self.num_agents-np.sum(self.B_k)) + (T*self.num_tasks-np.sum(self.T_i))/(T*self.num_tasks) + np.sum(self.B_k, where = self.B_k < 0)/(B*self.num_agents-np.sum(self.B_k)) + (self.B_k>0).sum()/self.num_agents
            self.terminal = True

        #task unfinished
        else:
            reward = 1 - ((B*self.num_agents-np.sum(self.B_k))-(T*self.num_tasks-np.sum(self.T_i)))/(B*self.num_agents-np.sum(self.B_k)) + (T*self.num_tasks-np.sum(self.T_i))/(T*self.num_tasks) + np.sum(self.B_k, where = self.B_k < 0)/(B*self.num_agents-np.sum(self.B_k))
            # current_travel_energy = (B * self.num_agents - np.sum(self.B_k)) - np.sum(self.y_ik)
            # if current_travel_energy <= TRAVEL_ENERGY_THRESHOLD:
            #     reward = 0.55*np.sum(self.y_ik) / (T * self.num_tasks) + 0.4*np.sum(self.y_ik) / (B * self.num_agents - np.sum(self.B_k)) + 0.05*np.sum(self.B_k, where = self.B_k < 0) / (B * self.num_agents)
            # else:
            #     reward = 0.2*(TRAVEL_ENERGY_THRESHOLD - current_travel_energy) / TRAVEL_ENERGY_THRESHOLD + 0.7*np.sum(self.y_ik) / (T * self.num_tasks) + 0.1*np.sum(self.y_ik) / (B * self.num_agents - np.sum(self.B_k))

        new_pos_state = list(sum(self.tasks_positions + self.agents_positions, ()))
        new_state = new_pos_state + agents_actions + list(self.T_i) + list(self.B_k)
        return [new_state, reward, self.terminal]

    # def energy_required_back(self, pos_list):
    #     energy_required = []
    #     for pos in pos_list:
    #         forward_energy = min(abs(pos[0] - BASE_X), abs(pos[1] - BASE_Y)) * C_F
    #         howard_energy = abs(abs(pos[0]) - abs(pos[1])) * C_F * (2 ** (1 / 2) / 2) + C_H * (1 - (2 ** (1 / 2) / 2))
    #         energy_required.append(forward_energy + howard_energy)
    #     return energy_required

    def update_agents_energy(self, act_list):
        B_k = self.B_k
        for i in range(len(act_list)):
            if self.agents_positions[i] == (BASE_X,BASE_Y) and act_list[i]==8:
                B_k[i] = B_k[i]
            if 4 <= act_list[i] <= 7:
                B_k[i] = B_k[i] - C_F
            elif act_list[i] <= 3:
                B_k[i] = B_k[i] - C_F * (2 ** (1 / 2) / 2) - C_H * (1 - (2 ** (1 / 2) / 2))
            elif act_list[i] == 9:
                B_k[i] = B_k[i] - C_T - C_H
            else:
                B_k[i] = B_k[i] - C_H
        return B_k

    def update_tasks_energy(self, pos_list, act_list):
        T_i = self.T_i
        execute_idx = [i for i in range(len(act_list)) if act_list[i] == 9]
        agents_execute_pos = list(np.array(pos_list)[execute_idx])
        task_idx = []
        for i in range(len(agents_execute_pos)):
            for j in range(len(self.tasks_positions)):
                if (self.tasks_positions[j] == tuple(agents_execute_pos[i]) and self.B_k[execute_idx[i]]>C_T+C_H and self.T_i[j] >= C_T):
                    task_idx.append(j)
        for idx in task_idx:
            T_i[idx] = T_i[idx] - C_T

        return T_i

    def update_2D_array(self, pos_list, act_list):
        y_ik = self.y_ik
        execute_idx = [i for i in range(len(act_list)) if act_list[i] == 9]
        agents_execute_pos = list(np.array(pos_list)[execute_idx])
        for p in range(len(agents_execute_pos)):
            for q in range(len(self.tasks_positions)):
                if (self.tasks_positions[q] == tuple(agents_execute_pos[p]) and self.B_k[execute_idx[p]]>C_T+C_H):
                    y_ik[q][p] = y_ik[q][p] + C_T
        return y_ik

    def update_positions(self, pos_list, act_list):
        positions_action_applied = []
        for idx in range(len(pos_list)):
            if act_list[idx] != 8 and act_list[idx] != 9:
                # pos_act_applied = map(operator.add, pos_list[idx], self.action_diff[act_list[idx]])
                pos_act_applied = list(np.asarray(pos_list[idx]) + np.asarray(self.action_diff[act_list[idx]]))

                # checks to make sure the new pos in inside the grid
                # for key in pos_act_applied:
                #     print(key)
                for i in range(0, 2):
                    if pos_act_applied[i] < 0:
                        pos_act_applied[i] = 0
                    if pos_act_applied[i] >= self.grid_size:
                        pos_act_applied[i] = self.grid_size - 1
                positions_action_applied.append(tuple(pos_act_applied))
            else:
                positions_action_applied.append(pos_list[idx])

        # final_positions = []
        #
        # for pos_idx in range(len(pos_list)):
        #     if positions_action_applied[pos_idx] == pos_list[pos_idx]:
        #         final_positions.append(pos_list[pos_idx])
        #     elif positions_action_applied[pos_idx] not in pos_list and positions_action_applied[
        #         pos_idx] not in positions_action_applied[
        #                         0:pos_idx] + positions_action_applied[
        #                                      pos_idx + 1:]:
        #         final_positions.append(positions_action_applied[pos_idx])
        #     else:
        #         final_positions.append(pos_list[pos_idx])

        return positions_action_applied

    def get_action_space_size(self):
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
