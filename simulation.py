"""
Created on Wednesday Jan  16 2019
@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import numpy as np
import os
import random
import argparse
import pandas as pd
from stage1_env import Environment
from stage1_agent import Agent
import glob

ARG_LIST = ['learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency', 'maximum_exploration',
            'max_timestep', 'first_step_memory', 'replay_steps', 'number_nodes', 'target_type', 'memory',
            'prioritization_scale', 'dueling', 'agents_number', 'grid_size']


def get_name_brain(args, idx):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results/' + file_name_str + '_' + str(idx) + '.h5'


def get_name_rewards(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results/' + file_name_str + '.csv'


def get_name_timesteps(args):

    file_name_str = '_'.join([str(args[x]) for x in ARG_LIST])

    return './results/' + file_name_str + '.csv'


class Simulation(object):

    def __init__(self, arguments):
        # current_path = os.path.dirname(__file__)  # Where your .py file is located
        current_path = '/Users/cli22/Desktop/DNRL-main/DNRL'
        self.env = Environment(arguments, current_path)
        self.episodes_number = arguments['episode_number']
        self.render = arguments['render'] # delete
        self.recorder = arguments['recorder']
        self.max_ts = arguments['max_timestep']
        self.test = arguments['test']
        self.filling_steps = arguments['first_step_memory']
        self.steps_b_updates = arguments['replay_steps']
        self.max_random_moves = arguments['max_random_moves']

        self.num_agents = arguments['agents_number']
        self.num_tasks = self.num_agents
        self.grid_size = arguments['grid_size']

    def run(self, agents, file1, file2):

        total_step = 0
        rewards_list = []
        timesteps_list = []
        max_score = -10000
        for episode_num in range(self.episodes_number):
            state = self.env.reset()
            # if self.render:# delete
            #     self.env.render()

            random_moves = random.randint(0, self.max_random_moves)

            # create randomness in initial state
            for _ in range(random_moves):
                actions = [10 for _ in range(len(agents))]
                state, _, _ = self.env.step(actions)
                # if self.render:
                #     self.env.render()

            # converting list of positions to an array
            state = np.array(state)
            state = state.ravel()

            done = False
            reward_all = 0
            time_step = 0
            while not done and time_step < self.max_ts:

                # if self.render:
                #     self.env.render()
                actions = []
                i = 0
                for agent in agents:  
                    actions.append(agent.greedy_actor(state,i))
                    i += 1
                
                next_state, reward, done = self.env.step(actions)
                # converting list of positions to an array
                next_state = np.array(next_state)
                next_state = next_state.ravel()

                if not self.test:
                    for agent in agents:
                        agent.observe((state, actions, reward, next_state, done))
                        if total_step >= self.filling_steps:
                            agent.decay_epsilon()
                            if time_step % self.steps_b_updates == 0:
                                agent.replay()
                            agent.update_target_model()
                if time_step % 100 == 0:
                    print(f'current reward:{reward}')
                    print(f'current actions:{actions}')
                total_step += 1
                time_step += 1
                state = next_state
                reward_all += reward
                #
                # if self.render:
                #     self.env.render()

            rewards_list.append(reward_all)
            timesteps_list.append(time_step)

            print("Episode {p}, Score: {s}, Final Step: {t}, Goal: {g}".format(p=episode_num, s=reward_all,
                                                                               t=time_step, g=done))

            if not self.test:
                if episode_num % 100 == 0:
                    df = pd.DataFrame(rewards_list, columns=['score'])
                    df.to_csv(file1)

                    df = pd.DataFrame(timesteps_list, columns=['steps'])
                    df.to_csv(file2)

                    if total_step >= self.filling_steps:
                        if reward_all > max_score:
                            for agent in agents:
                                agent.brain.save_model()
                            max_score = reward_all


if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    # DQN Parameters
    parser.add_argument('-e', '--episode-number', default=1000000, type=int, help='Number of episodes')
    parser.add_argument('-l', '--learning-rate', default=0.00005, type=float, help='Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='RMSProp',
                        help='Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=1000000, type=int, help='Memory capacity')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('-t', '--target-frequency', default=10000, type=int,
                        help='Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=100000, type=int, help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory', default=0, type=float,
                        help='Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=4, type=float, help='Steps between updating the network')
    parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
    parser.add_argument('-tt', '--target-type', choices=['DQN', 'DDQN'], default='DDQN')
    parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='UER')
    parser.add_argument('-pl', '--prioritization-scale', default=0.5, type=float, help='Scale for prioritization')
    parser.add_argument('-du', '--dueling', action='store_true', help='Enable Dueling architecture if "store_false" ')

    parser.add_argument('-gn', '--gpu-num', default='2', type=str, help='Number of GPU to use')
    parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase if "store_false"')

    # Game Parameters
    parser.add_argument('-k', '--agents-number', default=9, type=int, help='The number of agents')
    parser.add_argument('-g', '--grid-size', default=300, type=int, help='Grid size')
    parser.add_argument('-ts', '--max-timestep', default=100000, type=int, help='Maximum number of timesteps per episode')

    parser.add_argument('-rm', '--max-random-moves', default=0, type=int,
                        help='Maximum number of random initial moves for the agents')


    # Visualization Parameters
    parser.add_argument('-r', '--render', action='store_false', help='Turn on visualization if "store_false"')
    parser.add_argument('-re', '--recorder', action='store_true', help='Store the visualization as a movie '
                                                                       'if "store_false"')

    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_num']
    sim = Simulation(args)

    state_size = sim.env.state_size
    action_space = sim.env.get_action_space_size()

    all_agents = []
    for b_idx in range(args['agents_number']):

        brain_file = get_name_brain(args, b_idx)
        all_agents.append(Agent(state_size, action_space, b_idx, brain_file, args))

    rewards_file = get_name_rewards(args)
    timesteps_file = get_name_timesteps(args)

    sim.run(all_agents, rewards_file, timesteps_file)
