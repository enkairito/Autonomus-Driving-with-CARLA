# -*- coding: utf-8 -*-
"""
@author: Marta Basquens


Main program for applying Reinforcement Learning on a car using CARLA Simulator. 

Usage: 
    1. Set all desired settings on script "Settings"
    2. Prior running this script, open CARLA Simulator on port specified in 
    "Settings" script under name "PORT"

For more information refer to the Bachelor Thesis
"STUDY OF THE IMPLEMENTATION OF AN AUTONOMOUS DRIVING SYSTEM"
"""

#-------------------------- set gpu using tf ---------------------------
import glob
import os
import sys
import tensorflow as tf
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


import random
import time
import numpy as np

import keras.backend.tensorflow_backend as backend
from threading import Thread

from tqdm import tqdm


import settings, carEnv, agent
     
    
if __name__ == '__main__':
        # For stats
        if (settings.LOAD_MODEL == False):
            ep_rewards = [-200]
        else: 
            ep_rewards = [settings.MIN_REWARD]
    
        # For more repetitive results
        random.seed(10)
        np.random.seed(1)
        tf.compat.v1.set_random_seed(24)
    
        # Memory fraction, used mostly when trai8ning multiple agents
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=settings.MEMORY_FRACTION)
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))
    
        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')
    
        # Create agent and environment
        agent = agent.DQNAgent()
        env = carEnv.CarEnv()
    
    
        # Start training thread and wait for training to be initialized
        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent.training_initialized:
            time.sleep(0.01)
    
        # Initialize predictions - forst prediction takes longer as of initialization that has to be done
        # It's better to do a first prediction then before we start iterating over episode steps
        agent.get_qs(np.ones((env.im_height, env.im_width, 3)))
    
        # Iterate over episodes
        for episode in tqdm(range(1, settings.EPISODES + 1), ascii=True, unit='episodes'):
            #try:
    
                env.collision_hist = []
    
                # Update tensorboard step every episode
                agent.tensorboard.step = episode
    
                # Restarting episode - reset episode reward and step number
                if (settings.LOAD_MODEL == False):
                    episode_reward = 0
                    step = 1
                else:
                    episode_reward = settings.AVG_REWARD
                    step = settings.PRV_STEP
    
                # Reset environment and get initial state
                current_state = env.reset()
    
                # Reset flag and start iterating until episode ends
                done = False
                episode_start = time.time()
    
                # Play for given number of seconds only
                while True:
    
                    # This part stays mostly the same, the change is to query a model for Q values
                    if np.random.random() > settings.epsilon:
                        # Get action from Q table
                        action = np.argmax(agent.get_qs(current_state))
                    else:
                        # Get random action
                        action = np.random.randint(0, 4)
                        # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                        time.sleep(1/settings.FPS)
    
                    new_state, reward, done, _ = env.step(action)
    
                    # Transform new continous state to new discrete state and count reward
                    episode_reward += reward
    
                    # Every step we update replay memory
                    agent.update_replay_memory((current_state, action, reward, new_state, done))
    
                    current_state = new_state
                    step += 1
    
                    if done:
                        break
    
                # End of episode - destroy agents
                for actor in env.actor_list:
                    actor.destroy()
    
                # Append episode reward to a list and log stats (every given number of episodes)
                ep_rewards.append(episode_reward)
                if not episode % settings.AGGREGATE_STATS_EVERY or episode == 1:
                    average_reward = sum(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])/len(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
                    min_reward = min(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
                    max_reward = max(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
                    agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=settings.epsilon)
    
                    # Save model, but only when min reward is greater or equal a set value
                    if min_reward >= settings.MIN_REWARD:
                        agent.model.save(f'models/{settings.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
                
                # Decay epsilon
                if (settings.TRAINING == True):
                    if ((average_reward < -150) and (settings.epsilon <= 0.4)):
                        settings.epsilon *= settings.EPSILON_RECOVERY
                        settings.epsilon = max(settings.MAX_EPSILON, settings.epsilon)
                    elif ((average_reward < -100) and (step > settings.EPISODES/1.5)):
                        settings.epsilon *= settings.EPSILON_RECOVERY
                        settings.epsilon = max(settings.MAX_EPSILON, settings.epsilon)
                    else: 
                        settings.epsilon *= settings.EPSILON_DECAY
                        settings.epsilon = max(settings.MIN_EPSILON, settings.epsilon)
    
    
        # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.model.save(f'models/{settings.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')