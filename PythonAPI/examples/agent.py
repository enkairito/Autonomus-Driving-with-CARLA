# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 23:11:12 2019

@author: marta
"""

import settings, ModTensorboard
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Dense, MaxPooling2D, Dropout, Conv2D, Flatten, Activation 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model

import random
import time
import numpy as np

class DQNAgent:
        def __init__(self):
            if (settings.LOAD_MODEL == False):
                self.model = self.create_model()
                self.target_model = self.create_model()
            else: 
                self.model = load_model(settings.MODEL)
                self.target_model = load_model(settings.MODEL)
                
            
            self.target_model.set_weights(self.model.get_weights())
    
            self.replay_memory = deque(maxlen=settings.REPLAY_MEMORY_SIZE)
    
            self.tensorboard = ModTensorboard.ModifiedTensorBoard(log_dir=f"logs/{settings.MODEL_NAME}-{int(time.time())}")
            self.target_update_counter = 0
            self.graph = tf.compat.v1.get_default_graph()
    
            self.terminate = False
            self.last_logged_episode = 0
            self.training_initialized = False
    
        def create_model(self):
            model = Sequential()
            
            model.add(Conv2D(64, (3, 3), input_shape=(settings.IM_HEIGHT, settings.IM_WIDTH,3),padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.1))
                
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
               
            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(64))
            
            model.add(Dense(4, activation='linear'))
            model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
            return model
    
        def update_replay_memory(self, transition):
            # transition = (current_state, action, reward, new_state, done)
            self.replay_memory.append(transition)
    
        def train(self):
            if len(self.replay_memory) < settings.MIN_REPLAY_MEMORY_SIZE:
                return
    
            minibatch = random.sample(self.replay_memory, settings.MINIBATCH_SIZE)
    
            current_states = np.array([transition[0] for transition in minibatch])/255
            with self.graph.as_default():
                current_qs_list = self.model.predict(current_states, settings.PREDICTION_BATCH_SIZE)
    
            new_current_states = np.array([transition[3] for transition in minibatch])/255
            with self.graph.as_default():
                future_qs_list = self.target_model.predict(new_current_states, settings.PREDICTION_BATCH_SIZE)
    
            X = []
            y = []
    
            for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + settings.DISCOUNT * max_future_q
                else:
                    new_q = reward
    
                current_qs = current_qs_list[index]
                current_qs[action] = new_q
    
                X.append(current_state)
                y.append(current_qs)
    
            log_this_step = False
            if self.tensorboard.step > self.last_logged_episode:
                log_this_step = True
                self.last_log_episode = self.tensorboard.step
    
            with self.graph.as_default():
                self.model.fit(np.array(X)/255, np.array(y), batch_size=settings.TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)
    
    
            if log_this_step:
                self.target_update_counter += 1
    
            if self.target_update_counter > settings.UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0
    
        def get_qs(self, state):
            return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
        def train_in_loop(self):
            X = np.random.uniform(size=(1, settings.IM_HEIGHT, settings.IM_WIDTH, 3)).astype(np.float32)
            y = np.random.uniform(size=(1, 4)).astype(np.float32)
            with self.graph.as_default():
                self.model.fit(X,y, verbose=False, batch_size=1)
    
            self.training_initialized = True
    
            while True:
                if self.terminate:
                    return
                self.train()
                time.sleep(0.01)