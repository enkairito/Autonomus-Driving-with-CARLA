# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 23:07:55 2019

@author: marta
"""
import cv2
import math
import socket
import pygame
import random
import time
import numpy as np

import carla
import settings

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

class CarEnv:
        SHOW_CAM = settings.SHOW_PREVIEW
        STEER_AMT = 1.0
        im_width = settings.IM_WIDTH
        im_height = settings.IM_HEIGHT
        front_camera = None
    
        def __init__(self):

            self.client = carla.Client(settings.HOST, settings.PORT)
            self.client.set_timeout(settings.TIMEOUT)
                
            try: 
                self.world = self.client.get_world()
            except: 
                while (socket.timeout == False):
                    self.world = self.client.get_world()
                    
                self.client = carla.Client(settings.HOST, settings.PORT)
                self.world = self.client.get_world() 
            
            self.blueprint_library = self.world.get_blueprint_library()
            self.model_3 = self.blueprint_library.filter(settings.CAR_MODEL)[0]
    
        def reset(self):
            self.collision_hist = []
            self.actor_list = []
            
            if (settings.FIXED_SPAWN == True):            
                self.transform = carla.Transform(settings.SPAWN_POINT, settings.ROTATION) 
            else: 
                self.transform = random.choice(self.world.get_map().get_spawn_points()) 
                
            try: 
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            except: 
                while (socket.timeout == False):
                    self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            self.actor_list.append(self.vehicle)
            
            # Camera
            if (settings.CAMERA == True):
                if (settings.SEMANTIC_SEGMENTATION == True): 
                    self.cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
                else:
                    self.cam = self.blueprint_library.find('sensor.camera.rgb')
                self.cam.set_attribute("image_size_x", f"{self.im_width}")
                self.cam.set_attribute("image_size_y", f"{self.im_height}")
                self.cam.set_attribute("fov", f"110")
        
                transform = carla.Transform(carla.Location(x=2.5, z=0.7))
                try: 
                    self.sensor = self.world.spawn_actor(self.cam, transform, attach_to=self.vehicle)
                except: 
                    while (socket.timeout == False):
                        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
                        
                self.actor_list.append(self.sensor)
                self.sensor.listen(lambda data: self.process_img(data))
        
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
                time.sleep(4)
            
            # Collision sensor
            if (settings.COLLISION_SENSOR == True): 
                colsensor = self.blueprint_library.find("sensor.other.collision")
                
                try: 
                    self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
                except: 
                     while (socket.timeout == False):
                         self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
                
                self.actor_list.append(self.colsensor)
                self.colsensor.listen(lambda event: self.collision_data(event))
            
            # LIDAR
            if (settings.LIDAR == True):
                lidarsensor = self.blueprint_library.find("sensor.lidar.ray_cast")
                try: 
                    self.lidarsensor = self.world.spawn_actor(lidarsensor, transform, attach_to=self.vehicle)
                except: 
                     while (socket.timeout == False):
                         self.lidarsensor = self.world.spawn_actor(lidarsensor, transform, attach_to=self.vehicle)    
                        
                self.actor_list.append(self.lidarsensor)
                self.sensor.listen(lambda data: self.lidar_data())
            
            # GNNS
            if (settings.GNSS == True):
                gnss = self.blueprint_library.find('sensor.other.gnss')
                try: 
                    self.gnss = self.world.spawn_actor(gnss, transform, attach_to=self.vehicle)
                except: 
                     while (socket.timeout == False):
                         self.gnss = self.world.spawn_actor(gnss, transform, attach_to=self.vehicle)    
                        
                self.actor_list.append(self.gnss)
                self.sensor.listen(lambda event: self.gnss_data())
            
            # Lane invasion sensor
            if (settings.LANE_INVASION == True):
                laneinvasion = self.blueprint_library.find('sensor.other.lane_invasion')
                try: 
                    self.laneinvasion = self.world.spawn_actor(laneinvasion, transform, attach_to=self.vehicle)
                except: 
                     while (socket.timeout == False):
                         self.laneinvasion = self.world.spawn_actor(laneinvasion, transform, attach_to=self.vehicle)    
                        
                self.actor_list.append(self.laneinvasion)
                self.sensor.listen(lambda event: self.laneinvasion_data())
    

            while self.front_camera is None:
                time.sleep(0.01)
    
            self.episode_start = time.time()
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
    
            return self.front_camera
    
        def collision_data(self, event):
            self.collision_hist.append(event)
        
        def gnss_data(self, event):
            self.lat = event.latitude
            self.lon = event.longitude
            
        def laneinvasion_data(self, event):
            self.invasion_hist.append(event)
            
        def lidar_data(self, image):  
           i = np.array(image.raw_data)
           i2 = i.reshape((int(i.shape[0]/3),3))
           lidar_data = np.array(i2[:,:2])
           lidar_data *= min(self.hud.dim)/ 100.0
           lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
           lidar_data = np.fabs(lidar_data)
           lidar_data = lidar_data.astype(np.int32)
           lidar_data = lidar_data.reshape((-1, 2))
           lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
           lidar_img = np.zeros((lidar_img_size), dtype=int)
           lidar_img[tuple(lidar_data.T)] = (255,255,255)
           self.surface = pygame.surfarray.make_surface(lidar_img)
           
            
        def process_img(self, image):
            i = np.array(image.raw_data)
            #print(i.shape)
            i2 = i.reshape((self.im_height, self.im_width, 4))
            i3 = i2[:, :, :3]
            if self.SHOW_CAM:
                cv2.imshow("", i3)
                cv2.waitKey(1)
            self.front_camera = i3
    
    # DEFINE ACTIONS
        def step(self, action):
            # left turn (90º)
            if action == 0:
                self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
            # go ahead
            elif action == 1:
                self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
            # right turn (90º)
            elif action == 2:
                self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
            # brake
            elif action == 3: 
                self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=0))
            
    # DEFINE REWARDS
                
            v = self.vehicle.get_velocity()
            kmh = float(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
            
            a = self.vehicle.get_acceleration()
            
            if len(self.collision_hist) != 0:
                done = True
                reward = -20
            elif ( ((self.episode_start + 10 < time.time()) and (kmh < 1))): # if it gets stuck with hand brake or brakes more than it should
                done = True
                reward = -8
            elif kmh < 50:
                done = False
                reward = -3
            elif (action == 0 or action == 2):
                done = False
                reward = -1*abs(a.z)
            elif (self.episode_start > 10):
                done = False
                reward = 5
           
            elif kmh > 50:
                done = False
                reward = 3
            else: 
                done = False
    
          #Penalize if the car collides really fast
            if (done and (self.episode_start <= 10)):
                reward = -10
    
            return self.front_camera, reward, done, None