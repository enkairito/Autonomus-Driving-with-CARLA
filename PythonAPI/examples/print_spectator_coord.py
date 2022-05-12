# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:42:33 2019

@author: marta
"""

#!/usr/bin/env python

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

_HOST_ = "localhost"
_PORT_ = 6000
_SLEEP_TIME_ = 1


def main():
	client = carla.Client(_HOST_, _PORT_)
	client.set_timeout(5.0)
	world = client.get_world()
	
	# print(help(t))
	# print("(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z))
	

	while(True):
		t = world.get_spectator().get_transform()
		# coordinate_str = "(x,y) = ({},{})".format(t.location.x, t.location.y)
		coordinate_str = "(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z)
		print (coordinate_str)
		time.sleep(_SLEEP_TIME_)



if __name__ == '__main__':
	main()