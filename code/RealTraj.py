from scipy import optimize
import numpy as np
from numpy import arctan, cos, sin, sign, pi, arcsin, arccos
from numpy.linalg import norm
import matplotlib.pyplot as plt
from itertools import combinations

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import losses, regularizers
from keras import backend as K
import matplotlib.pyplot as plt

def overlap(x, y, r):
    return x**2+y**2<=r**2

def outbound(ball, geometry):
    width, height = geometry
    x, y, r, m, vx, vy = ball
    if x-r<0 or y-r<0 or x+r>width or y+r>height:
        return True
    else:
        return False
    
def check_overlap(balls, step):
    comb = combinations(range(len(balls)), 2)
    for i, j in comb:
        if overlap(balls[i][0]-balls[j][0],balls[i][1]-balls[j][1],balls[i][2]+balls[j][2]):
            print("Overlapping at step "+str(step)+"!")
            return True
    return False

def col_time_ball(ball_1, ball_2, dt):
    x1, y1, r1, m1, vx1, vy1 = ball_1
    x2, y2, r2, m2, vx2, vy2 = ball_2
    if overlap(x1+(vx1-vx2)*dt-x2, y1+(vy1-vy2)*dt-y2, r1+r2): # collide within dt
        beta = arctan((y1-y2)/(x1-x2)) - (sign(x1-x2)-1)*pi/2
        rot_mat = np.array([[cos(beta), sin(beta)],[-sin(beta), cos(beta)]])
        rot_mat_inv = np.array([[cos(-beta), sin(-beta)],[-sin(-beta), cos(-beta)]])
        x1pp = ((x1-x2)**2+(y1-y2)**2)**0.5
        vx1pp, vy1pp = rot_mat@np.array([vx1 - vx2, vy1 - vy2])
        d12 = np.array([-x1pp, 0])
        v12 = np.array([vx1pp, vy1pp])
        theta = arccos(d12@v12/(norm(d12)*norm(v12))) * (-1)**((sign(vy1pp)-1)/2)
        phi = arcsin(x1pp/(r1+r2)*sin(theta))
        alpha = phi - theta
        s1c = sin(alpha)/sin(theta)*(r1+r2)
        dtc = s1c/norm(v12)
        return dtc
    else:
        return dt+1
    
def col_time_wall(ball_1, geometry, dt):
    x1, y1, r1, m1, vx1, vy1 = ball_1
    width, height = geometry
    dt_c = dt + 1
    wall_index = 0
    if vx1 > 0:
        time = (width - x1 - r1)/vx1
        if time < dt_c:
            dt_c = time
            wall_index = -2
    if vx1 < 0:
        time = (x1-r1)/(-vx1)
        if time < dt_c:
            dt_c = time
            wall_index = -1
    if vy1 > 0:
        time = (height - y1 - r1)/vy1
        if time < dt_c:
            dt_c = time
            wall_index = -4
    if vy1 < 0:
        time = (y1-r1)/(-vy1)
        if time < dt_c:
            dt_c = time
            wall_index = -3    
    return dt_c, wall_index

def event_handler(balls, geometry, dt):
    comb = combinations(range(len(balls)), 2)
    dt_c = dt
    ball_1_index = -1
    item_2_index = -5
    for i, j in comb:
        dt_c_b = col_time_ball(balls[i], balls[j], dt)
        if dt_c_b < dt_c:
            dt_c = dt_c_b
            ball_1_index = i
            item_2_index = j
    for i in range(len(balls)):
        dt_c_w, wall_index = col_time_wall(balls[i], geometry, dt)
        if dt_c_w < dt_c:
            dt_c = dt_c_w
            ball_1_index = i
            item_2_index = wall_index
    return dt_c, ball_1_index, item_2_index

def update(balls,obj1, obj2, dt_c, geometry, dt = 1): 
	width, height = geometry
	if obj1 == -1:
		for i in range(len(balls)):
			x1, y1, r1, m1, vx1, vy1 = balls[i]
			x1 += vx1*dt
			y1 += vy1*dt
			balls[i] = x1, y1, r1, m1, vx1, vy1
	else:
		if obj2 < 0: # this is the wall
			x1, y1, r1, m1, vx1, vy1 = balls[obj1]
			if obj2 == -1: # left vertical
				y1 = y1 + vy1*dt
				x1 = 2*r1 - (x1 + vx1*dt)
				vx1 = - vx1
			if obj2 == -2: # right vertical
				y1 = y1 + vy1*dt
				x1 = 2*(width-r1) - (x1 + vx1*dt)
				vx1 = - vx1
			if obj2 == -3: # bottom horizontal
				x1 = x1 + vx1*dt
				y1 = 2*r1 - (y1 + vy1*dt)
				vy1 = - vy1
			if obj2 == -4: # top horizontal
				x1 = x1 + vx1*dt
				y1 = 2*(height-r1) - (y1 + vy1*dt)
				vy1 = - vy1
			balls[obj1] = x1, y1, r1, m1, vx1, vy1
			for i in range(len(balls)):
				if i!=obj1:
					x1, y1, r1, m1, vx1, vy1 = balls[i]
					x1 += vx1*dt
					y1 += vy1*dt
					balls[i] = x1, y1, r1, m1, vx1, vy1
            
		else:  # two balls 
			x1, y1, r1, m1, vx1, vy1 = balls[obj1]
			x2, y2, r2, m2, vx2, vy2 = balls[obj2]
            
            #eneold = m1*(vx1**2+vy1**2) + m2*(vx2**2+vy2**2)
            
            # New axes with new x pointing from ball 2 to ball 1. 
            # Beta is the rotation angle from old axis to the new axis
			beta = arctan((y1-y2)/(x1-x2)) - (sign(x1-x2)-1)*pi/2

            # The corresponding rotation matrix and its inverse
			rot_mat = np.array([[cos(beta), sin(beta)],[-sin(beta), cos(beta)]])
			rot_mat_inv = np.array([[cos(-beta), sin(-beta)],[-sin(-beta), cos(-beta)]])

            # New coordinate and speed of ball 1. Both for ball 2 are zero
			x1pp = ((x1-x2)**2+(y1-y2)**2)**0.5
			vx1pp, vy1pp = rot_mat@np.array([vx1 - vx2, vy1 - vy2])
			d12 = np.array([-x1pp, 0])
			v12 = np.array([vx1pp, vy1pp])

            # Three angles defining the collision point
			theta = arccos(d12@v12/(norm(d12)*norm(v12))) * (-1)**((sign(vy1pp)-1)/2)

			phi = arcsin(x1pp/(r1+r2)*sin(theta))
			alpha = phi - theta

            # Distance travelled by ball 1 before collision
			s1c = sin(alpha)/sin(theta)*(r1+r2)

            # Time before the collision
			dtc = s1c/norm(v12)
			if dtc!=dt_c:
				print("Critical error!")

            # Matrix transforming relative velocity to normal velocity and then back to components
			col_angle_mat = np.array([[cos(alpha)**2, sin(alpha)*cos(alpha)],[sin(alpha)*cos(alpha), sin(alpha)**2]])

            # Distance from collision point to destination
			scf = norm(v12)*(dt - dtc)

            # Coordinates of scf
			x1ppp = scf*cos(phi+alpha)+(r1+r2)*cos(alpha)
			y1ppp = scf*sin(phi+alpha)+(r1+r2)*sin(alpha)

            # total distance travelled by the reference frame
			drf = np.array([vx2, vy2])*dt + rot_mat_inv@col_angle_mat@(np.array([vx1pp, vy1pp])*2*m1/(m1+m2))*(dt-dtc)

            # New x1 and y1
			x1new, y1new = rot_mat_inv@np.array([x1ppp, y1ppp]) + drf + np.array([x2,y2])
			x2new, y2new = drf + np.array([x2,y2])

            # New velocities of the two balls
            
			if m2 == np.inf:
				param1 = 0
				param2 = -2
			else:
				param1 = 2*m1/(m1+m2)
				param2 = 2*(m1-m2)/(m1+m2)
            
			vx1new, vy1new = np.array([vx1, vy1]) - \
			rot_mat_inv@col_angle_mat@(np.array([vx1pp, vy1pp])*param1) + \
			rot_mat_inv@col_angle_mat@(np.array([vx1pp, vy1pp])*param2)
			vx2new, vy2new = np.array([vx2, vy2]) + rot_mat_inv@col_angle_mat@np.array([vx1pp, vy1pp])*param1
         
            #enenew = m1*(vx1new**2+vy1new**2) + m2*(vx2new**2+vy2new**2)
            
#             if np.round(enenew,4) != np.round(eneold,4):
#                 print("Ball hit ball problem!")
#                 col_angle_mat = np.array([[cos(alpha)**2, -sin(alpha)*cos(alpha)],[-sin(alpha)*cos(alpha), sin(alpha)**2]])
#                 vx1new, vy1new = np.array([vx1, vy1]) - \
#                 rot_mat_inv@col_angle_mat@(np.array([vx1pp, vy1pp])*(2*m1/(m1+m2))) + \
#                 rot_mat_inv@col_angle_mat@(np.array([vx1pp, vy1pp])*(2*(m1-m2)/(m1+m2)))
#                 vx2new, vy2new = np.array([vx2, vy2]) + rot_mat_inv@col_angle_mat@np.array([vx1pp, vy1pp])*(2*m1/(m1+m2))
#                 enenew = m1*(vx1new**2+vy1new**2) + m2*(vx2new**2+vy2new**2)
#                 if np.round(enenew,4) != np.round(eneold,4):
#                     print("Still not correct!")
#                     print([x1, y1, r1, m1, vx1, vy1, x2, y2, r2, m2, vx2, vy2])
                    
			balls[obj1] = x1new, y1new, r1, m1, vx1new, vy1new
			balls[obj2] = x2new, y2new, r2, m2, vx2new, vy2new

			for i in range(len(balls)):
				if i!=obj1 and i!=obj2:
					x1, y1, r1, m1, vx1, vy1 = balls[i]
					x1 += vx1*dt
					y1 += vy1*dt
					balls[i] = x1, y1, r1, m1, vx1, vy1
                
def generate_ball(exist_balls, geometry):
	width, height = geometry
	flag = 1
	while flag == 1:
		r = np.random.rand()*10+15
		x, y = np.random.rand(2)*[width-2*r,height-2*r]+[r,r]
		m = np.random.rand()*10+15
		vx, vy = np.random.rand(2)*[10,10] - 5
		ball = [x, y, r, m, vx, vy]
		if outbound(ball,geometry):
			continue
		flag = 0
		for exist_ball in exist_balls:
			if overlap(x-exist_ball[0], y-exist_ball[1], r+exist_ball[2]):
				flag = 1
				break
	return ball