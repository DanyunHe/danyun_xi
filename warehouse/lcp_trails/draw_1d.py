from ba_1d import Ball
import pygame
from pygame import Color, Rect
import sys
from math import fabs
import numpy as np
import math
import random

class BallWorld(object):
	SCREEN_WIDTH, SCREEN_HEIGHT = 300, 300
	def __init__(self,x1=30,y1=30,x2=70,y2=70,x3=150,y3=50,x4=150, y4=150,\
		speed_x1=1000,speed_y1=0,speed_x2=-500,speed_y2=0,speed_x3=300,speed_y3=-1000,speed_x4=1000, speed_y4=1000,\
		left=0, top=0, width=100, height=100, verbose = 0):
		pygame.init()
		self.verbose = verbose
		self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
		self.clock = pygame.time.Clock()
		self.balls = []
		# initialize Ball(x,y,speed_x,speed_y,r,color,name)
		self.balls.append(Ball(x1, speed_x1, 20, Color('yellow'), 'yellow', self.verbose))
		self.balls.append(Ball(x2, speed_x2, 20, Color('red'),'red', self.verbose))
		# self.balls.append(Ball(x3, speed_x3, 30, Color('blue'), 'blue', self.verbose))	
		# self.balls.append(Ball(x4, speed_x4, 10, Color('green'), 'green', self.verbose))	
		# initialize wall 	
		self.border = Rect(left, top, width, height)
		self.border.left=left
		self.border.top=top
		self.border.width=width
		self.border.height=height
	

	def update(self,dt):
		t_remaining = dt

		ball_hit_ball = np.array([0,0])
		ball_hit_wall = np.array([0])

		while True:
			ball_col_time_min = dt
			wall_col_time_min = dt
			for i in range(len(self.balls)):
				for j in range(i+1,len(self.balls)):
					ball_col_time = self.balls[i].get_ball_col_time(self.balls[j])
					if ball_col_time < ball_col_time_min and ball_col_time>0:
						ball_col_time_min = ball_col_time 
						ball_hit_ball[0] = i
						ball_hit_ball[1] = j
				wall_col_time, index = self.balls[i].get_wall_col_time(self.border) # the wall collision time is the minimum of 4
				if wall_col_time < wall_col_time_min and wall_col_time>0:
					wall_col_time_min = wall_col_time 
					ball_hit_wall = i
					wall_index = index

			if np.min((wall_col_time_min, ball_col_time_min)) < t_remaining:

				## if there is still time for events
				if wall_col_time_min < ball_col_time_min:
					self.balls[ball_hit_wall].move_wall(wall_col_time_min, wall_index)
					for i in np.setdiff1d(np.arange(len(self.balls)),np.array([ball_hit_wall])):
						self.balls[i].move(wall_col_time_min)
					t_remaining -= wall_col_time_min
							 # update is done by calling move

				else:
					self.balls[ball_hit_ball[0]].move_ball(self.balls[ball_hit_ball[1]], ball_col_time_min)
					for i in np.setdiff1d(np.arange(len(self.balls)),np.array([ball_hit_ball])):
						self.balls[i].move(ball_col_time_min)
					t_remaining -= ball_col_time_min


			else:

				## no more collision is going to happen. Just normally move 't_remaining'
				## This block has to call at least once.

				for i in range(len(self.balls)):
					self.balls[i].move(t_remaining)
				break

		result_temp = []
		for i in range(len(self.balls)):
			result_temp += [self.balls[i].get_x(), self.balls[i].get_speed_x(),np.min((np.abs(self.balls[i].get_x()-self.border.x),
			np.abs(self.balls[i].get_x()-self.border.x-self.border.width)))]
		result.append(result_temp)




			## keep a track of what is happening within dt
			## possible outcome:(1) hit the wall (2) hit one other ball 
			## if hit one wall before hitting anything else


		# #check collision with other balls
		# for i in range(len(self.balls)):
		# 	for j in range(len(self.balls)):
		# 		if i < j:
		# 			self.balls[i].collision_with_other_ball(self.balls[j],dt)

		# #check collision with box border:
		# for b in self.balls:
		# 	b.collision_with_box(self.border,dt)
		
		# # move by time step dt
		# ene = 0
		# for b in self.balls:
		# 	b.log('ball ')
		# 	b.move(dt)
		# 	ene += b.get_energy()
		# print("Total energy:", ene)
			

	def log(self, ball, description):
		if self.verbose:
			print(description, 'x', ball.x, 'y', ball.y)

	def draw(self,i):
		pygame.draw.rect(self.screen, Color("grey"), self.border)
		pygame.draw.circle(self.screen, Color("yellow"), (int(xs1[i]), int(xs2[i])), 20)
	
	def quit(self):
		sys.exit()

	# N: number of episodes
	# T: number of steps
	# dt: size of time step 
	def run(self,N,T,dt, verbose = 0):
		pygame.key.set_repeat(30, 30)
		params=[] # data [t,num_balls,4], contain position and velocity information at all the time for all the balls
		for _ in range(N):

			# reset the position and velocity of balls 
			# for i in range(2):
			# 	angle=random.uniform(0,2*math.pi)
			# 	speed_x=10*math.cos(angle)
			# 	speed_y=10*math.sin(angle)
			# 	self.balls[i]._set_speed(speed_x,speed_y)
			# 	x=random.uniform(self.border.left,self.border.width-self.border.left)
			# 	y=random.uniform(self.border.top,self.border.height-self.border.top)
			# 	self.balls[i]._set_position(x,y)

			for i in range(T):
				self.update(dt)
				for b in self.balls:
					params.append(b.get_params())

				self.draw(i)
				for e in pygame.event.get():
					#if e.type == pygame.KEYDOWN:
						#if e.key == pygame.K_RETURN:
							#self.update()
							#self.draw()
					if e.type == pygame.QUIT:
						self.quit()		
				pygame.display.flip()								
		return params

if __name__=="__main__":
	verbose = 0
	random.seed(77)
	# initial balls position, speed, and box size 
	bw = BallWorld(x1=30.1,y1=30.2,x2=70.1,y2=70.2,x3=250.1,y3=100.2, x4=150, y4=150,\
		speed_x1=500,speed_y1=300,speed_x2=-200,speed_y2=1,speed_x3=300,speed_y3=-1000, speed_x4=1000, speed_y4=1000,\
		left=0, top=0, width=300, height=300, verbose = 0)
	result = []
	data = np.loadtxt('./own_movie/output_ene_con1')
	xs1 = data.T[0]
	xs2 = data.T[3]
	params=bw.run(1,3000,0.01) # generate data 
	if verbose:
		print(params[0])
		print(np.array(params).shape)
