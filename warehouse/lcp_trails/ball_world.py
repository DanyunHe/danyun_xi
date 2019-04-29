from ba import Ball
import pygame
from pygame import Color, Rect
import sys
from math import fabs
import numpy as np
import math
import random
import pandas as pd

class BallWorld(object):
	SCREEN_WIDTH, SCREEN_HEIGHT = 300, 300
	def __init__(self,x1=30,y1=30,x2=70,y2=70,x3=50,y3=50,x4=150, y4=150,\
		speed_x1=1000,speed_y1=0,speed_x2=-1000,speed_y2=0,speed_x3=0,speed_y3=-1000,speed_x4=1000, speed_y4=1000,\
		left=0, top=0, width=100, height=100, verbose = 0):
		pygame.init()
		self.verbose = verbose
		self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
		self.clock = pygame.time.Clock()
		self.balls = []
		# initialize Ball(x,y,speed_x,speed_y,r,color,name)
		self.balls.append(Ball(x1, y1, speed_x1,speed_y1, 5, Color('yellow'), 'yellow', self.verbose))
		#self.balls.append(Ball(x2, y2,speed_x2,speed_y2,20, Color('red'),'red', self.verbose))
		#self.balls.append(Ball(x3,y3, speed_x3, speed_y3, 20, Color('blue'), 'blue', self.verbose))	
		#self.balls.append(Ball(x4,y4, speed_x4, speed_y4, 0, Color('green'), 'green', self.verbose))	
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
				for i in range(len(self.balls)):
					self.balls[i].move(t_remaining)
				break

		result_temp = []
		for i in range(len(self.balls)):
			result_temp += [self.balls[i].get_x(), self.balls[i].get_speed_x(), np.min((np.abs(self.balls[i].get_x()-self.border.x),
			np.abs(self.balls[i].get_x()-self.border.x-self.border.width))),
			self.balls[i].get_y(), self.balls[i].get_speed_y(), np.min((np.abs(self.balls[i].get_y()-self.border.y),
			np.abs(self.balls[i].get_y()-self.border.y-self.border.height)))]
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

	def draw(self):
		pygame.draw.rect(self.screen, Color("grey"), self.border)
		for b in self.balls:
			b.draw(self.screen)
	
	def quit(self):
		sys.exit()

	# N: number of episodes
	# T: number of steps
	# dt: size of time step 
	def run(self,N,T,dt, verbose = 0):
		# pygame.key.set_repeat(30, 30)
		# params=[] # data [t,num_balls,4], contain position and velocity information at all the time for all the balls
		input_params=pd.DataFrame([],columns=['x1','y1','vx1','vy1',\
			'x2','y2','vx2','vy2','x3','y3','vx3','vy3',\
			'x4','y4','vx4','vy4','x5','y5','vx5','vy5'])
		out_params=pd.DataFrame([],columns=['x1','y1','vx1','vy1',\
			'x2','y2','vx2','vy2','x3','y3','vx3','vy3',\
			'x4','y4','vx4','vy4','x5','y5','vx5','vy5'])
		
		
		for num in range(N):

			# reset the position and velocity of balls 
			speed_x=np.random.uniform(10,70)
			speed_y=np.random.uniform(10,70)
			self.balls[0]._set_speed(speed_x,speed_y)
			self.balls[0]._set_position(50,50)

			int_params_=[]
			out_params_=[]
			for t in range(T):
				self.update(dt)
				
				if t>=10 and t<=14:
					for b in self.balls:
						int_params_+=b.get_params()

				if t>=40 and t<=44:
					
					for b in self.balls:
						out_params_+=b.get_params()

				# self.draw()
				for e in pygame.event.get():
					#if e.type == pygame.KEYDOWN:
						#if e.key == pygame.K_RETURN:
							#self.update()
							#self.draw()
					if e.type == pygame.QUIT:
						self.quit()		
				# pygame.display.flip()	
			
			input_params.loc[num]=int_params_
			out_params.loc[num]=out_params_							
		return input_params,out_params

if __name__=="__main__":
	random.seed(77)
	# initial balls position, speed, and box size 
	bw = BallWorld(x1=50.1,y1=50.2,x2=70.1,y2=70.2,x3=100.1,y3=100.2, x4=150, y4=150,\
		speed_x1=1000,speed_y1=303.5,speed_x2=-1000,speed_y2=1,speed_x3=1,speed_y3=-1000, speed_x4=1000, speed_y4=1000,\
		left=0, top=0, width=100, height=100, verbose = 0)
	verbose = 0
	result = []
	input_params,out_params=bw.run(10000,50,0.05) # generate data 
	print('done, saving data')
	input_params.to_csv('./data/4w_input_params.csv', encoding='utf-8', index=False)
	out_params.to_csv('./data/4w_out_params.csv', encoding='utf-8', index=False)

	if verbose:
		print(params[0])
		print(np.array(params).shape)
