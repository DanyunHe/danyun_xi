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
	SCREEN_WIDTH, SCREEN_HEIGHT = 500, 600
	def __init__(self,x1=30,y1=30,x2=70,y2=70,x3=50,y3=50,\
		speed_x1=1000,speed_y1=0,speed_x2=-1000,speed_y2=0,speed_x3=0,speed_y3=-1000,\
		left=0, top=0, width=100, height=100, verbose = 0):
		# pygame.init()
		self.verbose = verbose
		self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
		self.clock = pygame.time.Clock()
		self.balls = []
		# initialize Ball(x,y,speed_x,speed_y,r,color,name)
		self.balls.append(Ball(x1, y1, speed_x1,speed_y1, 20, Color('yellow'), 'yellow', self.verbose))
		self.balls.append(Ball(x2, y2,speed_x2,speed_y2,20, Color('red'),'red', self.verbose))
		self.balls.append(Ball(x3,y3, speed_x3, speed_y3, 20, Color('blue'), 'blue', self.verbose))	
		# initialize wall 	
		self.border = Rect(left, top, width, height)
		self.border.left=left
		self.border.top=top
		self.border.width=width
		self.border.height=height
		
	def update(self,dt):
	
		#check collision with other balls
		for i in range(len(self.balls)):
			for j in range(len(self.balls)):
				if i < j:
					self.balls[i].detect_collision_with_other_ball(self.balls[j],dt)

		#check collision with box border:
		for b in self.balls:
			b.detect_collision_with_box(self.border,dt)
		
		# move by time step dt
		ene = 0
		for b in self.balls:
			b.log('ball ')
			b.move(dt)
			ene += b.get_energy()
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
		input_params=pd.DataFrame([],columns=['x1','y1','vx1','vy1','x2','y2','vx2','vy2','x3','y3','vx3','vy3'])
		out_params=pd.DataFrame([],columns=['x1','y1','vx1','vy1','x2','y2','vx2','vy2','x3','y3','vx3','vy3'])
		
		for num in range(N):
			print('n=',num)

			# reset the position and velocity of balls 
			for i in range(3):
				angle=random.uniform(0,2*math.pi)
				speed_x=10*math.cos(angle)
				speed_y=10*math.sin(angle)
				self.balls[i]._set_speed(speed_x,speed_y)

				# r=self.balls[i].r
				# x=random.uniform(self.border.left+r,self.border.width-self.border.left-r)
				# y=random.uniform(self.border.top+r,self.border.height-self.border.top-r)
			self.balls[0]._set_position(40,40)
			self.balls[1]._set_position(300,200)
			self.balls[2]._set_position(200,300)

			

			for t in range(T):
				self.update(dt)
				if t==10:
					params=[]
					for b in self.balls:
						params+=b.get_params()

					input_params.loc[num]=params

				if t==40:
					params=[]
					for b in self.balls:
						params+=b.get_params()

					out_params.loc[num]=params

				# self.draw()
				# for e in pygame.event.get():
					#if e.type == pygame.KEYDOWN:
						#if e.key == pygame.K_RETURN:
							#self.update()
							#self.draw()
					# if e.type == pygame.QUIT:
						# self.quit()		
				# pygame.display.flip()								
		return input_params,out_params

if __name__=="__main__":
	random.seed(77)
	# initial balls position, speed, and box size 
	bw = BallWorld(x1=30,y1=30,x2=70,y2=70,x3=90,y3=50,\
		speed_x1=1000,speed_y1=300,speed_x2=-1000,speed_y2=0,speed_x3=0,speed_y3=-1000,\
		left=0, top=0, width=400, height=400, verbose = 0)

	input_params,out_params=bw.run(10000,50,1) # generate data 
	
	print('done, saving data')

	# input_params.to_pickle('./data/input_params.pkl')    #to save the dataframe, df to 123.pkl
	# out_params.to_pickle('./data/out_params.pkl')

	input_params.to_csv('./data/input_params.csv', encoding='utf-8', index=False)
	out_params.to_csv('./data/out_params.csv', encoding='utf-8', index=False)

