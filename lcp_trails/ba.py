import math
from pygame import Rect
import pygame
import numpy as np
from numpy import arctan as arctan
from numpy import cos as cos
from numpy import sin as sin

  
 
class Ball(object):

    def _set_speed(self, speed_x, speed_y):
        self.speed_x = speed_x
        self.speed_y = speed_y

    def _set_position(self,x,y):
        self.x=x
        self.y=y
    
    def __init__(self, x, y, speed_x,speed_y,r, color, name, verbose):
        self.name = name
        self.x = x
        self.y = y
        self._set_speed(speed_x,speed_y)
        self.r = r
        self.color = color
        self.mass=r**3
        self.verbose = verbose

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_speed_x(self):
        return self.speed_x

    def get_speed_y(self):
        return self.speed_y

    def get_new_x(self,dt):
        return self.x+self.speed_x*dt

    def get_new_y(self,dt):
        return self.y+self.speed_y*dt

    def get_params(self):
        return [self.x,self.y,self.speed_x,self.speed_y]

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.r)
    
    def move(self,dt):
        self.x=self.get_new_x(dt)
        self.y=self.get_new_y(dt)

    def get_energy(self):
        return self.speed_x**2+self.speed_y**2
        


    # detect collision and calculate new speed and position, change speed if collision happens 
    # tiemStep: dt  
    def detect_collision_with_other_ball(self, o,dt):

        # if (self.x-o.x)**2+(self.y-o.y)**2>(self.r+o.r)**2:
        #     self.status = 'overlap'

        new_x=self.get_new_x(dt)
        new_y=self.get_new_y(dt)
        o_new_x=o.get_new_x(dt)
        o_new_y=o.get_new_y(dt)


        ## if the new position is overlapping and the current position not, or both are overlapping 
        if ((new_x-o_new_x)**2+(new_y-o_new_y)**2<=(self.r+o.r)**2 and (self.x-o.x)**2+(self.y-o.y)**2>(self.r+o.r)**2) or \
        ((new_x-o_new_x)**2+(new_y-o_new_y)**2<=(self.r+o.r)**2 and (self.x-o.x)**2+(self.y-o.y)**2<=(self.r+o.r)**2 and \
            (new_x-o_new_x)**2+(new_y-o_new_y)**2<(self.x-o.x)**2+(self.y-o.y)**2):

        # if (new_x-o_new_x)**2+(new_y-o_new_y)**2<=(self.r+o.r)**2:
            if self.verbose:
                print("collision happens")
            mass=self.mass
            o_mass=o.mass
            m1 = mass
            m2 = o_mass
            if self.verbose:
                print('old speed',self.speed_x,self.speed_y)

            normal_tan = -(self.x - o.x)/(self.y - o.y)
            normal_angle = arctan(normal_tan)
            theta = normal_angle

            v1x = self.speed_x
            v1y = self.speed_y
            v2x = o.speed_x
            v2y = o.speed_y

            v1t = v1x*cos(theta) + v1y*sin(theta)
            v1n = v1x*sin(theta) - v1y*cos(theta)
            v2t = v2x*cos(theta) + v2y*sin(theta)
            v2n = v2x*sin(theta) - v2y*cos(theta)

            # v1n, v2n = ((m1-m2)/(m1+m2)*v1n+(2*m2)/(m1+m2)*v2n,
            #     (2*m1)/(m1+m2)*v1n+(m2-m1)/(m1+m2)*v2n)

            v1n, v2n = v2n, v1n
            self.speed_x = v1t*cos(theta) + v1n*sin(theta)
            self.speed_y = v1t*sin(theta) - v1n*cos(theta)

            o.speed_x = v2t*cos(theta) + v2n*sin(theta)
            o.speed_y = v2t*sin(theta) - v2n*cos(theta)

            # self.speed_x = (2*o_mass*o.speed_x+(mass-o_mass)*self.speed_x)/(mass+o_mass)
            # self.speed_y = (2*o_mass*o.speed_y+(mass-o_mass)*self.speed_y)/(mass+o_mass)
            # print('new speed p, q', self.speed_x, self.speed_y)
            # o.speed_x = (2*mass*self.speed_x+(o_mass-mass)*o.speed_x)/(mass+o_mass)
            # o.speed_y= (2*mass*self.speed_y+(o_mass-mass)*o.speed_y)/(mass+o_mass)
  

     # detect collition with wall, change speed if collision happens   
    def detect_collision_with_box(self, box, dt):
        #check vertical collision:
        self.detect_collision_with_vertical_line(box.x, dt)
        self.detect_collision_with_vertical_line(box.x + box.width, dt)
        self.detect_collision_with_horizontal_line(box.y, dt)
        self.detect_collision_with_horizontal_line(box.y + box.height, dt)
       
    # detect collition with vertical line, change speed if collision happens 
    # x: vertical line position 
    # dt: move by time step dt 
    def detect_collision_with_vertical_line(self, x,dt):
        new_x=self.get_new_x(dt)
        if abs(new_x-x)<=self.r:
            if self.verbose:
                print('collision with vertical line')
            self.speed_x=-self.speed_x

            
    def detect_collision_with_horizontal_line(self, y, dt):
        new_y=self.get_new_y(dt)
        if abs(new_y-y)<=self.r:
            if self.verbose:
                print('collision with horizontal line')
            self.speed_y=-self.speed_y

        
    
    def log(self, description):
        if self.verbose:
            print(description, self.name, self.x, self.y, self.speed_x, self.speed_y)
    
   
# COLLISION TESTING
if __name__=="__main__":
    #ball goes against the wall
    b3 = Ball(20, 40, 3, 0, 5, None)
    box = Rect(0, 0, 400, 400)
    b3.detect_collision_with_box(box,0.001)
    #print 'collision time', b3.collision_response.t
    #print 'b3 new speed', b3.collision_response.new_speed_x, b3.collision_response.new_speed_y
   
    #print 'b3 x, y', b3.x, b3.y
    b1 = Ball(300, 30, 5, -30, 15, None)
    box2 = Rect(5, 5, 490, 590)
    print('before collision', b1.speed_x, b1.speed_y)
    b1.detect_collision_with_box(box2, 1000)
   
    print('b1 after update',  b1.x, b1.y, b1.speed_x, b1.speed_y)
    b1.detect_collision_with_box(box2, 2)
  
    b1.log('before moving')

    b1.move(1)
    b1.log('after moving')
        
    