from scipy import optimize
import numpy as np
from numpy import arctan, cos, sin, sign, pi, arcsin, arccos
from numpy.linalg import norm
from itertools import combinations
import os
from tqdm import tqdm
import random
import RealTraj as rj 

random.seed(7)

def update(x1, x2, y1, y2, r1, r2, m1, m2, vx1, vx2, vy1, vy2, dt):
    xc, yc = collision_pt(x1, x2, y1, y2, r1, r2, vx1, vx2, vy1, vy2) # collision point in the frame of ball 2
    vx1_rf = vx1-vx2
    vy1_rf = vy1-vy2
    dt_c = (xc-x1)/vx1_rf # collision time
    #print("1st dtc",dt_c)
    if dt_c < 0 or dt_c > 1:
        print("something wrong")
    alpha = arctan((y2-yc)/(x2-xc))
    
    #print("alpha",alpha)
    
    vn1 = vx1_rf*cos(alpha)+vy1_rf*sin(alpha)
    vn2 = 0
    vt1 = vx1_rf*sin(alpha)-vy1_rf*cos(alpha)
    vt2 = 0
    
    
    vn1, vn2 = ((m1-m2)/(m1+m2)*vn1+(2*m2)/(m1+m2)*vn2,
                (2*m1)/(m1+m2)*vn1+(m2-m1)/(m1+m2)*vn2)

    vx1_rf = vt1*sin(alpha)+vn1*cos(alpha)
    vy1_rf = -vt1*cos(alpha)+vn1*sin(alpha)

    vx2_rf = vt2*sin(alpha)+vn2*cos(alpha)
    vy2_rf = -vt2*cos(alpha)+vn2*sin(alpha)

    #print(vx2_rf, vy2_rf)
    
    x1 = xc + vx1_rf*(dt-dt_c) + vx2*dt
    y1 = yc + vy1_rf*(dt-dt_c) + vy2*dt
    x2 = x2 + vx2_rf*(dt-dt_c) + vx2*dt
    y2 = y2 + vy2_rf*(dt-dt_c) + vy2*dt

    vx1 = vx1_rf + vx2
    vy1 = vy1_rf + vy2
    vx2 = vx2_rf + vx2
    vy2 = vy2_rf + vy2
    
    return x1, x2, y1, y2, r1, r2, m1, m2, vx1, vx2, vy1, vy2

def collision_pt(x1, x2, y1, y2, r1, r2, vx1, vx2, vy1, vy2):
    vx1_rf = vx1-vx2
    vy1_rf = vy1-vy2
    yc = np.roots([1+(vx1_rf/vy1_rf)**2, 
                     -2*(vx1_rf/vy1_rf)**2*y1+2*(vx1_rf/vy1_rf)*(x1-x2)-2*y2,
                    (vx1_rf/vy1_rf)**2*y1**2+(x1-x2)**2-2*(vx1_rf/vy1_rf)*(x1-x2)*y1+y2**2-(r1+r2)**2])
    xc = (vx1_rf/vy1_rf)*(yc-y1)+x1
    
    if (xc[0]-x1)**2+(yc[0]-y1)**2 > (xc[1]-x1)**2+(yc[1]-y1)**2:
        return xc[1], yc[1]
    
    if (xc[0]-x1)**2+(yc[0]-y1)**2 < (xc[1]-x1)**2+(yc[1]-y1)**2:
        return xc[0], yc[0]
    else:
        return -1

# generate data of ball-ball collision 
# initials: state before collision: (x1, x2, y1, y2, r1, r2, m1, m2, vx1, vx2, vy1, vy2) at time t
# finals: state after collision: (x1, x2, y1, y2, r1, r2, m1, m2, vx1, vx2, vy1, vy2) at time t+1
def ball_ball_update_data(n_sample=100000,dt=1):
	initials = []
	finals = []
	collision_points = []
	count=0
	while count<n_sample:
	    r1 = np.random.rand()*1.5+0.5
	    r2 = 1
	    m1 = np.random.rand()*1.5+0.5
	    m2 = 1
	#     m2 = 1
	    x1 = np.random.rand()*6-3
	    y1 = np.random.rand()*6-3
	    x2 = 0*np.random.rand()*4
	    y2 = 0*np.random.rand()*4
	    if (x1-x2)**2+(y1-y2)**2 > (r1+r2)**2: # initially not touching
	        vx1 = np.random.rand()*2-1
	        vy1 = np.random.rand()*2-1
	        vx2 = np.random.rand()*4-2
	        vy2 = np.random.rand()*4-2
	        if (x1+(vx1-vx2)*dt-x2)**2+(y1+(vy1-vy2)*dt-y2)**2 < (r1+r2)**2:
	            # initials.append([x1, x2, y1, y2, r1, r2, m1, m2, vx1, vx2, vy1, vy2])
	            initials.append([x1, y1, r1, m1,vx1, vx2, vy1, vy2])
	            x1, x2, y1, y2, r1, r2, m1, m2, vx1, vx2, vy1, vy2 = update(x1, x2, y1, y2, r1, r2, m1, m2, vx1, vx2, vy1, vy2, dt)
	            # finals.append([x1, x2, y1, y2, r1, r2, m1, m2, vx1, vx2, vy1, vy2])
	            finals.append([x1, y1, r1, m1,vx1, vx2, vy1, vy2])
	            count+=1
	        else:
	            continue
	    else:
	        continue
	initials = np.array(initials)
	finals = np.array(finals)

	print('ball ball velocity update data generated!')
	return initials,finals 


# generate data of ball-wall collision detection: 5*n_sample data, ensure each case appears n_sample times
# initials: state before collision: (x, y, r, vx, vy)
# finals: collision state by one-hot  
# hit -x: [1,0,0,0,0]
# hit  x: [0,1,0,0,0]
# hit -y: [0,0,1,0,0]
# hit  y: [0,0,0,1,0]
# no hit: [0,0,0,0,1]
def ball_wall_detect_data(dt=1,width=1,n_sample=10000):
	result_handler = []
	count = np.array([0,0,0,0,0])

	while np.any(count<n_sample):
	    r = np.random.rand()*0.09+0.01 # r should be 1/10 of the width
	    x = np.random.rand()*(width-2*r)+r # between r and width - r
	    y = np.random.rand()*(width-2*r)+r # between r and width - r
	    vx = np.random.rand()*0.2-0.1 # an -5 to 5 velocity
	    vy = np.random.rand()*0.2-0.1 # an -5 to 5 velocity
	    if x+vx*dt<r:
	        if (y+vy*dt>r and y+vy*dt < width - r): # hit -x
		        if count[0]<n_sample:
		            result_handler.append([x, y, r, vx, vy, 1,0,0,0,0])
		            count[0] += 1
	    elif x+vx*dt>width-r:
	        if (y+vy*dt>r and y+vy*dt < width - r): # hit +x
		        if count[1]<n_sample:
		            result_handler.append([x, y, r, vx, vy, 0,1,0,0,0])
		            count[1] += 1
	    elif y+vy*dt<r:
	        if (x+vx*dt>r and x+vx*dt < width - r): # hit -y
		        if count[2]<n_sample:
		        	result_handler.append([x, y, r, vx, vy, 0,0,1,0,0])
		        	count[2] += 1
	    elif y+vy*dt>width-r:
	        if (x+vx*dt>r and x+vx*dt < width - r): # hit +y
		        if count[3]<n_sample:
		            result_handler.append([x, y, r, vx, vy, 0,0,0,1,0])
		            count[3] += 1
	    else:
	        if count[4]<n_sample:
		        result_handler.append([x, y, r, vx, vy, 0,0,0,0,1])
		        count[4] += 1
	result_handler = np.array(result_handler)
	np.random.shuffle(result_handler)
	initials = result_handler.T[0:5].T
	finals = result_handler.T[5:].T

	print('ball wall detect data generated!')
	return initials, finals

# generate data of ball-ball collision detection: 2*n_sample data, ensure each case appears n_sample times 
# fix ball2: (x2, y2, r2, vx2, vy2) = (0,0,1,0,0)
# initials: state before collision of ball1 - (x1, y1, r1, vx1, vy1)
# finals: collision state (1: collision, 0: no collision)
def ball_ball_detect_data(dt=1,width=1,n_sample = 100000):
	results = []
	num_hit = 0
	num_nohit = 0
	
	while num_hit < n_sample or num_nohit < n_sample:
	  
	    r1 = np.random.rand()*1.5+0.5
	    r2 = 1
	    m1 = np.random.rand()*1.5+0.5
	    m2 = 1
	#     m2 = 1
	    x1 = np.random.rand()*12-6
	    y1 = np.random.rand()*12-6
	    x2 = 0*(np.random.rand()*50-25)
	    y2 = 0*(np.random.rand()*50-25)
	    if (x1-x2)**2+(y1-y2)**2 > (r1+r2)**2: # initially not touching
	        vx1 = np.random.rand()*2-1
	        vy1 = np.random.rand()*2-1
	        vx2 = (np.random.rand()*4-2)*0
	        vy2 = (np.random.rand()*4-2)*0
	        
	        
	        if (x1+(vx1-vx2)*dt-x2)**2+(y1+(vy1-vy2)*dt-y2)**2 <= (r1+r2)**2 and num_hit < n_sample: #hit
	            results.append([x1, y1, r1, vx1, vy1, 1])
	            num_hit += 1
	        if (x1+(vx1-vx2)*dt-x2)**2+(y1+(vy1-vy2)*dt-y2)**2 > (r1+r2)**2 and num_nohit < n_sample:
	            results.append([x1, y1, r1, vx1, vy1, 0])
	            num_nohit += 1
	    else:
	        continue
	results = np.array(results)
	np.random.shuffle(results)
	initials=results[:,:5]
	finals=results[:,-1].reshape(-1,1)
	print('ball ball detect data generated!')

	return initials,finals

# generate true trajectoris of balls 
def true_data(N_ball = 5,width = 300,height = 300,length = 1000,dt = 1):
	geometry = [width, height]
	ovlap = True
	count = 1
	while ovlap == True:
	    print("attempt: ", count)
	    count += 1
	    ovlap = False
	    balls = []
	    for i in range(N_ball):
	        balls.append(rj.generate_ball(balls, geometry))
	    results = []
	    results.append(np.array(balls))
	
	    for i in range(length):
	        dt_c, ball_1_index, item_2_index = rj.event_handler(balls,geometry, dt=dt)
	        rj.update(balls,ball_1_index, item_2_index, dt_c, geometry, dt = dt)
	        if rj.check_overlap(balls, i) == True:
	            ovlap = True
	        results.append(np.array(balls))
	    data = np.array(results).reshape(length+1,6*len(balls))
	    filename = 'test1'
	    np.savetxt('./data_results/'+filename, data)
	    params = [str(width),str(height)]
	    with open('./data_results/params_'+filename,'w') as file:
	        for param in params:
	            file.writelines(param+'\n')
	print("No overlap data generated!")
	return data 


# test
if __name__ == '__main__':

	"""
	initials,finals=ball_ball_update_data(n_sample=100,dt=1)
	print(initials.shape,finals.shape)
	initials,finals=ball_wall_detect_data(dt=1,width=1,n_sample=100)
	print(initials.shape,finals.shape)
	initials,finals=ball_ball_detect_data(dt=1,width=1,n_sample = 100)
	print(initials.shape,finals.shape)
	data=true_data()
	print(len(data))
	"""
	




