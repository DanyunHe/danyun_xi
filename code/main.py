import os
import torch
import numpy as np
from itertools import combinations
from TrainModels import ball_ball_update_net, ball_ball_update_opt_net, ball_wall_detect_net, ball_ball_detect_net
from DataGenerator import true_data
from copy import deepcopy
from tqdm import tqdm

def load_models():

	model_bb_update=ball_ball_update_net()
	model_bb_update.load_state_dict(torch.load('./saved_models/model_bb_update'))

	model_bb_opt_update=ball_ball_update_opt_net()
	model_bb_opt_update.load_state_dict(torch.load('./saved_models/model_bb_opt_update'))

	model_bb_detect=ball_ball_detect_net()
	model_bb_detect.load_state_dict(torch.load('./saved_models/model_bb_detect'))

	model_bw_detect=ball_wall_detect_net()
	model_bw_detect.load_state_dict(torch.load('./saved_models/model_bw_detect'))

	return model_bb_update,model_bb_opt_update,model_bb_detect,model_bw_detect

def prop(state,dt=1,width = 300):
    return state@np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],\
    	[0,0,0,1,0,0],[dt,0,0,0,1,0],[0,dt,0,0,0,1]])
def lwall(state,dt=1,width = 300):
    return state@np.array([[-1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],\
    	[0,0,0,1,0,0],[-dt,0,0,0,-1,0],[0,dt,0,0,0,1]])+np.array([2*state[2],0,0,0,0,0])
def rwall(state,dt=1,width = 300):
    return state@np.array([[-1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],\
    	[0,0,0,1,0,0],[-dt,0,0,0,-1,0],[0,dt,0,0,0,1]])+np.array([2*width - 2*state[2],0,0,0,0,0])
def dwall(state,dt=1,width = 300):
    return state@np.array([[1,0,0,0,0,0],[0,-1,0,0,0,0],[0,0,1,0,0,0],\
    	[0,0,0,1,0,0],[dt,0,0,0,1,0],[0,-dt,0,0,0,-1]])+np.array([0,2*state[2],0,0,0,0])
def uwall(state,dt=1,width = 300):
    return state@np.array([[1,0,0,0,0,0],[0,-1,0,0,0,0],[0,0,1,0,0,0],\
    	[0,0,0,1,0,0],[dt,0,0,0,1,0],[0,-dt,0,0,0,-1]])+np.array([0,2*width - 2*state[2],0,0,0,0])

def bw_update(cur, ind,dt=1,width = 300):
    if ind == 0:
        return lwall(cur,dt=1,width = 300)
    elif ind == 1:
        return rwall(cur,dt=1,width = 300)
    elif ind == 2:
        return dwall(cur,dt=1,width = 300)
    elif ind == 3:
        return uwall(cur,dt=1,width = 300)
    else:
        return prop(cur,dt=1,width = 300)

def main(N_ball=5,N_sample=1000,width = 300,height = 300):
	model_bb_update,model_bb_opt_update,model_bb_detect,model_bw_detect=load_models()
	model_bb_update.eval()
	model_bb_opt_update.eval()
	model_bb_detect.eval()
	model_bw_detect.eval()

	print('models loaded')
	real_data=true_data()
	np.savetxt('./data_results/test1', real_data)

	# prediction 
	balls = real_data[0].reshape(N_ball, 1, 6)
	result_wE = []
	result_wE.append(deepcopy(balls))
	for i in tqdm(range(N_sample)): ## 5000 time steps
	    for j,k in combinations(range(N_ball),2):
	        hit = np.array([])
	        x1, y1, r1, m1, vx1, vy1 = balls[j,0]
	        x2, y2, r2, m2, vx2, vy2 = balls[k,0]
	        x1p = (x1 - x2)/r2
	        y1p = (y1 - y2)/r2
	        r1p = r1/r2
	        m1p = m1/m2
	        vx1p = vx1/r2
	        vx2p = vx2/r2
	        vy1p = vy1/r2
	        vy2p = vy2/r2
	       	
	       	bb_detect=model_bb_detect(torch.tensor([[x1p, y1p, r1p, vx1p, vy1p]])).detach().numpy()
	        if  bb_detect > 0.6:
	            hit = np.array([j,k])
	            
	            bb_update=model_bb_opt_update(torch.tensor([[x1p, y1p, r1p, m1p, vx1p, vx2p, vy1p, vy2p]])).detach().numpy()
	       
	            x1p, x2p, y1p, y2p, vx1p, vx2p, vy1p, vy2p = bb_update[0]
	            x1 = x1p*r2 + x2
	            y1 = y1p*r2 + y2
	            x2 = x2p*r2 + x2
	            y2 = y2p*r2 + y2
	            vx1 = vx1p*r2
	            vy1 = vy1p*r2
	            vx2 = vx2p*r2
	            vy2 = vy2p*r2
	            balls[j] = np.array([[x1, y1, r1, m1, vx1, vy1]])
	            balls[k] = np.array([[x2, y2, r2, m2, vx2, vy2]])
	            break
	    for l in np.setdiff1d(np.arange(N_ball),hit):
	    	bw_detect=model_bw_detect(torch.tensor(balls[l][0][[0,1,2,4,5]]/width).unsqueeze(0).float())
	    	bw_detect=bw_detect.detach().numpy()
	    	ind = np.argmax(bw_detect[0])
	    	balls[l] = np.expand_dims(bw_update(balls[l][0], ind,dt=1,width = 300), axis=0)
	    result_wE.append(deepcopy(balls))
	result_wE = np.array(result_wE).reshape(N_sample+1,N_ball*6)
	np.savetxt('./data_results/test1_wE', result_wE)


if __name__ == '__main__':
	main()






	