import os
from itertools import combinations

def load_models():

	json_file = open('./models/model_ball_ball_1.json',"r")
	model_json = json_file.read()
	model_bb_update = model_from_json(model_json)
	model_bb_update.load_weights("./models/model_ball_ball_1.h5")

	json_file = open('./models/model1_woEcon.json',"r")
	model_json = json_file.read()
	model_bb_update_woE = model_from_json(model_json)
	model_bb_update_woE.load_weights("./models/model1_woEcon.h5")

	json_file = open('./models/model_bb_pred.json',"r")
	model_json = json_file.read()
	model_bb_pred = model_from_json(model_json)
	model_bb_pred.load_weights("./models/model_bb_pred.h5")

	json_file = open('./models/model_handler.json',"r")
	model_json = json_file.read()
	model_bw = model_from_json(model_json)
	model_bw.load_weights("./models/model_handler.h5")

	return model_bb_update,model_bb_update_woE,model_bb_pred,model_bw

def prop(state):
    return state@np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],\
    	[0,0,0,1,0,0],[dt,0,0,0,1,0],[0,dt,0,0,0,1]])
def lwall(state):
    return state@np.array([[-1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],\
    	[0,0,0,1,0,0],[-dt,0,0,0,-1,0],[0,dt,0,0,0,1]])+np.array([2*state[2],0,0,0,0,0])
def rwall(state):
    return state@np.array([[-1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],\
    	[0,0,0,1,0,0],[-dt,0,0,0,-1,0],[0,dt,0,0,0,1]])+np.array([2*width - 2*state[2],0,0,0,0,0])
def dwall(state):
    return state@np.array([[1,0,0,0,0,0],[0,-1,0,0,0,0],[0,0,1,0,0,0],\
    	[0,0,0,1,0,0],[dt,0,0,0,1,0],[0,-dt,0,0,0,-1]])+np.array([0,2*state[2],0,0,0,0])
def uwall(state):
    return state@np.array([[1,0,0,0,0,0],[0,-1,0,0,0,0],[0,0,1,0,0,0],\
    	[0,0,0,1,0,0],[dt,0,0,0,1,0],[0,-dt,0,0,0,-1]])+np.array([0,2*width - 2*state[2],0,0,0,0])

def bw_update(cur, ind):
    if ind == 0:
        return lwall(cur)
    elif ind == 1:
        return rwall(cur)
    elif ind == 2:
        return dwall(cur)
    elif ind == 3:
        return uwall(cur)
    else:
        return prop(cur)


