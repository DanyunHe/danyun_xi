import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

data_we=np.loadtxt('./data_results/test1_wE')
# data_woe=np.loadtxt('./all_results/test1_woE')
data_true=np.loadtxt('./data_results/test1')


def patches(data,n_ball=5):  
    patches = []
    for j in range(n_ball):
        x=data[0+6*j]
        y=data[1+6*j]
        r=data[2+6*j]
        circle = Circle((x, y), r)
        patches.append(circle)
    return patches

def main(n_frame=500):
	for i in range(n_frame):
	    fig, axes = plt.subplots(1,3,figsize=(10,5))
	    for k in range(3):
	        axes[k].axis('scaled')
	        axes[k].set_xlim(0,300)
	        axes[k].set_ylim(0,300)
	        
	    axes[0].set_title('predict, with energy')
	    axes[1].set_title('true')
	    axes[2].set_title('predict, without energy')
	    
	    patches1 = patches(data_we[i])
	    patches2=patches(data_true[i])
	    # patches3=patches(data_woe[i])
	    
	    p1 = PatchCollection(patches1, alpha=0.5)
	    p2 = PatchCollection(patches2, alpha=0.5)
	    # p3 = PatchCollection(patches3, alpha=0.5)
	    
	    axes[0].add_collection(p1)
	    axes[1].add_collection(p2)
	    # axes[2].add_collection(p3)
	    colors = np.linspace(0, 1, len(patches1))
	    p1.set_array(np.array(colors))
	    p2.set_array(np.array(colors))
	    # p3.set_array(np.array(colors))
	   

	    fig.savefig('./data_png/%06d.png'%i)
	    plt.close()

if __name__ == '__main__':
	main()

# make movie by 
# ffmpeg -framerate 50 -i ./%06d.png -preset slow -c:v libx264 -crf 17 -pix_fmt yuv420p -movflags faststart test.mov



