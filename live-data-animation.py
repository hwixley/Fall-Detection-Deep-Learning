import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from celluloid import Camera
from IPython.display import Video
from tqdm import tqdm

def loadNpy(filename):
    with open(os.getcwd() + "/train-val-test/"+ filename, "rb") as f: return np.load(f)

db = "db16.2/window-size-10/lag0/polar-features/"
data_type = ""

X_train, X_val, X_test = loadNpy(db + f"us-X{data_type}_train.npy"), loadNpy(db + f"us-X{data_type}_val.npy"), loadNpy(db + f"us-X{data_type}_test.npy")
y_train, y_val, y_test = loadNpy(db + f"y{data_type}_train.npy"), loadNpy(db + f"y{data_type}_val.npy"), loadNpy(db + f"y{data_type}_test.npy")

samples = X_train[:,:-52]
usr_ftrs = samples[:,-2:]
data = samples[:,:-2].reshape(-1, 10, 73)
#print(data.shape)

#colours=pl.cm.jet(np.linspace(0,1,500))
#Defining a colormap to pass to matplotlib's color parameter,
#just for aesthetics
#The values of t we'll graph over

fig, ax = plt.subplots(figsize=(4, 4))
#Set up our canvas-- we need to specify our figure size here
camera=Camera(fig)#Make a camera of the figure

plt.cla()
plt.yticks(np.linspace(0, 1, 10))
ecg_data = []
label_data = []
label_plots = []
label_colours = []

for window in tqdm(range(10)):
    sample = data[window, :, :]
    label = y_train[window]

    #label_data.append(label)
    if label == 1 and len(label_data) > 0 and label_data[-1] == 0:
        plot_data = [(100/13)*len(ecg_data)]*2
        #ax.plot(plot_data, [0,1], c="g")
        label_plots.append(plot_data)
        label_colours.append("g")
        

    if label == 0 and len(label_data) > 0 and label_data[-1] == 1:
        plot_data = [(100/13)*len(ecg_data)]*2
        #ax.plot(plot_data, [0,1], c="g")
        label_plots.append(plot_data)
        label_colours.append("r")

    label_data.append(label)

    for intvl in range(10):
        ecg = sample[intvl,:13]

        for i, el in enumerate(ecg):
            ecg_data.append(el)
            max_val = (100/13)*len(ecg_data)
            t_list = np.linspace(0, max_val, len(ecg_data))
            #Our animation will have each frame be a different value of k from the list above    
            ax.plot(t_list, ecg_data, c="b")

            for i, plot in enumerate(label_plots):
                ax.plot(plot, [0,1], c=label_colours[i])
            #llist = np.linspace(0, max_val, len(label_data))
            #ax.plot(llist, label_data, c="b")
            #ax.set_xticks(t_list)
            #For a given value of k, we'll plot our function with that k-value   
            camera.snap()

animation=camera.animate();#Make the animation
plt.close() #Stop the empty plot from displaying
animation.save('data_animation.mp4',fps=500/10) #Save the animation-- notes below
Video("data_animation.mp4") #Show the video you've just saved
