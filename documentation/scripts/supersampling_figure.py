import numpy as np
import matplotlib.pyplot as plt
from muDIC.phantoms.downsampler import coord_subpos,chess_board

N = 4
# make an empty data set

image = chess_board(1)[:4,:4]

center_coordinate = np.ones((1,1))*1.5
sub_coordinates_i =  coord_subpos(center_coordinate,1.0,4,np.arange(4)[:,np.newaxis]*np.ones((4,4)),0)*4.
sub_coordinates_j =  coord_subpos(center_coordinate,1.0,4,np.arange(4)[np.newaxis,:]*np.ones((4,4)),0)*4.


plt.plot(sub_coordinates_i.flatten(),sub_coordinates_j.flatten(),'o')
# make a figure + axes


for x in range(N + 1):
     plt.axhline(x-0.5, lw=2, color='k', zorder=5)
     plt.axvline(x-0.5, lw=2, color='k', zorder=5)
# draw the boxes
plt.imshow(image,alpha=0.7)#, interpolation='none', extent=[0, N, 0, N], zorder=0)
# turn off the axis labels
plt.axis('off')
plt.show()