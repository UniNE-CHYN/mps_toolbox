import mps
import numpy as np

nx = 550
ny = 500
image = mps.loadtxt("ti_categoricalSoilCracks.txt", nx, ny)

for i in range(3):
    subimage = mps.subimage(image, 200, 150)
    np.savetxt('sub'+str(i)+'.txt', subimage, delimiter='\n')
