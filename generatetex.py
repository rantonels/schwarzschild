import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import solveode

SIZE = 512
plt.ion()

img = np.zeros((SIZE,SIZE,3))

thetas = np.linspace(0.0,np.pi,SIZE)
rs = np.linspace(1.0,10.0,SIZE)

for y in range(SIZE):
    if (y%30 == 0):
        plt.imshow(img)
        plt.draw()
    print y
    r = rs[y]
    deff = solveode.deflection_array(r,thetas,{'maxangle':2*np.pi})
    Phi = deff[:,1]
    noteh = deff[:,0] >= 0

    normz = Phi/(2*np.pi)
    (rest,divs) = np.modf(10*normz)
    divs*=0.1

    img[:,y,0] = np.clip(noteh*divs,0.,1.) 
    img[:,y,1] = np.clip(noteh*rest,0.,1.)

plt.imsave('deflections_tex.png',img)
