# THIS SCRIPT IS AN EXPERIMENT
# it's not needed for the online Schwarzschild BH visualization
# and it's not good.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
import solveode
import random
import sys
import time

plt.ion()

SCSIZE = (640,480)
FOVDEG = 70
CAMPOS = np.array([0.,-10.,0.])

w,h = SCSIZE
fov = (FOVDEG * np.pi) / 180.

MAGENTA = np.array([1.,0,1.])
WHITE = np.array([1.,1.,1.])
BLACK = np.array([0.,0.,0.])
GREEN = np.array([0.,1.,0.])
DARK_GREEN = np.array([0.,.2,0.])
ZERO = np.array([0,0,0])

print("loading ha1024.jpg...")
img_bg = mpimg.imread('tex/ha1024.jpg')

def get_bg_uv(u,v):
    try:
        return img_bg[int(v*2048),int(u*4096),:]
    except ValueError and IndexError:
        return MAGENTA

def normalize(x):
    norm = np.linalg.norm(x)
    if (norm < 0.00001):
        return ZERO
    x /= np.linalg.norm(x)
    return x

def get_bg_vec3(vec):

    u = .5 +  \
        np.arctan2(vec[0],vec[1])/(2*np.pi)
    v = .5 +  \
        np.arctan2(np.sqrt(vec[0]**2+vec[1]**2),vec[2])/(2*np.pi)
    return get_bg_uv(
                np.clip((u+.5)%1,0,0.999),
                np.clip(v,0,0.999)
            )

def trace_ray_chunk(inc):


   
    return get_bg_vec3(normalize(r))


def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

# distance from the black hole is norm of camera position 
r = np.linalg.norm(CAMPOS)


# raytracing deflection angles

thetas = np.arange(0.01,np.pi,0.01)

defdata = solveode.deflection_array(r,thetas,{'maxangle':2*np.pi})
Phi = defdata[:,1]
deff = thetas - (np.pi - Phi)


def bigPhi(theta):
    index =  (theta - thetas[0])/(thetas[-1]-thetas[0]) * len(thetas)
    return ndimage.interpolation.map_coordinates(Phi,[[index]],order=1)[0]




# setup the image array
img = np.zeros((h,w,3))
for i in range(w):
    for j in range(h):
        img[j,i,:] = DARK_GREEN

#useful constant
origin = np.array([0,0,0])


print("raytracing.")


# an array of (x,y) pixel coordinates is shuffled. We want to process pixel uniformly,
# to get an idea of the image while it is forming.

totpix = w*h
pixlist = [ (i,j) for i in range(w) for j in range(h) ]

#random.shuffle(pixlist)




#CHNKSZ  = 200    # size of chunks. We process 200 rays at a time.
#NRCHNKS = totpix/CHNKSZ + 1 # number of chunks
#chunklistiterator = chunks(pixlist,CHNKSZ)
#chunklist = []
#for c in chunklistiterator:
#    chunklist.append(c)
#
#
#checkcnt = 0
#for c in chunklist:
#    checkcnt += len(c)
#
#
#if (checkcnt != totpix):
#    print("WTF",checkcnt,totpix)
#    exit()
#
donecheck = {}
for (x,y) in pixlist:
    donecheck[(x,y)] = 0
#
#ccounter = -1
#for c in chunklist:
#    ccounter += 1
#    
#    print((100*ccounter)/NRCHNKS, "%")   #print progress and refresh display
#    plt.imshow(img)                         #between every chunk.
#    plt.draw()
#
#    
#    #inc = {} # preparing the array of initial conditions. This will need to be an array of
#
#             # (r,dr/dphi) (phi = 0) pairs.
#
#    rays = {}


print("creating rays")

#compute rays
xfscales = np.linspace(-.5,.5,w)
yfscales = np.linspace(-.5,.5,h)*(float(h)/float(w))

#uvs = np.array( float(x-w/2)/w

rays = np.ones((len(xfscales),len(yfscales),3))

#generating rays
for pixel in pixlist:
    (x,y) = pixel
    rays[x,y,0] = 2*xfscales[x]
    rays[x,y,2] = 2*yfscales[y]

#normalizing

print("normalizing rays")

invnorms = 1/np.sqrt( np.einsum('ijk,ijk->ij',rays,rays))

rays = np.einsum('ijk,ij->ijk',rays,invnorms)


print("computing view angles")

normcampos = normalize(CAMPOS)

mindots = -np.einsum('k,ijk->ij',CAMPOS,rays)

thetarr = np.arccos(mindots)

print("interpolating deflection angles")

bigPhiArr = np.zeros((w,h))

for (x,y) in pixlist:
    bigPhiArr[x,y] = bigPhi(thetarr[x,y])


print("starting rendering...")

pixcount = 0

for pixel in pixlist:
        if (pixcount % 400 == 0):
            print((100*pixcount)/len(pixlist) , "%")
            img = np.clip(img,0.0,1.0)
            plt.imshow(img)
            plt.draw()


        (x,y) = pixel
        # we compute the viewing vector with standard trig

        xf = float(x)/w
        yf = float(y)/h
        xfscale = float(x-w/2)/w
        yfscale = float(y-h/2)/w

        # tsin = np.sin((yf-.5)*fov/w*h)
        # tcos = np.cos((yf-.5)*fov/w*h)
        # 
        # ray = np.array([np.sin((xf-.5)*fov) * tcos,
        #                 tcos,
        #                 tsin
        #             ])
       
        ray = rays[x,y,:]
        

        # theta is angle between - R and v
        #theta = np.arccos( - np.dot(normalize(CAMPOS),normalize(ray)) ) 
        #theta = np.arccos( mindots[x,y] )
        theta = thetarr[x,y] 

        # dr/dphi is trickier. We compute the radial and orthogonal components of the ray:

        #dr = np.dot(ray,CAMPOS)/r # signed projection of ray over CAMPOS
        #dx_vec = ray - (dr)*CAMPOS/r        # we obtain the orthogonal component by subtracting
                                        # the projection
        #dx = np.linalg.norm(dx_vec)

        #rprime = r * dr/dx    # dr/dphi = r * dr/dx (since dx = R dphi)

        #print ray,r,rprime
        #sys.exit()

        #finally we append our prepared initial conditions
        #inc[(x,y)] = np.array([r,rprime])

        #rays[(x,y)] = ray

        donecheck[(x,y)] = 1

    # we evolve our chunk of initial conditions.
    #tmpres = solveode.multicore_list(inc)

     
        if donecheck[(x,y)] == 0:
            print("ERROR: untraced ray in postprocessing")
            exit()
        if donecheck[(x,y)] == 2:
            print("ERROR: reprocessing pixel")
            exit()

        yp = h-y-1


        #debugging red pre-fill
        img[yp,x,:] = np.array([0.7,0.,0.])


        #phi = tmpres[(x,y)][0]
        #path = tmpres[(x,y)][1]
        #nray = rays[(x,y)]
       

        # wether we hit the horizon
        #horizon = False
        #if (  path[:,0] > 0.999 ).any() :
        #    horizon = True

        horizon = False

        ##debugging: a rough estimator for deflection (nearest approach to eh)
        #deflection = np.amax(path[-1,0])
        #strprime = abs(inc[(x,y)][1])*0.1
         


        #estimating deflection angle
        
        #upper = path[:,0] * (path[:,1] > 0) # this is u(phi) when u' > 0 and 0 otherwise
        #dangle = phi[ np.argmax(upper) ]

        #dangle = -1
        #findex = -1 
        #for i in range(len(path[:,0])):
        #    if (path[i,0] < 0.001): #and (path[i,1] < 0):
        #        dangle = phi[i]
        #        findex = i
        #        break
        #
        #if (findex < 0):
        #    horizon = True
            

        # for i in range(len(path[:,0])):
        #     print i, path[i,0]
        # print
        # print findex,dangle
        # sys.exit()

        # estimating final direction vector
        
        Phee = bigPhiArr[x,y] 
        if (Phee < 0):
            horizon = True
        


        avs = -normalize(np.cross( CAMPOS , ray ))
        xvs = normalize(np.cross( CAMPOS, avs))

        fvs = np.sin(Phee) * xvs + np.cos(Phee) * normalize(CAMPOS)

        

                
        # img[yp,x,:] = normalize(ray)
        img[yp,x,:] = get_bg_vec3( ray ) 
        #img[yp,x,:] = get_bg_vec3(fvs)
        # img[yp,x,:] =  (theta)/np.pi*WHITE 
        # img[yp,x,:] = deflection*WHITE 
        #img[yp,x,:] = (Phee-np.pi)/np.pi * WHITE
        # img[yp,x,:] = np.array([float(x)/w,float(y)/h,0])
        
        if horizon:
            img[yp,x,:] = BLACK



        donecheck[(x,y)] = 2

        pixcount += 1


img = np.clip(img,0.0,1.0)

plt.imsave('test.png',img)
plt.imshow(img)
plt.draw()
time.sleep(1)
