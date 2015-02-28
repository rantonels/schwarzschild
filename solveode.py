import numpy as np
from scipy.integrate import odeint
import sys
import math

from multiprocessing import Process, Queue
import random


def w(r):
    return (1-1/r)

def wu(u):
    return w(1/u)


def mUprime(u):
    return -.5*(2*u - 3*u**2 )


def func(u,t):
    if (u[0]>1) or (u[0] < 0.0001):
        return [0,0]
    return [u[1], mUprime(u[0])]

def gradient(u,t):
    return [[0,1],[1-3*u[0] , 0]]


def geod(r0,r0prime,options={}):

    #b = r0**2/w(r0) * omega0
    u0 = [ 1/r0 , -r0prime/(r0*r0) ]


    if('step' in options):
        timestep = options['step']
    else:
        timestep = 0.005

    if('maxangle' in options):
        maxangle = options['maxangle']
    else:
        maxangle = 6.28



    phi = np.arange(0,maxangle,timestep)

    l = phi

    u = odeint(func,u0,l, Dfun=gradient, printmessg=False)
    
    return (l,u)

def geodqueue(q,sci,options):
    
    out = {}

    sys.stdout = open("/dev/null", "w")
    sys.stderr = open("/dev/null", "w")
    
    for el in sci:
        #print el[0], el[1][0],el[1][1]
        res = geod(el[1][0],el[1][1],options)
        idd = el[0]
        out[idd] = res

    q.put(out)

def multicore_list(sc,options ={}): # sc is a dict with indices

    sci = []
    for i in sc:
        sci.append( (i,sc[i]) )

    #sci was populated as a list of (original id, data) tuples.


    #random.shuffle(sci) #shuffling here is not really necessary. Just adds complexity

    l4 = len(sci)/4
    chunks = [
            sci[0:l4],
            sci[l4:2*l4],
            sci[2*l4:3*l4],
            sci[3*l4:]
            ]


    q = Queue()
    processes = []
    for i in range(4):
        processes.append(  Process(target=geodqueue, args=(q,chunks[i],options)) )
    for i in range(4):
        processes[i].start()
    

    results = {}

    for i in range(4):
        got = q.get()
        results.update(got)

    for i in range(4):
        processes[i].join()


    #print len(results), len(sc)

    return results


def deflection_array(r,angles,options = {}):
    rprimes = - r * 1/np.tan(angles)

    inc = {}
    for i in range(len(angles)):
        inc[i] = [r,rprimes[i]]

    res = multicore_list(inc,options)

    ress = [ res[i] for i in range(len(angles)) ]

    deflections = np.zeros((len(angles),5))

    for i in range(len(rprimes)):
        deflections[i,0] = angles[i]

        #print res[i]
        #exit()

        phi  = res[i][0]
        path = res[i][1][:,0]
        pder = res[i][1][:,1]
        
        findex = -1
        for t in range(len(path)):
            if path[t] < 0.001:
                findex = t
                break
            if path[t] > 0.999:
                break

        if findex == -1:
            deflections[i,1] = -1
        else:
            deflections[i,1] = phi[t]

        #deflections[i,2] = path[0]
        #deflections[i,3] = path[1]
        #deflections[i,4] = path[2]

    return deflections


if __name__ == "__main__":
    
    thetas = np.arange(0.01,np.pi,0.01)
    deff = deflection_array(10.0,thetas,{'maxangle':2*np.pi})
    for i in range(len(deff)):
        print deff[i][0], (deff[i][0] - (np.pi -  deff[i][1] ))
    exit()


    rs = np.arange(1.47,1.53,0.0025)





    dirs = np.arange(-40.,-4.,0.2)

    bs = np.arange(0.1,4.,0.1)

    #inc = [ [b*1000,-b*(1000**2)] for b in bs ]
    inc = { d : [10.,d] for d in dirs }

    print "SOLVING"
    trajs = multicore_list(inc,{'maxangle':2*6.28})

    print "SAVING"
    for d in dirs:
        f = open('curves/infall%f'%d,'w')
        (l,u) = trajs[d]
        
        for i in range(len(l)):
            if u[i,0] > 1:
                break
            if u[i,0] < 0.0001:
                break
            f.write(str(l[i]) + "\t" +
                str(1/u[i,0]) + "\t" +
                str(u[i,0]) + "\t" +
                str(u[i,1])
                + "\n"
                )
        f.close()

    sys.exit()




    for d in dirs:
        print d

        f = open('curves/vel%f'%d,'w')

        (l,u) = geod(1.5,d)

       
