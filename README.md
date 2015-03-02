# [Schwarzschild Black Hole](http://spiro.fisica.unipd.it/~antonell/schwarzschild/)

A real-time WebGL visualization of a Schwarzschild Black Hole.

## Dependencies

python2, numpy, scipy, matplotlib

## Components

### solveode.py

`solveode.py` is an odeint()-powered general solver for the photon-orbit equation described in [here](http://spiro.fisica.unipd.it/~antonell/schwarzschild/). It is imported as a python module and provides a minimal set of function for parallel solution for lists of initial conditions.

### generatetex.py

`generatetex.py` is the script that generates the lookup texture for the deflection angle using `solveode.py`. It produces `deflections_tex.png`, used by the WebGL applet.

### index.html

This is the webpage for the live applet, including the shader source.
