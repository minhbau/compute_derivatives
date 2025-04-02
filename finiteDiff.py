import numpy as np
import matplotlib.pyplot as plt
'''
	Compute the first derivative with central finite difference (CFD), uniform grids
	with 4th, 6th, and 8th order accuracy.
	Coefficients can be found here, https://en.wikipedia.org/wiki/Finite_difference_coefficient

	NOTE: the same coefficients of the backward/forward approximations of 'even' derivatives
	However, 'minus' sign is added for 'odd' backward derivatives (1,3,5...)

	Run the following 'magma' code to get coefficients of forward/backward:
	https://math.stackexchange.com/questions/3358496/numerical-method-forward-finite-difference-coefficient
	P<S>:=PowerSeriesRing(Rationals());
	for n in [1..1] do
	for p in [1..8] do
		n,p,Evaluate(Truncate(Log(1+S+O(S^(p+1)))^n),S-1);
	end for;" ";
	end for;

	Written by Minh Bau Luong, CRFL, CCRC, KAUST, 02  Oct 2021
'''

def derivative_8thCFD(f,dx):
	'''
	8th order central finite difference, uniform grids
	'''
	N = f.size
	dfdx = np.zeros(N,dtype=np.float64)

	# Coefficients for interior grids
	c = [0, 4/5, -1/5, 4/105, -1/280] # c1 = 4/5; c2 = -1/5; c3 = 4/105; c4 = -1/280
	dfdx[4:-4] = -c[1]*f[3:-5] - c[2]*f[2:-6] - c[3]*f[1:-7] - c[4]*f[:-8]  \
			    + c[1]*f[5:-3] + c[2]*f[6:-2] + c[3]*f[7:-1] + c[4]*f[8:]

	# Four grids at Left boundary is computed by 8th order forward finite difference
	c=[-761/280, 8, -14, 56/3, -35/2, 56/5, -14/3, 8/7, -1/8]
	for i in range(4):
		dfdx[i] = c[0]*f[i] + c[1]*f[1+i] + c[2]*f[2+i] + c[3]*f[3+i] + c[4]*f[4+i] + c[5]*f[5+i] + c[6]*f[6+i] + c[7]*f[7+i] + c[8]*f[8+i]

	# Four grids at Right boundary is computed by 8th order backard finite difference
	# Here 'minus' sign is added for 'odd' backward derivatives in the loop at '-='
	c.reverse() # c[i] for grids: [- 8 - 7 -6 -5 -4 -3 -2 -1 0]
	for i in range(1,5): # -i for backward grids
		dfdx[-i] -= c[0]*f[-8-i] + c[1]*f[-7-i] + c[2]*f[-6-i] + c[3]*f[-5-i] + c[4]*f[-4-i] + c[5]*f[-3-i] + c[6]*f[-2-i] + c[7]*f[-1-i] + c[8]*f[-i]

	return dfdx/dx

def derivative_6thCFD(f,dx):
	'''
	6th order central finite difference, uniform grids
	'''
	N = f.size
	dfdx = np.zeros(N,dtype=np.float64)

	# Coefficients for interior grids
	c = [0, 3/4, -3/20, 1/60]
	dfdx[3:-3] = - c[1]*f[2:-4] - c[2]*f[1:-5] - c[3]*f[:-6] \
			     + c[1]*f[4:-2] + c[2]*f[5:-1] + c[3]*f[6:]

	# Three grids at Left boundary is computed by 6th order forward finite difference
	c=[-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6]
	for i in range(3):
		dfdx[i] = c[0]*f[i] + c[1]*f[1+i] + c[2]*f[2+i] + c[3]*f[3+i] + c[4]*f[4+i] + c[5]*f[5+i] + c[6]*f[6+i]

	# Three grids at Right boundary is computed by 6th order backard finite difference
	# Here 'minus' sign is added for 'odd' backward derivatives in the loop at '-='
	c.reverse() # c[i] for grids: [-6 -5 -4 -3 -2 -1	0]
	for i in range(1,4): # -i for backward grids
		dfdx[-i] -= c[0]*f[-6-i] + c[1]*f[-5-i] + c[2]*f[-4-i] + c[3]*f[-3-i] + c[4]*f[-2-i] + c[5]*f[-1-i] + c[6]*f[-i]

	return dfdx/dx

def derivative_4thCFD(f,dx):
	'''
	4th order central finite difference, uniform grids
	'''
	N = f.size
	dfdx = np.zeros(N,dtype=np.float64)

	# Coefficients for interior grids
	c = [0,  2/3, -1/12]
	dfdx[2:-2] = - c[1]*f[1:-3] - c[2]*f[:-4]\
			     + c[1]*f[3:-1] + c[2]*f[4:]

	# Two grids at Left boundary is computed by 4th order forward finite difference
	# coefficients are:
	c=[-25/12, 4, -3, 4/3, -1/4]
	for i in range(2):
		dfdx[i] = c[0]*f[i] + c[1]*f[1+i] + c[2]*f[2+i] + c[3]*f[3+i] + c[4]*f[4+i]

	# Two grids at Right boundary is computed by 4th order backard finite difference
	# Here 'minus' sign is added for 'odd' backward derivatives in the loop at '-='
	c.reverse() # c[i] for grids: [-4 -3 -2 -1	0]
	for i in range(1,3): # -i for backward grids
		dfdx[-i] -= c[0]*f[-4-i] + c[1]*f[-3-i] + c[2]*f[-2-i] + c[3]*f[-1-i] + c[4]*f[-i]

	return dfdx/dx

# ---------------------------------------------------------------
# Compare the central finite differences (CFD) with exact solution
# ---------------------------------------------------------------
h = 0.5; x = np.arange(0, 2*np.pi, h)
exact_solution = -np.sin(x)

# Finite difference approximation
y = np.cos(x)
dydx_4thCFD = derivative_4thCFD(y,h)
dydx_6thCFD = derivative_6thCFD(y,h)
dydx_8thCFD = derivative_8thCFD(y,h)

# Plot to compare 4th, 6th, and 8th order derivatives
plt.figure(figsize = (12, 8))
plt.plot(x, exact_solution, label = 'Exact solution')
plt.plot(x, dydx_8thCFD, '-x', label = '8thCFD')
plt.plot(x, dydx_6thCFD, '-o', label = '6thCFD')
plt.plot(x, dydx_4thCFD, '--', label = '4thCFD')
plt.legend()
plt.xlabel("x")
plt.ylabel("dfdx")
plt.show()

# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Compute the maximum error for different grid size
# ---------------------------------------------------------------
# define step size and the number of iterations
h = 10; iterations = 15

# Use list store the grid sizes the max error
dx = []
max_error_4th = []
max_error_6th = []
max_error_8th = []

for i in range(iterations):
	h /= 2 # halve the step size
	dx.append(h)
	x = np.arange(0, 10*np.pi, h)

	exact_solution = -np.sin(x)

	# Finite difference approximation
	y = np.cos(x)
	dydx_4th = derivative_4thCFD(y,h)
	dydx_6th = derivative_4thCFD(y,h)
	dydx_8th = derivative_4thCFD(y,h)
	# dydx = np.gradient(y,h)

	# compute exact solution
	exact_solution = -np.sin(x)

	# Compute max error between
	# numerical derivative and exact solution
	max_error_4th.append(max(abs(exact_solution - dydx_4th)))
	max_error_6th.append(max(abs(exact_solution - dydx_6th)))
	max_error_8th.append(max(abs(exact_solution - dydx_8th)))

# produce log-log plot of max error versus the grid sizes
plt.figure(figsize = (12, 8))
plt.loglog(dx, max_error_4th, '-x', label = '8thCFD')
plt.loglog(dx, max_error_6th, '-o', label = '6thCFD')
plt.loglog(dx, max_error_8th, '--', label = '4thCFD')
plt.xlabel("dx")
plt.ylabel("max_error")
plt.show()
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Test with a shock wave showing a stiff gradient
# ---------------------------------------------------------------
def stiff_gradient_function(x, x0=0.1, width=0.001):
    """
    Function with a stiff gradient centered at x0.
    Mimics the structure of a shock wave.

    Parameters:
    - x: array of points where the function is evaluated
    - x0: center of the steep gradient
    - width: controls how steep the transition is

    Returns:
    - f(x): np.array
    """
    return np.tanh((x - x0) / width)

# Example usage and visualization
x = np.linspace(0.0, 0.2, 2000)
f = stiff_gradient_function(x,width=0.0001)

plt.plot(x, f)
plt.title("Function with Stiff Gradient (Mimics a Shock)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

h = x[1]
dydx = derivative_4thCFD(f,h)
dydx_4thCFD = derivative_4thCFD(f,h)
dydx_6thCFD = derivative_6thCFD(f,h)
dydx_8thCFD = derivative_8thCFD(f,h)
plt.figure(figsize = (12, 8))
plt.plot(x, dydx_8thCFD, '-x', label = '8thCFD')
plt.plot(x, dydx_6thCFD, '-o', label = '6thCFD')
plt.plot(x, dydx_4thCFD, '--', label = '4thCFD')
plt.xlim(0.095, 0.105)
plt.legend()
plt.xlabel("x")
plt.ylabel("dfdx")
plt.show()
# ---------------------------------------------------------------