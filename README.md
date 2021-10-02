# Compute derivatives
First derivative is computed with 4th, 6th, and 8th order finite different methods (FDM) for a uniform grid.

# Validation:
## Cosine - the derivative of the sine function is used to compare its the exact solution with the FD approximation

```rb
h = 0.2; x = np.arange(0, 2*np.pi, h)
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
plt.show()
```
