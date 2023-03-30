#===
# Heat Equation

Solving

$$
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0
$$

Using [MethodOfLines.jl](http://methodoflines.sciml.ai/dev/) to sumbolically define the PDE system using the finite difference method (FDM).

From the [MethodOfLines tutorial](https://docs.sciml.ai/MethodOfLines/stable/tutorials/heat/).
===#

# ## 2D Steady-state Heat equation

using ModelingToolkit
using MethodOfLines
using DomainSets
using NonlinearSolve
using Plots

#---

@parameters x y
@variables u(..)

Dxx = Differential(x)^2
Dyy = Differential(y)^2

# PDE equation
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0

# Boundary conditions
bcs = [u(0, y) ~ x * y,
       u(1, y) ~ x * y,
       u(x, 0) ~ x * y,
       u(x, 1) ~ x * y]

# Space and time domains
domains = [
    x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)
]

# PDE system
@named pdesys = PDESystem([eq], bcs, domains, [x, y], [u(x, y)])

# Discretization of 2D sapce
N = 10
dx = 1 / N
dy = 1 / N

# Note that we pass in `nothing` for the time variable here,
# since we are creating a stationary problem without a dependence on time, only on space.
discretization = MOLFiniteDifference([x => dx, y => dy], nothing, approx_order=2, grid_align=edge_align)

# The corresponding NonlinearProblem
prob = discretize(pdesys, discretization)
sol = NonlinearSolve.solve(prob, NewtonRaphson())

# Visualize solution
heatmap(sol[x], sol[y], sol[u(x, y)],
        xlabel="x values", ylabel="y values", title="Steady State Heat Equation",
        aspect_ratio=:equal, xlims=(0.0, 1.0), ylims=(0.0, 1.0))

# ## Runtime information

import Pkg
Pkg.status()

#---

import InteractiveUtils
InteractiveUtils.versioninfo()
