#===
# Solving Brusselator PDE from a system of ODEs

From [solving large stiff equations](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/) in `DifferentialEqautions.jl` tutorial.

The Brusselator PDE system is defined as follows:

$$
\begin{align}
\frac{\partial u}{\partial t} &= 1 + u^2v - 4.4u + \alpha (\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}) + f(x, y, t) \\
\frac{\partial v}{\partial t} &= 3.4u - u^2 v + \alpha (\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2})
\end{align}
$$

where

$$
f(x, y, t) =
\begin{cases}
5 \qquad \text{if} (x - 0.3)^2 + (y - 0.6)^2 \leq 0.1^2 \ and \  t \geq 1.1  \\
0 \qquad \text{otherwise}
\end{cases}
$$

and the initial conditions are

$$
\begin{align}
u(x, y, 0) &= 22(y(1-y))^{1.5} \\
v(x, y, 0) &= 27(x(1-x))^{1.5}
\end{align}
$$

with the periodic boundary condition

$$
\begin{align}
u(x+1, y, 0) &= u(x, y, t)  \\
u(x, y+1, 0) &= u(x, y, t)
\end{align}
$$

on a timespan of $t \in [0, 11.5]$.


We could discretize it into a system of ODEs with the finite difference method (FDM).
===#

using OrdinaryDiffEq
using LinearAlgebra

#---

# The model
function make_brusselator_2d(N=32)

    ## Grid points
    xyd_brusselator = range(0, stop=1, length=N)
    dx = step(xyd_brusselator)

    ## Non-linerat part of the PDE
    brusselator_f(x, y, t) = 5.0 * (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) * (t >= 1.1)

    ## Boundary limits
    limit(a, N) = a == N+1 ? 1 : a == 0 ? N : a

    function brusselator_2d_loop!(du, u, p, t)
        A, B, alpha = p
        alpha = alpha/dx^2
        @inbounds for I in CartesianIndices((N, N))
            ## Indices
            i, j = Tuple(I)
            ## Corrdinates
            x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
            ## neuighbor indices
            ip1, im1, jp1, jm1 = limit(i+1, N), limit(i-1, N), limit(j+1, N), limit(j-1, N)

            ## Hand-written finite difference method
            du[i,j,1] = alpha*(u[im1,j,1] + u[ip1,j,1] + u[i,jp1,1] + u[i,jm1,1] - 4u[i,j,1]) +
                        B + u[i,j,1]^2*u[i,j,2] - (A + 1)*u[i,j,1] + brusselator_f(x, y, t)
            du[i,j,2] = alpha*(u[im1,j,2] + u[ip1,j,2] + u[i,jp1,2] + u[i,jm1,2] - 4u[i,j,2]) +
                        A*u[i,j,1] - u[i,j,1]^2*u[i,j,2]
        end
    end

    function init_brusselator_2d(xyd)
        N = length(xyd)
        u = zeros(N, N, 2)
        for I in CartesianIndices((N, N))
            x = xyd[I[1]]
            y = xyd[I[2]]
            u[I,1] = 22*(y*(1-y))^(3/2)
            u[I,2] = 27*(x*(1-x))^(3/2)
        end
        u
    end

    u0 = init_brusselator_2d(xyd_brusselator)

    return (f=brusselator_2d_loop!, u0 = u0)
end

#---

f, u0 = make_brusselator_2d()
tspan = (0., 11.5)
ps = (3.4, 1., 10.)
prob_ode_brusselator_2d = ODEProblem(f, u0, tspan, ps)

#---

@time solve(prob_ode_brusselator_2d, TRBDF2(), save_everystep=false)

# ## Using Jacobian-Free Newton-Krylov linear solver

using LinearSolve

@time solve(prob_ode_brusselator_2d, KenCarp47(linsolve=KrylovJL_GMRES()), save_everystep=false)

# ## Using Sundials solvers

using Sundials

@time solve(prob_ode_brusselator_2d, CVODE_BDF(linear_solver=:GMRES), save_everystep=false)

# ## Runtime information

import Pkg
Pkg.status()

#---

import InteractiveUtils
InteractiveUtils.versioninfo()
