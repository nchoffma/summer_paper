
# Trying to let Julia solve this as BVP

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    QuadGK, NLsolve, ForwardDiff, Optim, LaTeXStrings, 
    BoundaryValueDiffEq, OrdinaryDiffEq

gr()

# Parameters

# Preferences
const β = 0.95
const w = 1.2

# Type distribution (Bounded Pareto)
α = 2.0         # high types quite unlikely
θ_min = 0.6
θ_max = 4.8

function par(x, α, xm)
    # F(x), f(x), f′(x) for unbounded Pareto

    cdf = 1.0 - (xm / x) ^ α
    pdf = α * xm ^ α / (x ^ (α + 1.0))
    fpx = (-α - 1.0) * α * xm ^ α / (x ^ (α + 2.0))

    return cdf, pdf, fpx
end

# Initial Guesses 
R = 1.5
λ_1 = 0.1
θ_start = R + 0.001

function allocations!(du, u, p, t)
    
    # Solve for c₀, c₁, k
    U, μ = u # current guess
    β, w, α, θ_min, θ_max, R = p
    c0 = μ / (λ_1 * t * (R - t))
    c1 = exp(1.0 / β * (U - log(c0)))
    k = λ_1 * t * c0 ^ 2 / μ * (c1 / (β * c0) - R)
    ~, ft, fpt = par(t, α, θ_min)

    # Differential eqn. 
    du[1] = k / (t * c0)
    du[2] = λ_1 * c1 / β - μ * fpt / ft - 1.0
end

function bounds1!(resid, u, p, t)
    # For this form, need to supply an initial guess as vector
    resid[1] = u[1][2] + 1e-4 
    resid[2] = u[end][2]

end

function bounds2!(resid, sol, p)
    # Both should be close to 0
    resid[1] = sol(θ_start)[2] + 1e-4
    resid[2] = sol(θ_max)[2] + 1e-4 
end

u0 = [1.2 -1e-4]
tspan = (θ_start, θ_max)
p = (β, w, α, θ_min, θ_max, R)
bvp_shoot = BVProblem(allocations!, bounds2!, u0, tspan)
sol_shoot = solve(bvp_shoot, Shooting(Vern7()))

# # Doc example
# # Which also does not work. 
# const g = 9.81
# L = 1.0
# tspan = (0.0,pi/2)
# function simplependulum!(du,u,p,t)
#     θ  = u[1]
#     dθ = u[2]
#     du[1] = dθ
#     du[2] = -(g/L)*sin(θ)
# end

# u₀_2 = [-1.6, -1.7] # the initial guess
# function bc3!(residual, sol, p)
#     residual[1] = sol(pi/4)[1] + pi/2 # use the interpolation here, since indexing will be wrong for adaptive methods
#     residual[2] = sol(pi/2)[1] - pi/2
# end
# bvp3 = BVProblem(simplependulum!, bc3!, u₀_2, tspan)
# sol3 = solve(bvp3, Shooting(Vern7()))