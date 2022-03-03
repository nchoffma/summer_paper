
# Solves the Differential-Algebraic Equation (DAE) example
# https://docs.sciml.ai/v5.0.0/tutorials/dae_example.html

using DifferentialEquations, Plots, Sundials
gr()

# Solves the Roberts model
# Inputs are y₁, y₂, y₃ and their derivatives

# Residual function (want f = 0)
function f(out, du, u, p, t)
    # out is the residuals (modified in place)
    # u is the vector of functions [y₁, y₂, y₃]
    # du is the vector of derivatives [dy₁, dy₂, dy₃] (note dy₃ is not used)
    # p is the vector of parameters (not needed)
    # t is time 

    out[1] = - 0.04u[1] + 1e4*u[2]*u[3] - du[1]
    out[2] = + 0.04u[1] - 3e7*u[2]^2 - 1e4*u[2]*u[3] - du[2]
    out[3] = u[1] + u[2] + u[3] - 1
    out
end

# Initial conditions
u₀ = [1.0, 0.0, 0.0]
du₀ = [-0.04, 0.04, 0.0]
tspan = (0.0, 100000.0)

dvars = [true true false] # y₁ and y₂ appear as differentials (dy), y₃ does not 

prob = DAEProblem(f, du₀, u₀, tspan, differential_vars = dvars)
sol = solve(prob, IDA())

plot(sol, xscale=:log10, tspan=(1e-6, 1e5), layout=(3,1))