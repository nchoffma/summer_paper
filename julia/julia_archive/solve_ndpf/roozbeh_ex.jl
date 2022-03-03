
#= 
Solves the model from Roozbeh's notes using the DEA solver 
from DifferentialEquations.jl
=#

using DifferentialEquations, Plots, Sundials, Distributions

# Parameters

# Preferences
const σ = 1.5
const γ = 3.0
const ψ = 0.2726
const g = 1.0       # utilitarian SWF, all weights = 1

# θ distribution: Pareto Lognormal
const α = 3.0
const μ_θ = -1.0 / α
const σ_θ = 0.5 
const θ_min = 0.27
const θ_max = 104.18

# p = [σ, γ, ψ, α, μ_θ, σ_θ]

function normcdf(x)
    cdf.(Normal(), x) 
end

function normpdf(x)
    pdf.(Normal(), x)
end

function plogncdf(x, α, ν, τ)
    # Calculates F(x) and f(x) for a Pareto-Lognormal distribution
    
    if x < 0.0
        p = 0.0
        d = 0.0
    else
        arg1 = (log(x)-ν) / τ
        arg2 = (log(x) - ν - α * τ ^ 2) / τ
        A = exp(α .* ν + 0.5 * α ^ 2 * τ ^ 2)
        p = normcdf(arg1) - x ^ (-α) * A * normcdf(arg2) # CDF
        d = α * x ^ (-α - 1.0) * A * normcdf(arg2)       # PDF
    end
    return p, d
end

function ft_plogn(x, α, ν, τ)
    # Calculates f(θ) and f′(θ), where f() is the CDF for the PLN dist 

    ~, ft = plogncdf(x, α, ν, τ)
    fpt = -(α + 1.0) * ft / x + exp(α * ν + α ^ 2 * τ ^ 2 / 2.0) * α * x ^ (-α - 1.0) * 
        normpdf( (log(x) - ν -α * τ ^2 ) / τ) / τ / x
    return ft, fpt

end

# Define the residual function
# u = [U μ c l]
# du = [dU dμ dc dl]
# p = [σ, γ, ψ, α, μ, σ_θ]

λ = 0.1
function resid(out, du, u, p, t)
    # Gets the residuals

    # f(θ) and f′(θ)
    ft, fpt = ft_plogn(t, α, μ_θ, σ_θ)

    # Calculate residuals
    out[1] = ψ * (u[4] ^ γ) / t - du[1]
    out[2] = λ * u[3] ^ σ - g - fpt / ft * u[2] - du[2]
    out[3] = u[3] ^ (1 - σ) / (1 - σ) - 
        ψ / γ * u[4] ^ γ - u[1]
    out[4] = t - ψ * u[4] ^ (γ - 1) * u[3] ^ σ + 
        u[2] / (λ * t) * ψ * γ * u[4] ^ (γ - 1)
    out
end

# Initial conditions
u0 = zeros(4)
du0 = zeros(4)
# some Initial conds. cause failure
# idea: use unconstrained allocation
tspan = (θ_min, θ_max)

dvars = [true, true, false, false]
prob = DAEProblem(resid, du0, u0, tspan, differential_vars = dvars)
sol = solve(prob, IDA())
