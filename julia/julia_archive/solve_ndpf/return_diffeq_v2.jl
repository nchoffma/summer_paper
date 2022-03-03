
# Imposing boundary conditions


using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    QuadGK, NLsolve, ForwardDiff, Optim, LaTeXStrings
gr()
# Parameters

# Preferences
const β = 0.95
const w = 1.2

# Type distribution (Bounded Pareto)
α = 2.0         # high types quite unlikely
θ_min = 0.6
θ_max = 4.8

# Initial Guesses
R = 1.5
λ_1 = 0.1
θ_start = R + 0.001
θ_end = θ_max - 0.001 # get weird results at boundary 

function par(x, α, xm)
    # F(x), f(x), f′(x) for unbounded Pareto

    cdf = 1.0 - (xm / x) ^ α
    pdf = α * xm ^ α / (x ^ (α + 1.0))
    fpx = (-α - 1.0) * α * xm ^ α / (x ^ (α + 2.0))

    return cdf, pdf, fpx
end

# par(θ_max, α, θ_min)

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

# Solving for one U0
p = (β, w, α, θ_min, θ_max, R)
u0 = [1.1 -1e-5] # μ needs to be slightly negative  
tspan = (θ_start, θ_end) # only solving on this interval
prob = ODEProblem(allocations!, u0, tspan, p)
gpath = solve(prob)

# Plot one solution
pU = plot(gpath, vars = (0, 1), title = L"U(\theta)",
    xlab = L"\theta")
pμ = plot(gpath, vars = (0, 2), title = L"\mu(\theta)",
    xlab = L"\theta")
plot(pU, pμ)

# Plot paths for different initial guesses 
function tax_shoot(U0)
    u0 = [U0 -1e-4]
    tspan = (θ_start, θ_max) # only solving on this interval
    prob = ODEProblem(allocations!, u0, tspan, p)
    gpath = solve(prob)
    return gpath
end

tax_shoot(-2.0)

U0_vals = [-3.0 -1.0 0.1 2.0 3.0]
tgrid = range(θ_start, θ_max, length = 100)
U_paths = zeros(100, length(U0_vals))
μ_paths = copy(U_paths)

for i in 1:length(U0_vals)
    for j in 1:100
        soln = tax_shoot(U0_vals[i])
        U_paths[j, i] = soln(tgrid[j])[1]
        μ_paths[j, i] = soln(tgrid[j])[2]
    end
end
pU = plot(tgrid, U_paths, title = L"U(\theta)",
    xlab = L"\theta", label = U0_vals,
    legendtitle = L"U(\theta_{start})",
    legend = :bottomright)
pμ = plot(tgrid, μ_paths, title = L"\mu(\theta)",
    xlab = L"\theta", label = U0_vals,
    legendtitle = L"U(\theta_{start})",
    legend = false)
plot(pU, pμ)
savefig("julia/solve_ndpf/shooting_issue.png")


# Plot the solution


# tax_shoot(1.8)

# The problem is that no value of U(θ̲) seems to make μ(θ̄) = 0
