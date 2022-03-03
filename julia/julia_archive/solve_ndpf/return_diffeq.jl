
# Solves the taxation problem as a system of ODEs

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    QuadGK, NLsolve, ForwardDiff, Optim, LaTeXStrings
gr()
# Parameters

# Preferences
const β = 0.95
const w = 1.2

# Type distribution (Bounded Pareto)
α = 3.0         # high types quite unlikely
θ_min = 0.6
θ_max = 4.8

# Initial Guesses
R = 1.3
λ_1 = 1.0
θ_start = R + 0.0001

# Differential Equation

# p = (β, w, α, θ_min, θ_max) # parameters 

function par(x, α, xm)
    # F(x), f(x), f′(x) for unbounded Pareto
    # Lack of upper bound frees us from the constraint that μ(θ̄) = 0

    cdf = 1.0 - (xm / x) ^ α
    pdf = α * xm ^ α / (x ^ (α + 1.0))
    fpx = (-α - 1.0) * α * xm ^ α / (x ^ (α + 2.0))

    return cdf, pdf, fpx
end

# par(θ_max, α, θ_min) # capture all but top 

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

# Shooting
function U_shoot(U0)

end

# Goal: approximate ∫[c₀ + k]fdθ over tgrid 
function feas0(t, path)
    U, μ = path(t) # use the interpolation ability
    c0 = μ / (λ_1 * t * (R - t))
    c1 = exp(1.0 / β * (U - log(c0)))
    k = λ_1 * t * c0 ^ 2 / μ * (c1 / (β * c0) - R)
    ~, ft, ~ = par(t, α, θ_min)

    return (c0 + k) * ft

end

function feas1(t, path)
    U, μ = path(t) 
    c0 = μ / (λ_1 * t * (R - t)) 
    c1 = exp(1.0 / β * (U - log(c0)))
    k = λ_1 * t * c0 ^ 2 / μ * (c1 / (β * c0) - R)
    ~, ft, ~ = par(t, α, θ_min)

    return (t * k - c1) * ft

end

# Shooting function 
function tax_shoot(U0)
    # Shooting algorithm
    # Idea: find U(θ̲) such that the feasibility constraints clear

    # Given U0, solve the differential equation
    p_shoot = (β, w, α, θ_min, θ_max, R)
    θ_start = R + 0.001
    u0 = [U0 -1e-4] # μ needs to be slightly negative  
    tspan = (θ_start, θ_max) # only solving on this interval
    prob = ODEProblem(allocations!, u0, tspan, p_shoot)
    gpath = solve(prob)
    
    # Check feasibility
    
    # Total consumption and lending from θ < R types
    tgrid = gpath.t
    FR, ~, ~ = par(R, α, θ_min)
    c0_lend = w / (1.0 + β) * FR
    c1_lend = R * c0_lend

    # feasibility (z)
    z = zeros(2)
    total_spend0, ~ = quadgk(t -> feas0(t, gpath), tgrid[1], tgrid[end])
    z[1] = w - total_spend0 - c0_lend
    total_spend1, ~ = quadgk(t -> feas1(t, gpath), tgrid[1], tgrid[end])
    z[2] = total_spend1 - c1_lend

    norm = maximum(abs.(z))

end

res = optimize(tax_shoot, -3.0, -0.1)
U0_opt = res.minimizer

# Solve using the best 
p = (β, w, α, θ_min, θ_max, R)
u0 = [U0_opt -1e-4] # μ needs to be slightly negative  
tspan = (θ_start, θ_max) # only solving on this interval
prob = ODEProblem(allocations!, u0, tspan, p)
gpath = solve(prob)

# One potential issue: μ never "turns around," but instead decreases 
# monotonically. So, setting μ(θ̄) = 0 is tricky. 

# Plot the solution
pU = plot(gpath, vars = (0, 1), labels = L"U(\theta)",
    xlab = L"\theta")
pμ = plot(gpath, vars = (0, 2), labels = L"\mu(\theta)",
    xlab = L"\theta")

# Allocations
tgrid = gpath.t
soln = convert(Array, gpath)[1, :, :]
U_opt = soln[1, :]
μ_opt = soln[2, :]

c0_opt = μ_opt ./ (λ_1 * tgrid .* (R .- tgrid))
c1_opt = exp.(1.0 ./ β * (U_opt .- log.(c0_opt)))
k_opt = λ_1 * tgrid .* c0_opt .^ 2 ./ μ_opt .* (c1_opt ./ (β * c0_opt) .- R)
b_opt = c0_opt + k_opt .- w

pc = plot(tgrid, [c0_opt c1_opt],
    label = [L"c_0(\theta)" L"c_1(\theta)"],
    xlab = L"\theta") 
pk = plot(tgrid, k_opt, label = L"k(\theta)",
    xlab = L"\theta")
    
plot(pU, pμ, pc, pk)
savefig("julia/solve_ndpf/allocs.png")

# # What's up with k(θ)
# t = tgrid[10]
# U, μ = gpath[10]
# c0 = μ / (λ_1 * t * (R - t))
# c1 = exp(1.0 / β * (U - log(c0)))
# k = 1e-4 * t * c0 ^ 2 / μ * (c1 / (β * c0) - R)