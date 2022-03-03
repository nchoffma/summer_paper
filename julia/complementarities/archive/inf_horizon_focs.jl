#= 
Testing FOCs in the component planner's problem
=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles

gr()

println("******** comp_inf_horizon.jl ********")

# Parameters
const β = 0.95        # discounting
θ_min = 1.
θ_max = 2.

ϵ = 4.
R = 1.5 # near value from primal (static)
A0 = -1.
# LF allocations, as initial guess for Newton 
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 0.3 * ones(nt)
klf = 0.7 * ones(nt)
c1_lf = tgrid .* klf
wp_lf = log.(c1_lf) ./ (1. - β) # get theta*k forever
Ulf = log.(c0_lf) + β .* wp_lf
mu_lf = 0.2 .* ones(nt)
mu_lf[[1 end]] .= 0.

function fdist(x)
    
    # Bounded Pareto
    a = 1.5 # shape parameter
    L = θ_min
    H = θ_max 
    # den = 1.0 - (L / H) ^ a 
    
    # cdf = (1.0 - L ^ a * x ^ (-a)) / den 
    # pdf = a * L ^ a * x ^ (-a - 1.0) / den 
    # fpx = a * (-a - 1.0) * L ^ a * x ^ (-a - 2.0)
    
    # Uniform
    cdf = (x - L) / (H - L)
    pdf = 1. / (H - L)
    fpx = 0.
    
    return cdf, pdf, fpx
    
end

function foc_k(x, U, μ, θ, Y)
        
    c0, k, wp = x
    
    # FOC vector
    focs = zeros(3)
    focs[1] = A0 / (β * R * c0) * (1. - β) * exp((1. - β) * wp) + k / (θ * c0 ^ 2) * μ - 1.
    focs[2] = (1. / R) * (Y / k * θ ^ (ϵ - 1)) ^ (1.0 / ϵ) - μ / (θ * c0) - 1.
    focs[3] = log(c0) + β * wp - U 
    
    # Jacobian matrix
    dfocs = zeros(3, 3)
    dfocs[1, 1] = -A0 / (β * R * c0 ^ 2) * (1. - β) * exp((1. - β) * wp) - 2k * μ / (θ * c0 ^ 3)
    dfocs[1, 2] = μ / (θ * c0 ^ 2)
    dfocs[1, 3] = A0 / (β * R * c0) * (1. - β) ^ 2 * exp((1. - β) * wp) 
    
    dfocs[2, 1] = μ / (θ * c0 ^ 2)
    dfocs[2, 2] = (-1. / ϵ) * (1. / R) * θ ^ (1. - 1. / ϵ) * Y ^ (1. / ϵ) * k ^ (-1. / ϵ - 1.)
    
    dfocs[3, 1] = 1.0 / c0
    dfocs[3, 3] = β
    
    return focs, dfocs
    
end

i = 1
x0 = [c0_lf[i] klf[i] wp_lf[i]]'
foc, dfoc = foc_k(x0, Ulf[i], mu_lf[i], tgrid[i], 1.)
xt = newton_k(Ulf[i], mu_lf[i], tgrid[i], i, 1.)

function newton_k(U, μ, θ, i, Y)
    x0 = [c0_lf[i] klf[i] wp_lf[i]]'
    mxit = 500
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0
    
    while diff > tol && its < mxit
        focs, dfocs = foc_k(x0, U, μ, θ, Y)
        println("x0 = $x0")
        println(dfocs)
        diff = norm(focs, Inf)
        d = dfocs \ focs 
        while minimum(x0[[1 2]] - d[[1 2]]) .< 0.0 # c0 and k need to be positive, w need not be
            d = d / 2.0
            if maximum(d[[1 2]]) < tol
                fail = 1
                println("warning: newton failed")
                break
            end
        end
        if fail == 1
            break
        end
        
        x0 = x0 - d 
        its += 1
        
    end
    return x0, fail
    
end

function alloc_single(U, μ, t, Y)
    
    it = findmin(abs.(tgrid .- t))[2] # for finding initial guess
    
    x, f = newton_k(U, μ, t, it, Y)
    c0, k, wp = x 
    return c0, k, wp
    
end

# Optimal allocations along the solution 
function opt_allocs(Y)
    c0p = zeros(nt)
    kp = zeros(nt)
    wp = zeros(nt)
    
    for i in 1:nt
        # println(i)
        c0p[i], kp[i], wp[i] = alloc_single(Ulf[i], mu_lf[i], tgrid[i], Y)
    end
    
    return c0p, wp, kp
end

c0p, wp, kp = opt_allocs(1.)
plot(tgrid, [c0p kp wp],
    label = [L"c_0" L"k" L"w^\prime"])

a = [0.0 0.0 -0.0; 0.0 -1.265655434250118 0.0; 6.622190579091928e-10 0.0 0.95]
det(a)