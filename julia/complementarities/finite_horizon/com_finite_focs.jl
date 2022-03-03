#= 
Attempting to clear the FOCs in the finite horizon model 
=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles, ApproxFun

gr()
println("******** comp_finite_vf.jl ********")

# Parameters
const β = 0.95        # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

# Discrete allocations, as initial guess for Newton 
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
# c0_lf = 0.5 * ones(nt)
# klf = 0.5 * ones(nt)
# c1_lf = tgrid .* klf
# wp_lf = log.(c1_lf) ./ (1. - β)
# Ulf = log.(c0_lf) + β * wp_lf
# μlf = 0.1 * ones(nt)
# μlf[[1 end]] .= 0. 

matpath = "julia/complementarities/finite_horizon/discrete_results_finite/"
c0_lf = readdlm(matpath * "csol.txt", '\t', Float64, '\n')
klf   = readdlm(matpath * "ksol.txt", '\t', Float64, '\n')
wp_lf = readdlm(matpath * "wsol.txt", '\t', Float64, '\n')
a0_ig = readdlm(matpath * "a0sol.txt", '\t', Float64, '\n')
gams = readdlm(matpath * "gamma.txt", '\t', Float64, '\n')
Ulf = log.(c0_lf) + β * wp_lf
μlf = 0.1 * ones(nt)
μlf[[1 end]] .= 0. 

ne = nt - 1     # number of intervals
nq = 5          # number of points for GL integration

# Distribution for output 
function fdist(x)
    
    L = θ_min
    H = θ_max 
    
    # Uniform
    cdf = (x - L) / (H - L)
    pdf = 1. / (H - L)
    fpx = 0.
    
    return cdf, pdf, fpx
    
end

function qgausl(n, a, b)
    # Gauss-Legendre quadrature nodes and weights 
    # n nodes/weights on interval a, b

    xi, ωi = gausslegendre(n)                   # n nodes/weights on [-1, 1]
    x_new = (xi .+ 1.0) * (b - a) / 2.0 .+ a    # change interval 
    return x_new, ωi

end


function gauss_leg(f, n, a, b)
    # Uses Gauss-Legendre quadrature with n nodes over [a,b]
    # Upshot: can be quicker
    # Downside: no idea how accurate the solution is (not adaptive)
    
    # Get nodes and weights
    xi, ωi = gausslegendre(n)
    
    # Compute approximation 
    x_new = (xi .+ 1) * (b - a) / 2.0 .+ a # change of variable
    approx = (b - a) / 2.0 * (ωi' * f.(x_new))
    return approx
    
end

function foc_k(x, U, μ, θ)

    c, wp, k = x
    ptilde = exp(-(1. - β) / ϵ * wp)
    phat = (θ * k) ^ (-1. / ϵ)
    pnext = ptilde * pbar

    # Force the next-period state to be in the domain
    if pnext < pL
        pnext = pL 
        # println("trimming low")
    elseif pnext > pH 
        pnext = pH
        # println("trimming high")
    end

    # FOCs
    focs = zeros(3)
    focs[1] = 1. - A0(pnext) * (1. - β) * exp((1. - β) * wp) / (β * R * c) - μ * k / (θ * c ^ 2)
    focs[2] = 1. - 1. / R * pbar * phat * θ + μ / (θ * c)
    focs[3] = log(c) + β * wp - U 

    # Jacobian
    dfocs = zeros(3, 3)
    gw = -A0'(pnext) * exp(2(1. - β)wp) * pbar * ((1. - β) ^ 2) / ϵ + 
        A0(pnext) * (1. - β) ^ 2 * exp((1. - β)wp)
    Gw = A0(pnext) * (1. - β) * exp((1. - β) * wp)
    
    dfocs[1, 1] = 1 / (β * R * c ^ 2) * Gw + 2 * μ * k / (θ * c ^ 3)
    dfocs[1, 2] = -1 / (β * R * c) * gw 
    dfocs[1, 3] = -μ / (θ * c ^ 2)

    dfocs[2, 1] = -μ / (θ * c ^ 2)
    dfocs[2, 3] = 1 / (R * ϵ) * pbar * θ ^ (1. - 1. / ϵ) * k ^ (-1. / ϵ - 1.)

    dfocs[3, 1] = 1. / c
    dfocs[3, 2] = β

    return focs, dfocs

end

function newton_k(U, μ, θ, x0)
    mxit = 500
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0
    
    while diff > tol && its < mxit
        focs, dfocs = foc_k(x0, U, μ, θ)
        diff = norm(focs, Inf)
        d = dfocs \ focs 
        while minimum(x0[[1 3]] - d[[1 3]]) .< 0.0 # c0 and k need to be positive, w need not be
            d = d / 2.0
            if maximum(d) < tol
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

function clear_focs()
    c = zeros(nt)
    k = similar(c)
    wp = similar(c)
    for i in 1:nt
        xx = [c0_lf[i, ipr] wp_lf[i, ipr] klf[i, ipr]]'
        x, ~ = newton_k(U, mu, th, xx)
        c[i], wp[i], k[i] = x
        println(i)

    end
end