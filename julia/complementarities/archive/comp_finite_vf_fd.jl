#= 

Solves the dynamic (T<∞) case using the Finite Element Method
This code uses the ForwardDiff package to get the derivatives
in the Newton step, which ensures accuracy but is slow. 

Note: the syntax A0' works for the derivative of the Chebyshev approximation

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles, ApproxFun

gr()
println("******** comp_finite_vf_fd.jl ********")

# Parameters
const β = 0.95        # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

# LF allocations, as initial guess for Newton 
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 0.5 * ones(nt)
klf = 0.5 * ones(nt)
c1_lf = tgrid .* klf
wp_lf = log.(c1_lf) ./ (1. - β)
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

    # FOC vector
    focs = zeros(eltype(x), 3)
    # focs[1] = 1. - A0(pnext) * (1. - β) * exp((1. - β) * wp) / (β * R * c) - μ * k / (θ * c ^ 2)
    # focs[2] = 1. - 1. / R * pbar * phat * θ + μ / (θ * c)
    # focs[3] = log(c) + β * wp - U 
    focs = [1. - A0(pnext) * (1. - β) * exp((1. - β) * wp) / (β * R * c) - μ * k / (θ * c ^ 2), 
        1. - 1. / R * pbar * phat * θ + μ / (θ * c),
        log(c) + β * wp - U ]
    
    return focs
    
end

function newton_k(U, μ, θ, x0)
    
    mxit = 500
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0
    
    while diff > tol && its < mxit
        focs = foc_k(x0, U, μ, θ)
        dfocs = ForwardDiff.jacobian(x -> foc_k(x, U, μ, θ), x0)
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

function alloc_single(U, μ, t)
    
    it = findmin(abs.(tgrid .- t))[2] # for finding initial guess
    x0 = [c0_lf[it] wp_lf[it] klf[it]]'

    x, f = newton_k(U, μ, t, x0)
    c0, wp, k = x 
    return c0, wp, k
    
end

function qgausl(n, a, b)
    # Gauss-Legendre quadrature nodes and weights 
    # n nodes/weights on interval a, b

    xi, ωi = gausslegendre(n)                   # n nodes/weights on [-1, 1]
    x_new = (xi .+ 1.0) * (b - a) / 2.0 .+ a    # change interval 
    return x_new, ωi

end

function fem_resids!(INTR, x0)
    # Takes in pre-allocated residuals and guess x0,
    # returns new residuals

    a0 = x0[1:nt]
    b0 = x0[nt + 1:end]

    for n in 1:ne

        # Get theta values for interval 
        x1 = tgrid[n]
        x2 = tgrid[n + 1]

        # Get nodes and weights 
        thq, wq = qgausl(nq, x1, x2)
        for i in 1:nq
            
            # Get weights, lengths, etc.
            th = thq[i]
            wth = wq[i]
            delta_n = x2 - x1 
            ep = 2.0 * (th - x1) / delta_n - 1.0
            bs1 = 0.5 * (1.0 - ep)
            bs2 = 0.5 * (1.0 + ep)

            # Approximations and allocations
            U = a0[n] * bs1 + a0[n + 1] * bs2
            mu = b0[n] * bs1 + b0[n + 1] * bs2
            Upr = (a0[n + 1] - a0[n]) / delta_n
            mupr = (b0[n + 1] - b0[n]) / delta_n
            xx = [c0_lf[n] wp_lf[n] klf[n]]'
            x, ~ = newton_k(U, mu, th, xx)
            c0, wp, k = x

            # Evauluating and updating residuals 
            ~, ft, fpt = fdist(th)
            FU = Upr - k / (th * c0)
            ptilde = exp(-(1. - β) / ϵ * wp)
            mu_a = γ - A0(pbar * ptilde) / (β * R) * (1. - β) * exp((1. - β) * wp) - fpt / ft * mu
            Fmu = mupr - mu_a

            INTR[n] += bs1 * wth * FU
            INTR[n + 1] += bs2 * wth * FU
            INTR[nt + n] += bs1 * wth * Fmu
            INTR[nt + n + 1] += bs2 * wth * Fmu

        end    
    end
    return INTR
end

# State space for p̄
pL = 0.1
pH = 1.2
m_cheb = 6

R = 1.6

# Build initial guess 
S0 = Chebyshev(pL..pH)
p0 = points(S0, m_cheb)

# Initial Guess
a0_pts = ones(m_cheb) 
A0 = Fun(S0, ApproxFun.transform(S0, a0_pts))

# Testing γ
ip = 4
pbar = p0[ip] 
γ = 0.3

# x0 = [Ulf; μlf]
# resids = zeros(2nt)
# errs_2 = fem_resids!(resids, x0)
# derrs_a = ForwardDiff.jacobian(x -> fem_resids!(zeros(eltype(x), 2nt), x), x0) 
# dstep = derrs_a \ errs_2

function fem_newton(; ω = 1.)
    # Solves the ODE system, using the finite element method 
    # ω is the dampening/acceleration parameter 

    a0 = Ulf
    b0 = μlf

    tol = 1e-8
    diff = 10.0
    its = 0
    mxit = 100

    print(" FEM Progress \n")
    print("----------------\n")
    print(" its        diff\n")
    print("----------------\n")

    while diff > tol && its < mxit
        x0 = [a0; b0]
        resids = zeros(2nt)
        INTR = fem_resids!(resids, x0)
        dINTR = ForwardDiff.jacobian(x -> fem_resids!(zeros(eltype(x), 2nt), x), x0)
        diff = norm(INTR, Inf)
        its += 1

        # Newton update
        dstep = dINTR \ INTR
        a0 = a0 - dstep[1:nt] * ω
        b0 = b0 - dstep[nt + 1:end] * ω
        b0 = abs.(b0)                           # enforce positivity (dual)
        b0[[1 end]] .= 0.                       # enforce boundary conditions

        # Display progress 
        if mod(its, 1) == 0
            @printf("%2d %12.8f\n", its, diff) 
        end

    end

    return a0, b0, diff 

end

a0, b0, diff = fem_newton(ω = 0.4)