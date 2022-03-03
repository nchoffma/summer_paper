#= 

Testing in the finite-horizon (T>2) component planner's problem 

Using homogeneity result, can just solve for w=0 in each period

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles, ApproxFun

gr()
println("******** comp_finite_vf_test.jl ********")

# Parameters
const β = 0.95        # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

# Autaurky allocations
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 1. / (1. + β) * w * ones(nt)
klf = β / (1. + β) * w * ones(nt)
c1_lf = tgrid .* klf
wp_lf = log.(c1_lf)
Ulf = log.(c0_lf) + β * wp_lf
μlf = 0.05 * ones(nt)
μlf[[1 end]] .= 0. 

# Finite horizon
capT = 2
shft = (1. - β) / (1. - β ^ (capT + 1))             # shifter for finite horizon, t = 0,1,...,T on allocations
shft_w = (1. - β ^ capT) / (1. - β ^ (capT + 1) )   # shifter on promise utility

# Price decomposition
p_tilde(x) = exp(-1. / ϵ * shft * shft_w * x)
p_hat(t,k) = (t * k) ^ (-1. / ϵ)

# Extrapolation function 
function extrap_pbar(pb)
    # If p̄ > pH (only applies if this is the case!), we 
    # linearly extrapolate At near pH to get the right slope

    del = 0.1 # distance to determine slope
    slp = (A0(pH) - A0(pH - del)) / del
    ext_val = A0(pH) + slp * (pb - pH)
    return ext_val 
end

# A0(pH)
# extrap_pbar(1.6)

# extrap_pbar(pH)

# Functions for truncated normal dist 
tmean = (θ_max + θ_min) / 2
tsig = 0.2
function tnorm_pdf(x)
    pdf.(truncated(Normal(tmean, tsig), θ_min, θ_max), x)
end

function tnorm_cdf(x) # for convenience
    cdf.(truncated(Normal(tmean, tsig), θ_min, θ_max), x)
end

function tnorm_fprime(x)
    ForwardDiff.derivative(tnorm_pdf, x)
end 

# Distribution for output 
function fdist(x)

    # Uniform
    L = θ_min
    H = θ_max 
    cdf = (x - L) / (H - L)
    pdf = 1. / (H - L)
    fpx = 0.

    # # Truncated normal
    # cdf = tnorm_cdf(x)
    # pdf = tnorm_pdf(x)
    # fpx = tnorm_fprime(x)
    
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

function find_bracket(f; bkt0 = (-0.3, -0.2))
    # Finds the bracket (x, x + 0.1) containing the zero of 
    # the function f
    
    a, b = bkt0
    its = 0
    mxit = 1000
    
    while f(a) * f(b) > 0.0 && its < mxit
        if f(a) > 0.0           # f(a), f(b) positive
            if f(a) > f(b)      # f decreasing
                a = copy(b)
                b += 0.1
            else                # f increasing
                b = copy(a)
                a -= 0.1
            end
        else                    # f(a), f(b) negative
            if f(a) > f(b)      # f increasing
                b = copy(a)
                a -= 0.1
            else                # f decreasing
                a = copy(b)
                b += 0.1
            end
        end
        its += 1
    end
    if its == mxit 
        println([a, f(a), b, f(b)])
        error("bracket not found")
    end
    return (a, b)
end


function foc_k(x, U, μ, θ)
        
    c, wp, k = x
    ptilde = p_tilde(wp)
    phat = (θ * k) ^ (-1. / ϵ)
    pnext = ptilde * pbar

    # Extrapolate if next-period state leaves the domain
    if pnext > pH 
        Apn = extrap_pbar(pnext)
        # Apn = A0(pH)
    else
        Apn = A0(pnext)
    end

    # FOC vector
    focs = zeros(eltype(x), 3)
    focs[1] = 1. - Apn * shft * exp(shft * wp) / (β * R * c) - μ * k / (θ * c ^ 2)
    focs[2] = 1. - 1. / R * pbar * phat * θ + μ / (θ * c)
    focs[3] = log(c) + β * wp - U 
    
    return focs
    
end

# i = 1
# U = Ulf[i]
# mu = μlf[i]
# x0 = [c0_lf[i] wp_lf[i] klf[i]]'
# th = tgrid[i]
# focs = foc_k(x0, U, mu, th)
# dfocs = ForwardDiff.jacobian(x -> foc_k(x, U, mu, th), x0)
# x, f = newton_k(U, mu, th, i)

function newton_k(U, μ, θ, i)
    x0 = [c0_lf[i] wp_lf[i] klf[i]]'
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
    
    x, f = newton_k(U, μ, t, it)
    c0, wp, k = x 
    return c0, wp, k
    
end

function de_system!(du, u, p, t)
    U, μ = u
    c0, wp, k = alloc_single(U, μ, t)
    ptilde = p_tilde(wp)
    phat = (t * k) ^ (-1. / ϵ)
    ~, ft, fpt = fdist(t)
    pnext = pbar * ptilde

    # Extrapolate if next-period state leaves the domain
    if pnext > pH 
        # println("pnext = $pnext")
        Apn = extrap_pbar(pnext)
        # Apn = A0(pH)
    else
        Apn = A0(pnext)
    end

    du[1] = k / (c0 * t)
    du[2] = γ - Apn / (β * R) * shft * exp(shft * wp) - fpt / ft * μ
end

function tax_shoot(U_0)
    u0 = [U_0 0.0]
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan)
    gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
    # gpath = solve(prob, alg_hints = [:stiff]) 
    return gpath.u[end][2] # want to get μ(θ̲) = 0
    
end

function test_shoot(Um)
    println("U0 = $Um")
    ep = try
        tax_shoot(Um)
    catch y # all the different ways it can fail 
        if isa(y, UndefVarError)
            NaN
        elseif isa(y, LAPACKException)
            NaN
        elseif isa(y, DomainError)
            NaN
        elseif isa(y, SingularException)
            NaN
        end
    end
    return ep
end

# State space for p̄
pL = 0.
pH = 1.2
m_cheb = 8

R = 1.6

# Build space for pbar
S0 = Chebyshev(pL..pH)
# p0 = reverse(points(S0, m_cheb))
p0 = points(S0, m_cheb)

# Initial Guess
a0_pts = readdlm("julia/complementarities/finite_horizon/results/a_t1.txt", '\t', Float64, '\n') 
A0 = Fun(S0, ApproxFun.transform(S0, a0_pts[:, 1]))

pt = range(pL, pH, length = 100)
display(plot(pt, A0.(pt)))

# A0(x) = x ^ 0.

# Testing γ

# Results for AT = 1 case
# ip = 1 # γ* ∈[0.34, 0.35] (unif) bkt0 = (-0.5, -0.4)
# ip = m_cheb # γ* ∈[0.33, 0.34] bkt0 = (-2.5, -2.4)
# ip = 2 # γ* ∈[0.34, 0.35] bkt0 = (-0.5, -0.4)
# ip = 3 # γ* ∈[0.34, 0.35] bkt0 = (-0.5, -0.4)
# ip = 4 # bkt0 = (-0.5, -0.4)
# ip = 5 # bkt0 = (-0.5, -0.4)
# ip = 6 # bkt0 = (-0.5, -0.4)
# ip = 7 # bkt0 = (-4.5, -4.4)

# ip = 1 # γ* ∈[0.514, 0.515] (trunc norm)


# Results for AT-1 (approx) case
# ip = 4 # γ ∈[0.38, 0.39] bkt0 = (-0.5, -0.4)
# ip = 3 # γ ∈[0.33, 0.34] bkt0 = (-0.5, -0.4)
# ip = 2 # γ ∈[0.27, 0.28] bkt0 = (-2.5, -2.4) (either works)
ip = 1 # γ ∈[0.22, 0.23] bkt0 = (-2.5, -2.4)
# ip = m_cheb # γ∈[0.4, 0.41] bkt0 = (-0.5, -0.4)
# Can change the bkt0, gamma bracket if pbar > 1
pbar = p0[ip] 

γ = 0.23

# test_ums = range(-10., 0., length = 21)
# testvals = [test_shoot(test_ums[i]) for i in 1:length(test_ums)]
# display(plot(test_ums, testvals))

bkt = find_bracket(um -> tax_shoot(um), bkt0 = (-2.5, -2.4)) 
Umin_opt = find_zero(x -> tax_shoot(x), bkt)

u0 = [Umin_opt 0.0]
tspan = (θ_min, θ_max)
prob = ODEProblem(de_system!, u0, tspan)
gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 

# Promise-keeping (U* = 0)
function pkc_integrand(t, gpath)
    U, mu = gpath(t)
    Ft, ft, fpt = fdist(t)
    return U * ft
end

pkc = gauss_leg(t -> pkc_integrand(t, gpath), 50, θ_min, θ_max)

# function clear_focs()
#     c = zeros(nt)
#     k = similar(c)
#     wp = similar(c)
#     for i in 1:nt
#         U = Ulf[i]
#         mu = μlf[i]
#         x, ~ = newton_k(U, mu, tgrid[i], i)
#         c[i], wp[i], k[i] = x
#         # println(i)

#     end

#     return c, k, wp
# end

# # c_test, k_test, wp_test = clear_focs(); 
# # sum(isnan.(c_test))
# # plot(tgrid, c_test)