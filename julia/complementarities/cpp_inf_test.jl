#= 

Testing in the infinite-horizon component planner's problem 

TODO: try different initial guess for newton
in particular, can we use the allocations from the prior iteration as SG?

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles, ApproxFun

gr()
println("******** cpp_inf_test.jl ********")

# Parameters
β = 0.8         # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

solnpath = "julia/complementarities/results/"
m_cheb = 5

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

# One-update allocations 
# solnpath = "julia/complementarities/results/"
# cg = readdlm(solnpath * "csol.txt")
# kg = readdlm(solnpath * "ksol.txt")
# wg = readdlm(solnpath * "wsol.txt")
# Ug = readdlm(solnpath * "Usol.txt")
# mug = readdlm(solnpath * "musol.txt")

cg = repeat(c0_lf, 1, m_cheb)
kg = repeat(klf, 1, m_cheb)
wg = repeat(wp_lf, 1, m_cheb)

# Finite horizon
capT = Inf
shft = (1. - β) / (1. - β ^ (capT + 1))             # shifter for finite horizon, t = 0,1,...,T on allocations
shft_w = (1. - β ^ capT) / (1. - β ^ (capT + 1) )   # shifter on promise utility

# Price decomposition
p_tilde(x) = exp(-1. / ϵ * shft * shft_w * x)
p_hat(t,k) = (t * k) ^ (-1. / ϵ)

# Extrapolation function 
function extrap_pbar(pb; return_slp = false)
    # If p̄ > pH (only applies if this is the case!), we 
    # linearly extrapolate At near pH to get the right slope

    del = 0.001 # distance to determine slope
    slp = (A0(pH) - A0(pH - del)) / del
    ext_val = A0(pH) + slp * (pb - pH)

    if return_slp
        return ext_val, slp
    else
        return ext_val
    end
     
end

# tf = extrap_pbar(2.2, return_slp = true)

# function Afull(x)
#     if x>pH 
#         return extrap_pbar(x)
#     else
#         return A0(x)
#     end

# end

# extr = range(1., 3., length = 100)
# plot(extr, Afull.(extr))

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

function find_bracket(f; bkt0 = (-0.3, -0.2), step = 0.1, print_every = false)
    # Finds the bracket (x, x + step) containing the zero of 
    # the function f
    
    a, b = bkt0
    its = 0
    mxit = 10_000

    if print_every
        println([a, f(a), b, f(b)])
    end
    
    while f(a) * f(b) > 0.0 && its < mxit
        if f(a) > 0.0           # f(a), f(b) positive
            if f(a) > f(b)      # f decreasing
                a = copy(b)
                b += step
            else                # f increasing
                b = copy(a)
                a -= step
            end
        else                    # f(a), f(b) negative
            if f(a) > f(b)      # f increasing
                b = copy(a)
                a -= step
            else                # f decreasing
                a = copy(b)
                b += step
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
        # println("pnext = $pnext")
        Apn = extrap_pbar(pnext)
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

function foc_k_a(x, U, μ, θ)

    c, wp, k = x
    ptilde = p_tilde(wp)
    phat = (θ * k) ^ (-1. / ϵ)
    pnext = ptilde * pbar

    # Extrapolate if next-period state leaves the domain
    if pnext > pH
        # If we're above the bound, A(p̄′) is the interpolated value, A′(p̄′) is just the slope
        # println("using extrap")
        Apn, Apn_p = extrap_pbar(pnext, return_slp = true) 
    else
        Apn = A0(pnext)
        Apn_p = A0'(pnext)
    end

    # FOC vector 
    focs = zeros(3)
    focs[1] = 1. - Apn * shft * exp(shft * wp) / (β * R * c) - μ * k / (θ * c ^ 2)
    focs[2] = 1. - 1. / R * pbar * phat * θ + μ / (θ * c)
    focs[3] = log(c) + β * wp - U 

    # Jacobian
    Gw = Apn * shft * exp(shft * wp)
    gw = Apn * (shft ^ 2) * exp(shft * wp) - 
        shft ^ 2 / ϵ * pbar * Apn_p * exp(shft * (1. - 1. / ϵ) * wp)

    dfocs = zeros(3, 3)
    dfocs[1, 1] = Gw / (β * R * c ^ 2) + 2μ * k / (θ * c ^ 3)
    dfocs[1, 2] = -gw / (β * R * c) 
    dfocs[1, 3] = -μ / (θ * c ^ 2)

    dfocs[2, 1] = -μ / (θ * c ^ 2)
    dfocs[2, 3] = 1. / (R * ϵ) * pbar * θ ^ (1. - 1. / ϵ) * k ^ (-1. / ϵ - 1.)

    dfocs[3, 1] = 1. / c
    dfocs[3, 2] = β

    return focs, dfocs

end

# i = 1
# U = Ulf[i]
# mu = μlf[i]
# x0 = [cg[i, ip] wg[i, ip] kg[i, ip]]'
# th = tgrid[i]

# # focs = foc_k(x0, U, mu, th) 
# # dfocs = ForwardDiff.jacobian(x -> foc_k(x, U, mu, th), x0)

# focsa, dfocsa = foc_k_a(x0, U, mu, th)

# x, f = newton_k(U, mu, th, x0)
# x, f = newton_k(U, mu, th, x0, nd = 0.1)

# x00 = [cg[i, ip]; wg[i, ip]; kg[i, ip]]
# ifocs(x) = foc_k(x0, U, mu, th)
# nlsolve(ifocs, x0, method = :newton)

function newton_k(U, μ, θ, x0; nd = 1)
    
    mxit = 500
    tol = 1e-10
    diff = 10.0
    its = 0
    fail = 0
    
    while diff > tol && its < mxit
        # focs = foc_k(x0, U, μ, θ)
        # dfocs = ForwardDiff.jacobian(x -> foc_k(x, U, μ, θ), x0)
        focs, dfocs = foc_k_a(x0, U, μ, θ)
        diff = norm(focs, Inf)
        d = dfocs \ focs 
        while minimum(x0[[1 3]] - d[[1 3]]) .< 0.0 # c0 and k need to be positive, w need not be
            d = d / 2.0
            if maximum(d) < tol
                fail = 1
                println("warning: newton failed")
                println(x0 - d)
                break
            end
        end
        if fail == 1
            break
        end
        
        x0 = x0 - d * nd
        # println(x0)
        its += 1
        # println(its)
        
    end
    return x0, fail
    
end

function alloc_single(U, μ, t)
    
    it = findmin(abs.(tgrid .- t))[2]       # nearest θ value on tgrid
    x0 = [cg[it, ip] wg[it, ip] kg[it, ip]]'

    x, f = newton_k(U, μ, t, x0)
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
        Apn = extrap_pbar(pnext)
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
pH = 0.5

R = 1.5

# Build space for pbar
S0 = Chebyshev(pL..pH)
p0 = points(S0, m_cheb)

# Updated A0
# at1 = vec(readdlm(solnpath * "a_t1.txt"))
# at2 = vec(readdlm(solnpath * "a_t2.txt"))
# at3 = vec(readdlm(solnpath * "a_t3.txt"))
# at7 = vec(readdlm(solnpath * "a_t7.txt"))
# at2_pfi = vec(readdlm(solnpath * "a_t2_pfi.txt"))
# at4_pfi = vec(readdlm(solnpath * "a_t4_pfi.txt"))
# at5_pfi = vec(readdlm(solnpath * "a_t5_pfi.txt"))

at0 = ones(m_cheb)

# # Convolutions
# ω = 0.3
# a1 = ω * at1 + (1. - ω) * ones(m_cheb)
# a2 = ω * at2 + (1. - ω) * a1
# a3 = ω * at3 + (1. - ω) * a2

A0 = Fun(S0, ApproxFun.transform(S0, at0)) # this is just 1 everywhere

# pt = range(pL, pH, length = 100)
# display( plot(pt, A0.(pt)) )

# Testing γ

# A0 case (low β, low pH)
ip = 1 # γ*∈[ 0.238, 0.239] bkt = (-3.03, -3.02)
pbar = p0[ip]


γ = 0.239
println("γ = $γ")

# test_ums = range(-10., -0., length = 101)
# testvals = [test_shoot(test_ums[i]) for i in 1:length(test_ums)]
# display(plot(test_ums, testvals))

bkt = (-3.03, -3.02)
stp = round(bkt[2] - bkt[1], digits = 5)
bkt = find_bracket(um -> tax_shoot(um), bkt0 = bkt, step = stp) 
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
#         x0 = [cg[i, ip] wg[i, ip] kg[i, ip]]'
#         x, ~ = newton_k(U, mu, tgrid[i], x0)
#         c[i], wp[i], k[i] = x
#         # println(i)

#     end

#     return c, k, wp
# end

# c_test, k_test, wp_test = clear_focs();
# sum(isnan.(c_test))
# plot(tgrid, c_test)
# plot(tgrid, wp_test)
# plot(tgrid, k_test)
