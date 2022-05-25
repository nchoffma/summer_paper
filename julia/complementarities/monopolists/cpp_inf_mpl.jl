#= 

Solves the planner's infinite-horixon CMP, 
with θ∼N(μ,σ) and households as monopolists

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, NLsolve, 
    ForwardDiff, Optim, LaTeXStrings, Roots, 
    FastGaussQuadrature, Printf, DelimitedFiles, ApproxFun

gr()
println("******** cpp_inf_mpl.jl ********")

# Parameters
β = 0.9         # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

R = 1.1

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

# State space w 
wL = -1.
wH = 1. 

m_cheb = 5
S0 = Chebyshev(wL..wH)
wp0 = points(S0, m_cheb)

# Functions for truncated normal dist 
tmean = (θ_max + θ_min) / 2
tsig = 0.3
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

    # Truncated normal
    cdf = tnorm_cdf(x)
    pdf = tnorm_pdf(x)
    fpx = tnorm_fprime(x)
    
    return cdf, pdf, fpx
    
end

# Gauss-legendre quadrature
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
    mxit = 50_000

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
    Cp_wp = C0'(wp)

    # FOC vector
    focs = zeros(eltype(x), 3)
    focs[1] = 1. - Cp_wp / (β * R * c) - μ * k / (θ * c ^ 2)
    focs[2] = 1. - (ϵ - 1.) / (R * ϵ) * θ ^ (1. - 1. / ϵ) * k ^ (-1. / ϵ) + 
        μ / (θ * c)
    focs[3] = log(c) + β * wp - U 
    
    return focs
    
end

function newton_k(U, μ, θ, i)
    x0 = [c0_lf[i] wp_lf[i] klf[i]]'
    mxit = 500
    tol = 1e-10
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
    ~, ft, fpt = fdist(t)
    Cp_wp = C0'(wp)


    du[1] = k / (c0 * t)
    du[2] = γ - Cp_wp / (β * R) - fpt / ft * μ

end

function tax_shoot(U_0)
    u0 = [U_0 0.0]
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan)
    gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
    return gpath.u[end][2] # want to get μ(θ̲) = 0
    
end

# Promise-keeping (U* = 0)
function pkc_integrand(t, gpath)
    U, mu = gpath(t)
    Ft, ft, fpt = fdist(t)
    return U * ft
end

# Optimal allocations along the path
function opt_allocs(gpath)
    cp = zeros(nt)
    kp = similar(cp)
    wp = similar(cp)
    mup = similar(cp)
    
    for i in 1:nt
        U, mu = gpath(tgrid[i])
        cp[i], wp[i], kp[i] = alloc_single(U, mu, tgrid[i])
        mup[i] = mu

    end

    return cp, wp, kp, mup
end

# Value of C1
function C_integrand(t, gpath)
    
    U, mu = gpath(t)
    Ft, ft, fpt = fdist(t)
    ct, wpt, kt = alloc_single(U, mu, t)
    
    return ft * (ct + kt + 1 / R * (C0(wp) - (t * kt) ^ (1. - 1. / ϵ)) )

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

# tax_shoot(3.)

# Testing 
cpts0 = ones(m_cheb) 
C0 = Fun(S0, ApproxFun.transform(S0, cpts0))

ip = 1
wstar = wp0[ip]

γ = 1.
println("γ = $γ")

test_ums = range(-20., 20., length = 41)
testvals = [test_shoot(test_ums[i]) for i in 1:length(test_ums)]
display(plot(test_ums, testvals))