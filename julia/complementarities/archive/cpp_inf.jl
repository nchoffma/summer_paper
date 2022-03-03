#= 

Solves the component planner's problem with w = 0, for p̄∈P

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles, ApproxFun

gr()
println("******** cpp_inf_horizon.jl ********")

# Parameters
const β = 0.95        # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.
R = 1.5

# State space for p̄
pL = 0.01
pH = 5.
m_cheb = 10

# LF allocations, as initial guess for Newton 
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 0.5 * ones(nt)
klf = 0.5 * ones(nt)
c1_lf = tgrid .* klf
wp_lf = log.(c1_lf) ./ (1. - β)

# Distribution for output 
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

    # # Truncated normal
    # mu = (H + L) / 2.
    # sigma = 0.5
    # fx = TruncatedNormal(mu, sigma, L, H)
    # cdf = cdf(fx, x)
    # pdf = pdf(fx, x)
    # fpx = ForwardDiff.derivative(y -> pdf(fx, y), x)
    
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

function cheb_approx(fvals, a, b)
    # Approximates the function defined by fvals on m_cheb nodes
    # over interval [a,b] using Chebyshev polynomials of order n_cheb
    # Uses the default order of n = m - 1, which is fine 

    m_cheb = length(fvals)
    S = Chebyshev(a..b)
    p = points(S, m_cheb)
    fx = Fun(S, ApproxFun.transform(S, fvals))
    return fx

end

# f(x) = x ^ (1. / 3.)
# S = Chebyshev(0..2)
# p = points(S, 10)
# fvs = f.(p)
# fhat = cheb_approx(fvs, 0, 2)
# fp = ForwardDiff.derivative(f, 1)
# fhatp = ForwardDiff.derivative(fhat, 1)

function find_bracket(f; bkt0 = (-0.5, -0.4))
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

function full_solve(γ, A0, pbar)
    
    # Given a guess at the funciton A0 = Ã₀, solves the model at pbar
    
    function foc_k(x, U, μ, θ)
        
        c, wp, k = x
        ptilde = exp(-(1. - β) / ϵ * wp)
        phat = (θ * k) ^ (-1. / ϵ)

        # FOC vector
        focs = zeros(eltype(x), 3)
        focs[1] = 1. + A0(pbar * ptilde) * (1. - β) * exp((1. - β)w) - μ * k / (θ * c ^ 2)
        focs[2] = 1. - 1. / R * pbar * phat * θ + μ / (θ * c)
        focs[3] = log(c) + β * wp - U 
        
        return focs
        
    end
    
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
        ptilde = exp(-(1. - β) / ϵ * wp)
        phat = (θ * k) ^ (-1. / ϵ)
        ~, ft, fpt = fdist(t)
        du[1] = k / (c0 * t)
        du[2] = γ - A0(pbar * ptilde) / (β * R) * (1. - β) * exp((1. - β) * wp) - fpt / ft * μ
    end
    
    function tax_shoot(U_0)
        u0 = [U_0 0.0]
        tspan = (θ_min, θ_max)
        prob = ODEProblem(de_system!, u0, tspan)
        gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
        return gpath.u[end][2] # want to get μ(θ̲) = 0
        
    end
    
    # Optimal allocations along the solution 
    function opt_allocs(gpath)
        c0p = zeros(nt)
        kp = zeros(nt)
        wpp = zeros(nt)
        
        for i in 1:nt
            t = tgrid[i]
            U, mu = gpath(t)
            c0p[i], wpp[i], kp[i] = alloc_single(U, mu, t)
        end
        
        return c0p, c1p, kp
    end
    
    # Integrands
    
    # Promise-keeping (U* = 0)
    function pkc_integrand(t, gpath)
        U, mu = gpath(t)
        Ft, ft, fpt = fdist(t)
        return U * ft
    end
    
    # Total cost 
    function cost_integrand(t, gpath, pbar)
        U, μ = gpath(t)
        Ft, ft, fpt = fdist(t)
        c0, wp, k = alloc_single(U, μ, t)
        ptilde = exp(-(1. - β) / ϵ * wp)
        phat = (θ * k) ^ (-1. / ϵ)

        ct = (c0 + k + (1. / R) * (A0(pbar * ptilde) * 
            exp((1. - β) * wp) - pbar * phat * t * k)) * ft 
        return ct
    end
    
    # Solve using shooting
    bkt = find_bracket(um -> tax_shoot(um), bkt0 = (-0.5, -0.4))
    Umin_opt = find_zero(x -> tax_shoot(x), bkt) 
    u0 = [Umin_opt, 0.0]
    p = (Y_0)
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan, p)
    gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
    c0p, wpp, kp = opt_allocs(gpath)
    pkc = gauss_leg(t -> pkc_integrand(t, gpath), 20, θ_min, θ_max); 

    return gpath, c0p, wpp, kp, pkc
end

function clear_pkc(γ, A0, pbar)
    yopt, gpath, c0p, c1p, kp, pkc = full_solve(γ, A0, pbar)
    return pkc # want to zero this out 
end

# Build initial guess 
S0 = Chebyshev(pL..pH)
p0 = points(S0, m_cheb)
a0_pts = 1. ./ p0
A0 = Fun(S, ApproxFun.transform(S0, a0_pts))

clear_pkc(2., A0, p0[1])