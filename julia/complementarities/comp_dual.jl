#= 
Solving the model with complementarities- DUAL PROBLEM
=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, Printf, DelimitedFiles

gr()

println("******** comp_dual.jl ********")

# Parameters
const β = 0.95        # discounting
θ_min = 1.
θ_max = 2.

w = 1.
ϵ = 4.
R = 1.5 # near value from primal 

# LF allocations, as initial guess for Newton 
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 0.5 * ones(nt)
klf = 0.5 * ones(nt)
c1_lf = tgrid .* klf

# Distribution for output
# Note: this shooting method assumes that mu(theta_max) = 0, which 
# requires that the distribution be bounded 
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

function full_solve(γ)
    # Given multipliers lam = [R, γ], solves the full model 
    # and checks PKC

    # This function contains many sub-functions, which are built using 
    # R and γ
    
    @printf "\nγ = %.8f\n" γ
    
    function foc_k(x, U, μ, θ, Y)
        
        c0, c1, k = x
        
        # FOC vector
        focs = zeros(3)
        focs[1] = c1 / (β * R * c0) + k / (θ * c0 ^ 2) * μ - 1.
        focs[2] = (1. / R) * (Y / k * θ ^ (ϵ - 1)) ^ (1.0 / ϵ) - μ / (θ * c0) - 1.
        focs[3] = log(c0) + β * log(c1) - U 
        
        # Jacobian matrix
        dfocs = zeros(3, 3)
        dfocs[1, 1] = -c1 / (β * R * c0 ^ 2) - 2k * μ / (θ * c0 ^ 3)
        dfocs[1, 2] = 1.0 / (β * R * c0)
        dfocs[1, 3] = μ / (θ * c0 ^ 2)
        
        dfocs[2, 1] = μ / (θ * c0 ^ 2)
        dfocs[2, 3] = (-1. / ϵ) * (1. / R) * θ ^ (1. - 1. / ϵ) * Y ^ (1. / ϵ) * k ^ (-1. / ϵ - 1.)
        
        dfocs[3, 1] = 1.0 / c0
        dfocs[3, 2] = β / c1 
        
        return focs, dfocs
        
    end
    
    function newton_k(U, μ, θ, i, Y)
        x0 = [c0_lf[i] c1_lf[i] klf[i]]'
        mxit = 500
        tol = 1e-6
        diff = 10.0
        its = 0
        fail = 0
        
        while diff > tol && its < mxit
            focs, dfocs = foc_k(x0, U, μ, θ, Y)
            diff = norm(focs, Inf)
            d = dfocs \ focs 
            while minimum(x0 - d) .< 0.0
                d = d / 2.0
                if maximum(d) < tol
                    fail = 1
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
        c0, c1, k = x 
        return c0, c1, k
        
    end
    
    function de_system!(du, u, p, t)
        U, μ = u
        Y = p[1]
        c0, c1, k = alloc_single(U, μ, t, Y)
        
        ~, ft, fpt = fdist(t)
        du[1] = k / (c0 * t)
        du[2] = γ - c1 / (β * R) - fpt / ft * μ
    end
    
    function tax_shoot(U_0, Y)
        u0 = [U_0 0.0]
        p = (Y)
        tspan = (θ_min, θ_max)
        prob = ODEProblem(de_system!, u0, tspan, p)
        gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
        return gpath.u[end][2] # want to get μ(θ̲) = 0
        
    end
    
    # Optimal allocations along the solution 
    function opt_allocs(gpath, Y)
        c0p = zeros(nt)
        kp = zeros(nt)
        c1p = zeros(nt)
        
        for i in 1:nt
            t = tgrid[i]
            U, mu = gpath(t)
            c0p[i], c1p[i], kp[i] = alloc_single(U, mu, t, Y)
        end
        
        return c0p, c1p, kp
    end
    
    # Integrands
    # CES aggregator
    function ces_integrand(t, gpath, Y)
        # CES aggregator for output
        # gpath is the solution to the ODE system at Y 
        U, μ = gpath(t)
        ~, ~, k = alloc_single(U, μ, t, Y)
        ~, ft, ~ = fdist(t)
        return (t * k) ^ ((ϵ - 1.) / ϵ) * ft
        
    end

    # Promise-keeping (U* = 0)
    function pkc_integrand(t, gpath)
        U, mu = gpath(t)
        Ft, ft, fpt = fdist(t)
        return U * ft
    end
    
    function solve_model(Y_0)
        # Solves the model, given starting guess for Y 
        
        Y_1 = copy(Y_0)
        mxit = 250
        its = 1
        diff = 10.0
        tol = 1e-5
        bkt_init = (-2.3, -2.2)
        gpath = 0.                 # may not be the most efficient way to initialize, but works
        Umin_opt = 0. 
        
        print("\nSolving for Y \n")
        print("-----------------------------\n")
        print(" its     diff         y1\n")
        print("-----------------------------\n")
        
        while diff > tol && its < mxit
            
            bkt = find_bracket(um -> tax_shoot(um, Y_0), bkt0 = bkt_init)
            bkt_init = bkt
            Umin_opt = find_zero(x -> tax_shoot(x, Y_0), bkt) 
            u0 = [Umin_opt, 0.0]
            p = (Y_0)
            tspan = (θ_min, θ_max)
            prob = ODEProblem(de_system!, u0, tspan, p)
            gpath = solve(prob, alg_hints = [:stiff]) 
            Y_1 = gauss_leg(t -> ces_integrand(t, gpath, Y_0), 20, θ_min, θ_max) ^ (ϵ / (ϵ - 1.))
            diff = abs(Y_0 - Y_1)
    
            if mod(its, 20) == 0
                @printf("%3d %12.8f %12.8f\n", its, diff, Y_1)
            end
            if diff > 0.01
                    ω = 1.2 # a bit of acceleration, may make it unstable
                else
                    ω = 1.0
                end
            Y_0 = ω * Y_1 + (1. - ω) * Y_0
            its += 1
        end
        
        return Y_1, gpath, Umin_opt
    end
    
    y0 = 5.
    ts = @elapsed yopt, gpath, U0 = solve_model(y0) 
    c0p, c1p, kp = opt_allocs(gpath, yopt)
    pkc = gauss_leg(t -> pkc_integrand(t, gpath), 20, θ_min, θ_max);
    
    @printf "\nModel solved in %.3f seconds\n" ts
    @printf "∫UdF       = %5f\n" pkc
    
    return yopt, gpath, c0p, c1p, kp, pkc
end

γ0 = 0.8
# yopt, gpath, c0p, c1p, kp, pkc = full_solve(γ0)

function clear_pkc(γ)
    yopt, gpath, c0p, c1p, kp, pkc = full_solve(γ)
    return pkc # want to zero this out 
end

# clear_pkc(0.85)

γ_star = find_zero(clear_pkc, (0.8, 0.85))

# Plot results
yopt, gpath, c0, c1, k, pkc = full_solve(γ_star)
ppath = plot(gpath, label = [L"U(\theta)" L"\mu(\theta)"],
    title = "DE Solution", xlab = L"\theta", 
    legend = :bottomright)
pt0 = plot(tgrid, [c0 k w .- c0 .- k], 
    label = [L"c_0(\theta)" L"k(\theta)" L"b(\theta)"],
    title = "t = 0", xlab = L"\theta", legend = :bottomright)
pt1 = plot(tgrid, [c1 tgrid .* k],
    label = [L"c_1(\theta)" L"\theta k(\theta)"],
    title = "t = 1", xlab = L"\theta", legend = :topleft)
palloc = plot(pt0, pt1)
display(plot(palloc, ppath, layout = (2,1)))