#= 

Computes the dual in the infinite-horizon case, with w=0

Once we have these in hand, calculating value and policy functions 
for any w is straightforward. 
    
    The steps are:
    1) Given A, solve for γ* as in the dual 
    2) Check that A = ...
    3) Update, etc. 
    
    Allocation vector is x = [c0 wp k]
    
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

R = 1.55    # Higher if Y explodes
ϵ = 4.

# LF allocations, as initial guess for Newton 
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 0.5 * ones(nt)
klf = 0.5 * ones(nt)
c1_lf = tgrid .* klf
wp_lf = log.(c1_lf) ./ (1. - β)
Ulf = log.(c0_lf) + β .* wp_lf
mu_lf = 0.1 .* ones(nt)
mu_lf[[1 end]] .= 0.

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

# Try a few values
function test_shoot(Um, Y)
    println("U0 = $Um")
    ep = try
        tax_shoot(Um, Y)
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

# A0 = 9.0 # Starting guess for A
# γ = 0.4
# Y0 = 3.
# tax_shoot(-0.1, Y0)

# test_ums = range(-20, 20, length = 41)
# testvals = [test_shoot(test_ums[i], Y0) for i in 1:length(test_ums)]
# display(plot(test_ums, testvals))

# bkt_init = (-0.1, 0.)
# bkt = find_bracket(um -> tax_shoot(um, Y0), bkt0 = bkt_init)
# Umin_opt = find_zero(x -> tax_shoot(x, Y0), bkt)
# u0 = [Umin_opt, 0.0]
# p = (Y0)
# tspan = (θ_min, θ_max)
# prob = ODEProblem(de_system!, u0, tspan, p)
# gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
# Y_1 = gauss_leg(t -> ces_integrand(t, gpath, Y0), 20, θ_min, θ_max) ^ (ϵ / (ϵ - 1.))

function full_solve(γ, A0, bkt_start)
    
    # Solves for Y and returns the PKC value (and allocations)
    
    function foc_k(x, U, μ, θ, Y)
        
        c0, wp, k = x
        
        # FOC vector
        focs = zeros(3)
        focs[1] = A0 / (β * R * c0) * (1. - β) * exp((1. - β) * wp) + k / (θ * c0 ^ 2) * μ - 1.
        focs[2] = (1. / R) * (Y / k * θ ^ (ϵ - 1)) ^ (1.0 / ϵ) - μ / (θ * c0) - 1.
        focs[3] = log(c0) + β * wp - U 
        
        # Jacobian matrix
        dfocs = zeros(3, 3)
        dfocs[1, 1] = -A0 / (β * R * c0 ^ 2) * (1. - β) * exp((1. - β) * wp) - 2k * μ / (θ * c0 ^ 3)
        dfocs[1, 2] = A0 / (β * R * c0) * (1. - β) ^ 2 * exp((1. - β) * wp)
        dfocs[1, 3] = μ / (θ * c0 ^ 2)
        
        dfocs[2, 1] = μ / (θ * c0 ^ 2)
        dfocs[2, 3] = (-1. / ϵ) * (1. / R) * θ ^ (1. - 1. / ϵ) * Y ^ (1. / ϵ) * k ^ (-1. / ϵ - 1.)
        
        dfocs[3, 1] = 1.0 / c0
        dfocs[3, 2] = β
        
        return focs, dfocs
        
    end
    
    function newton_k(U, μ, θ, i, Y)
        x0 = [c0_lf[i] wp_lf[i] klf[i]]'
        mxit = 500
        tol = 1e-6
        diff = 10.0
        its = 0
        fail = 0
        
        while diff > tol && its < mxit
            focs, dfocs = foc_k(x0, U, μ, θ, Y)
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
    
    function alloc_single(U, μ, t, Y)
        
        it = findmin(abs.(tgrid .- t))[2] # for finding initial guess
        
        x, f = newton_k(U, μ, t, it, Y)
        c0, wp, k = x 
        return c0, wp, k
        
    end
    
    function de_system!(du, u, p, t)
        U, μ = u
        Y = p[1]
        c0, wp, k = alloc_single(U, μ, t, Y)
        
        ~, ft, fpt = fdist(t)
        du[1] = k / (c0 * t)
        du[2] = γ - A0 / (β * R) * (1. - β) * exp((1. - β) * wp) - fpt / ft * μ
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
    
    # Total cost 
    function cost_integrand(t, gpath, Y)
        U, μ = gpath(t)
        Ft, ft, fpt = fdist(t)
        c0, wp, k = alloc_single(U, μ, t, Y)
        pt = (Y / (t * k)) ^ (1. / ϵ)
        ct = (c0 + k + (1. / R) * (A0 * exp((1. - β) * wp) - pt * t * k)) * ft 
        return ct
    end
    
    # Solve for Y 
    function solve_model(Y_0, bkt_init = (-1.1, -1.))
        # Solves the model, given starting guess for Y 
        
        Y_1 = copy(Y_0)
        mxit = 1000
        its = 1
        diff = 10.0
        tol = 1e-5
        gpath = 0.
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
            if diff > 0.001
                ω = 1.2 # a bit of acceleration, may make it unstable
            else
                ω = 0.8
            end
            Y_0 = ω * Y_1 + (1. - ω) * Y_0
            its += 1
        end
        
        return Y_1, gpath, Umin_opt, bkt_init
    end
    
    ts = @elapsed yopt, gpath, Umin_opt, bkt = solve_model(Y0, bkt_start)
    c0p, wp, kp = opt_allocs(gpath, yopt)
    total_u = gauss_leg(t -> pkc_integrand(t, gpath), 20, θ_min, θ_max)
    
    @printf "\nModel solved in %.3f seconds\n" ts
    @printf "\n∫UdF = %5f\n" total_u
    
    return yopt, gpath, total_u, c0p, wp, kp, bkt, cost_integrand
    
end

# A0 = 9.0    # Starting guess for A
# A0 = 6.484733359932181  # A1, A0 = 9
# A0 = 4.749626875604206  # A1, A0 = 6.484733359932181
# A0 = 3.5333621539049584 # A1, A0 = 4.749626875604206
# A0 = 2.6677204453436887 # A1, A0 = 3.5333621539049584
# A0 = 2.042657145519835 # A1, A0 = 2.6677204453436887
# A0 = 1.5850650645335251 # A1, A0 = 2.042657145519835
# A0 = 1.2456786452574076 # A1, A0 = 1.5850650645335251
# A0 = 0.9908279250060863 # A1, A0 = 1.2456786452574076
A0 = 0.7971853543110302 # A1, A0 = 0.9908279250060863

# Y0 = 0.54
# bkt = (-3.9, -3.8) # Work for A0 = 6.484733359932181 (gamstar ≈ 0.237)

# Y0 = 0.41
# bkt = (-4.5, -4.4) # Work for A0 = 4.749626875604206 (gamstar ≈ 0.17667357525631125)
# Y0 = 0.305
# bkt = (-4.7, -4.6)  # Work for A0 = 3.5333621539049584 (gamstar = 0.1333902201652526)
# Y0 = 0.23
# bkt = (-4.8, -4.7)  # Work for A0 = 2.6677204453436887 (gamstar = 0.1333902201652526)
# Y0 = 0.18
# bkt = (-4.9, -4.8) # Work for A0 = 2.042657145519835 (gamstar = 0.07925570983886716)
# Y0 = 0.15
# bkt = (-4.9, -4.8) # Work for A0 = 1.5850650645335251 (gamstar = 0.0622858673095703)
# Y0 = 0.11
# bkt = (-5., -4.9) # Work for A0 = 1.5850650645335251 (gamstar = 0.04954291687011717)
# Y0 = 0.1
# bkt = (-5.1, -5.) # Work for A0 = A0 = 0.9908279250060863 (gamstar = 0.039860525512695304)

Y0 = 0.08
bkt = (-5.1, -5.)
yopt, gpath, total_u, c0p, wp, kp, bkt, ~ = full_solve(0.0324, A0, bkt);

function clear_pkc(γ, A0, bkt)
    yopt, gpath, total_u, c0p, wp, kp, bkt, ~ = full_solve(γ, A0, bkt)
    return total_u # want to zero this out
end

gamtol = 1e-8
fs = @elapsed gamstar = find_zero(γ -> clear_pkc(γ, A0, bkt), (0.0324, 0.0325),
    atol = gamtol, rtol = gamtol, xatol = gamtol, xrtol = gamtol, Bisection()); # 

yopt, gpath, total_u, c0p, wp, kp, bkt, costfxn = full_solve(gamstar, A0, bkt);
A1 = gauss_leg(t -> costfxn(t, gpath, yopt), 20, θ_min, θ_max)

# function iterate_A(A0)
#     # Goal: Solve for A using contraction mapping
    
#     # Setup and tolerances
#     A1 = copy(A0)
#     mxit = 100
#     diff = 10.
#     tol = 1e-5
#     gpath = 0.
#     Umin_opt = 0.

#     for its in 1:mxit
        


#     end

#     println("\nA0 = $A0\n")

# end