#= 
Solving the model with complementarities
=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, Printf, DelimitedFiles

gr()

println("******** comp_mod.jl ********")

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

function full_solve(lam)
    # Given multipliers lam = λ_0, λ_1, solves the full model 
    # and checks feasibility constraints 

    # This function contains many sub-functions, which are built using 
    # λ_0 and λ_1 
    
    λ_0, λ_1 = lam 
    R = λ_0 / λ_1
    @printf "\nλ_0 = %.8f\nλ_1 = %.8f\nR   = %.4f\n" λ_0 λ_1 R
    
    function foc_k(x, U, μ, θ, Y)
        
        c0, c1, k = x
        
        # FOC vector
        focs = zeros(3)
        focs[1] = c1 / (β * c0) - k / (λ_1 * θ * c0 ^ 2) * μ - R 
        focs[2] = (Y / k * θ ^ (ϵ - 1)) ^ (1.0 / ϵ) + μ / (λ_1 * θ * c0) - R 
        focs[3] = log(c0) + β * log(c1) - U 
        
        # Jacobian matrix
        dfocs = zeros(3, 3)
        dfocs[1, 1] = -c1 / (β * c0 ^ 2) + 2k * μ / (λ_1 * θ * c0 ^ 3)
        dfocs[1, 2] = 1.0 / (β * c0)
        dfocs[1, 3] = -μ / (λ_1 * θ * c0 ^ 2)
        
        dfocs[2, 1] = -μ / (λ_1 * θ * c0 ^ 2)
        dfocs[2, 3] = - 1.0 / ϵ * (Y * θ ^ (ϵ - 1)) ^ (1.0 / ϵ) * k ^ (-1.0 / ϵ - 1.0)
        
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
        du[2] = λ_1 * c1 / β - μ * fpt / ft - 1
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
        return (t * k) ^ ((ϵ - 1) / ϵ) * ft
        
    end
    
    # BC0
    function bc_integrand_0(t, gpath, Y)
        U, mu = gpath(t)
        c0, c1, k = alloc_single(U, mu, t, Y)
        Ft, ft, fpt = fdist(t)
        return (c0 + k) * ft
    end
    
    # BC1
    function bc_integrand_1(t, gpath, Y)
        U, mu = gpath(t)
        c0, c1, k = alloc_single(U, mu, t, Y)
        Ft, ft, fpt = fdist(t)
        return c1 * ft
    end
    
    function solve_model(Y_0)
        # Solves the model, given starting guess for Y 
        
        Y_1 = copy(Y_0)
        mxit = 250
        its = 1
        diff = 10.0
        tol = 1e-5
        bkt_init = (-1.5, -1.4)
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
    
    y0 = 1.
    ts = @elapsed yopt, gpath, U0 = solve_model(y0) 
    c0p, c1p, kp = opt_allocs(gpath, yopt)
    bc0 = gauss_leg(t -> bc_integrand_0(t, gpath, yopt), 20, θ_min, θ_max) - w;
    bc1 = gauss_leg(t -> bc_integrand_1(t, gpath, yopt), 20, θ_min, θ_max) - yopt;
    
    @printf "\nModel solved in %.3f seconds\n" ts
    @printf "\n∫[c₀ + k]dF - w = %5f\n" bc0 
    @printf "∫c₁dF - Y       = %5f\n" bc1
    
    return yopt, gpath, c0p, c1p, kp, bc0, bc1, R
end

# lam0 = [1.9486835977214818, 0.7258225833213333] # NLsolve improvement (ϵ = 4.0, θ_max = 4.0)
# lam0 = [1.948764904485699, 0.9278033236050093] # NLsolve improvement (ϵ = 4.0, θ_max = 3.0)
lam0 = [1.9491874662622317, 1.2727276699236132] # NLsolve improvement (ϵ = 4.0, θ_max = 2.0)

function clear_bcs(lam)
    # Takes in a vector lam = [λ_0, λ_1] and returns the feasibility violations
    yopt, gpath, c0, c1, k, bc0, bc1, R = full_solve(lam)
    return [bc0, bc1]
end

# Attempting to solve
total_t = @elapsed res = nlsolve(clear_bcs, lam0, ftol = 1e-6, xtol = 1e-8, iterations = 10_000)
lamstar = res.zero
yopt, gpath, c0, c1, k, bc0, bc1, R = full_solve(lamstar);

@printf "\nFull solve in %.3f seconds\n" total_t

# Wedges and plots 
# τ_k = 1. .- c1 ./ (c0 .* β) .* (ϵ / (ϵ - 1.)) .* (tgrid .^ ((1. - ϵ) / ϵ)) .* 
#     ((k ./ yopt) .^ (1. / ϵ)) # monopolists
τ_k = 1. .- c1 ./ (c0 .* β) .* (tgrid .^ ((1. - ϵ) / ϵ)) .* 
    ((k ./ yopt) .^ (1. / ϵ)) # price-takers
τ_b = 1. .- c1 ./ (β * R * c0)

# open("julia/complementarities/tau_k_3.txt", "w") do io
#     writedlm(io, τ_k)
# end

ppath = plot(gpath, label = [L"U(\theta)" L"\mu(\theta)"],
    title = "DE Solution", xlab = L"\theta", 
    legend = :bottomright)
pt0 = plot(tgrid, [c0 k w .- c0 .- k], 
    label = [L"c_0(\theta)" L"k(\theta)" L"b(\theta)"],
    title = "t = 0", xlab = L"\theta", legend = :topleft)
pt1 = plot(tgrid, [c1 tgrid .* k],
    label = [L"c_1(\theta)" L"\theta k(\theta)"],
    title = "t = 1", xlab = L"\theta", legend = :topleft)
pwedge = plot(tgrid, [τ_k τ_b], 
    label = [L"\tau_k(\theta)" L"\tau_b(\theta)"],
    title = "Wedges " *  "\$\\epsilon = $ϵ\$", xlab = L"\theta",
    legend = :left)

display(plot(pt0, pt1, ppath, pwedge))
savefig("julia/complementarities/prelim_soln.png")

# # Verify global incentive constraints
# function verify_global()
#     ics_all = zeros(nt, nt)
#     for i in 1:nt, j in 1:nt
#         ics_all[i, j] = log(max(c0[j] + k[j] - tgrid[j] * k[j] / tgrid[i], 1e-12)) + β * log(c1[j]) - 
#             log(c0[i]) - β * log(c1[i])
#     end
#     return ics_all
# end

# ics_all = verify_global()
# @show all(ics_all .<= 1e-4); # violations are small, and near diagonal (not a huge problem)
# # findall(ics_all .> 1e-6)

invrate = k ./ c0
ptheta = (yopt ./ (tgrid .* k)) .^ (1. / ϵ)
thetap = tgrid .* ptheta

p_return = plot(tgrid, thetap ./ R,
    label = L"\theta p(\theta)/R" ,
    legend = :topleft)

tstar = tgrid[findmax(thetap ./ R)[2]]

p_inv = plot(tgrid, invrate,
    label =  L"k/c_0",
    legend = :topleft)
display(plot(p_return, p_inv))
savefig("julia/complementarities/wedge_determs.png")

tk2 = (1. .+ invrate) .* (1. .- R ./ thetap)
tb2 = -invrate .* (1. .- thetap ./ R)
display(plot(tgrid, [tk2 tb2])) # checks out!

# # Plots for paper
# figpath = "writing/draft_2021/figures/"

# # allocations
# plot(pt0, pt1)
# savefig(figpath * "allocations.png")

# # Wedges
# plot(tgrid, [τ_k τ_b], 
#     label = [L"\tau_k(\theta)" L"\tau_b(\theta)"],
#     legend = :topleft)
# vline!([tstar], label = "", color = "black")
# pwedge_paper = annotate!(tstar + 0.04, 0.03, L"\theta^*")
# savefig(figpath * "wedges.png")

# # determinants
# plot(p_return, p_inv)
# savefig(figpath * "determs.png")