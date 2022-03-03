#= 

Solves the dynamic CPP for w=0 (take two)

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, Printf, DelimitedFiles

gr()

println("******** dyn_cmp.jl ********")

# Parameters
const β = 0.95        # discounting
const θ_min = 1.
const θ_max = 2.
const ϵ = 4.

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
    
    # Uniform
    cdf = (x - L) / (H - L)
    pdf = 1. / (H - L)
    fpx = 0.
    
    return cdf, pdf, fpx
    
end

function gauss_leg(f, n, a, b)
    # Uses Gauss-Legendre quadrature with n nodes over [a,b]
    # Upside: can be quicker
    # Downside: no idea how accurate the solution is (not adaptive)
    
    # Get nodes and weights
    xi, ωi = gausslegendre(n)
    
    # Compute approximation 
    x_new = (xi .+ 1) * (b - a) / 2.0 .+ a # change of variable
    approx = (b - a) / 2.0 * (ωi' * f.(x_new))
    return approx
    
end

function find_bracket(f; bkt0 = (-0.3, -0.2), step_size = 0.1)
    # Finds the bracket (x, x + 0.1) containing the zero of 
    # the function f
    
    a, b = bkt0
    its = 0
    mxit = 1000
    
    while f(a) * f(b) > 0.0 && its < mxit
        if f(a) > 0.0           # f(a), f(b) positive
            if f(a) > f(b)      # f decreasing
                a = copy(b)
                b += step_size
            else                # f increasing
                b = copy(a)
                a -= step_size
            end
        else                    # f(a), f(b) negative
            if f(a) > f(b)      # f increasing
                b = copy(a)
                a -= step_size
            else                # f decreasing
                a = copy(b)
                b += step_size
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

# test_ums = range(-10., 0., length = 20)
# testvals = [test_shoot(test_ums[i]) for i in 1:length(test_ums)]
# display(plot(test_ums, testvals))

# bkt = find_bracket(tax_shoot, bkt0 = (-5.1, -5.))
# u0_opt = find_zero(tax_shoot, bkt)

# u0 = [u0_opt 0.0]
# tspan = (θ_min, θ_max)
# prob = ODEProblem(de_system!, u0, tspan)
# gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 

# total_u = gauss_leg(t -> util_integrand(gpath, t), 50, θ_min, θ_max)

function find_gstar(γ, A; return_all = false)
    # Given γ and A, solves the system and returns ∫UdF (default)
    # or the full solution (return_all = true)
    
    # Build the functions
    function foc_k(x, U, μ, θ)
        
        c, wp, k = x
        ptilde = exp(-(1. - β) / ϵ * wp)
        phat = (θ * k) ^ (-1. / ϵ)
    
        # FOC vector
        focs = zeros(eltype(x), 3)
        focs[1] = 1. - (1. - β) / (β * R * c) * A * exp((1. - β) * wp) - μ * k / (θ * c)
        focs[2] = 1. - κ / R * θ ^ (1. - 1. / ϵ) * k ^ (-1. / ϵ) + μ / (θ * c)
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
        ~, ft, fpt = fdist(t)
        du[1] = k / (c0 * t)
        du[2] = γ - (1. - β) / (β * R) * A * exp((1. - β) * wp) - μ * fpt / ft 
    end
    
    function tax_shoot(U_0)
        u0 = [U_0 0.0]
        tspan = (θ_min, θ_max)
        prob = ODEProblem(de_system!, u0, tspan)
        gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
        return gpath.u[end][2] # want to get μ(θ̲) = 0
        
    end
    
    function cost_integrand(gpath, t)
        U, μ = gpath(t)
        Ft, ft, fpt = fdist(t)
        c0, wp, k = alloc_single(U, μ, t)
        pprime = κ * (t * k) ^ (-1. / ϵ)
        ct = (c0 + k + (1. / R) * (A * exp((1. - β) * wp) - pprime * t * k)) * ft 
        return ct
    end
    
    function util_integrand(gpath, t)
        U, μ = gpath(t)
        Ft, ft, fpt = fdist(t)
        return U * ft
    end

    # Optimal allocations along the solution 
    function opt_allocs(gpath)
        c0p = zeros(nt)
        wpp = zeros(nt)
        kpp = zeros(nt)
        
        for i in 1:nt
            t = tgrid[i]
            U, mu = gpath(t)
            c0p[i], wpp[i], kpp[i] = alloc_single(U, mu, t)
        end
        
        return c0p, wpp, kpp
    end

    # Find the bracketing interval to get μ(θ̲) = 0
    bkt = find_bracket(tax_shoot, bkt0 = (-5.1, -5.))
    u0_opt = find_zero(tax_shoot, bkt)
    
    # Solve the system
    u0 = [u0_opt 0.0]
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan)
    gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
    
    total_u = gauss_leg(t -> util_integrand(gpath, t), 50, θ_min, θ_max)
    
    if return_all
        c0p, wpp, kpp = opt_allocs(gpath)
        a_update = gauss_leg(t -> cost_integrand(gpath, t), 50, θ_min, θ_max)
        return gpath, c0p, wpp, kpp, a_update
    else
        return total_u
    end 
    
end

# R = 1.2
# κ = 1.
# A0 = 1.
# find_gstar(0.052, A0)

# bkt = find_bracket(g -> find_gstar(g, A0), bkt0 = (0.051, 0.052), step_size = 0.001)
# gamstar = find_zero(g -> find_gstar(g, A0), bkt)
# gpath, c0p, wpp, kpp, a_update = find_gstar(gamstar, A0, return_all = true)

function iterate_a(A0)

    # Iterates on A until convergence
    A1 = copy(A0)
    mxit = 250
    its = 1
    diff = 10.
    tol = 1e-5
    bkt_gamma_init = (0.071, 0.072)
    gamsol = 0.

    print("\nSolving for A \n")
    print("-----------------------------\n")
    print(" its     diff         A1\n")
    print("-----------------------------\n")

    while diff > tol && its < mxit
        bkt_gam = find_bracket(g -> find_gstar(g, A0), bkt0 = bkt_gamma_init, step_size = 0.001)
        bkt_gamma_init = bkt_gam
        gamstar = find_zero(g -> find_gstar(g, A0), bkt_gam)
        gpath, c0p, wpp, kpp, A1 = find_gstar(gamstar, A0, return_all = true)
        diff = abs(A1 - A0)
        @printf("%3d %12.8f %12.8f\n", its, diff, A1)
        A0 = copy(A1)
        if diff > 0.001
            ω = 1.4 # acceleration 
        else
            ω = 1.
        end
        A0 = ω * A1 + (1. - ω) * A0
        its += 1
        gamsol = copy(gamstar)
    end
    gpath, c0p, wpp, kpp, ~ = find_gstar(gamsol, A1, return_all = true)
    return gpath, c0p, wpp, kpp, A1, gamsol, bkt_gamma_init

end

R = 1.2
κ = 0.9
A0 = 1.4012872592148484 # A* for R = 1.2, κ = 0.9
gpath, c0p, wpp, kpp, A1, gamsol, bkt_gamma_init = iterate_a(A0);

pck = plot(tgrid, [c0p kpp], 
    label = [L"c(\theta)" L"k(\theta)"],
    xlab = L"\theta", legend = :topleft)
pw = plot(tgrid, wpp,
    label = L"w^\prime(\theta)",
    xlab = L"\theta", legend = :topleft)
display(plot(pck, pw))

wspace = range(-2., 12., length = nt)
cwvals = c0p * exp.((1. - β) .* wspace)';

# c(θ,w)
surface(tgrid, wspace, cwvals, 
    title = L"c(\theta,w)",
    xlab = L"\theta", ylab = L"w",
    legend = false)

# Wedges 
figpath = "writing/final_2021/figures/"
function calc_wedges()
    # tkgrid[i, j, k] is for θ_t = i, θ_t+1 = j, w = k
    # tbgrid[i, j] is for θ_t = i, θ_t+1 = j

    tkgrid = zeros(nt, nt, nt)
    tbgrid = zeros(nt, nt)
    for i in 1:nt, j in 1:nt
        tbgrid[i, j] = 1. - c0p[j] * exp((1. - β)wspace[j]) / (β * R * c0p[i])
        for k in 1:nt
            pwt = κ * (tgrid[i] * kpp[i] * exp((1. - β)wspace[i])) ^ (-1. / ϵ)
            tkgrid[i, j, k] = 1. - c0p[j] * exp((1. - β)wspace[j]) / (β * pwt * c0p[i])
        end
    end
    return tkgrid, tbgrid
end

tkgrid, tbgrid = calc_wedges();

# τ_k
# Fix θₜ
tfix = 10 
ptf = surface(tgrid, wspace, tkgrid[tfix, :, :], 
    xlab = L"\theta_{t+1}", ylab = L"w",
    zlab = L"\tau_{t+1,k}(\theta^{t+1})",
    legend = false)

# Fix w
wfix = 10
ptw = surface(tgrid, tgrid, tkgrid[:, :, wfix], 
    xlab = L"\theta_{t}", ylab = L"\theta_{t+1}",
    legend = false)

display(plot(ptf, ptw))
savefig(figpath * "dyn_tauk.png")


# τ_b
# This one only depends on the pair {\theta_t, \theta_t+1}
display(surface(tgrid, tgrid, tbgrid, 
    xlab = L"\theta_{t}", ylab = L"\theta_{t+1}",
    zlab = L"\tau_{t+1,b}(\theta^{t+1})",
    legend = false))
savefig(figpath * "dyn_taub.png")
