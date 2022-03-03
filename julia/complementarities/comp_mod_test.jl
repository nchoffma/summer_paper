#= 

Solves the model with complementarities
Testing script: without solving function that takes lambdas

TODO: Find market-clearing lambda values using NLsolve
    - This requires wrapping the whole thing in a function that 
      takes a vector [λ_0 λ_1]
    - May also require try/catch magic, so that it doesn't wander
      into an instable region. 
    - Note: Need to have R≈2--otherwise, consumption in first period is too cheap
      relative to second, and Y goes off into unstable region 

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, Printf

gr()

println("******** comp_mod_TEST.jl ********")

# Parameters
const β = 0.95        # discounting
const θ_min = 1.0
const θ_max = 4.0

w = 1.0
ϵ = 0.8

# Multipliers (works with λ_1 = 1., λ_0 = 2.)
# The solution (and stability) is VERY sensitive to these. 
# Move in small increments!
# λ_1 = 0.581
# λ_0 = 1.25

λ_0 = 1.8952800879273943
λ_1 = 0.9285132899059869
R = λ_0 / λ_1 
@printf "\nλ_0 = %.4f\nλ_1 = %.4f\nR   = %.4f\n" λ_0 λ_1 R

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

function foc_k(x, U, μ, θ, Y)
    
    c0, c1, k = x

    # FOC vector
    focs = zeros(3)
    focs[1] = c1 / (β * c0) - k / (θ * c0 ^ 2) * μ - R 
    focs[2] = (Y / k * θ ^ (ϵ - 1)) ^ (1.0 / ϵ) + μ / c0 - R 
    focs[3] = log(c0) + β * log(c1) - U 

    # Jacobian matrix
    dfocs = zeros(3, 3)
    dfocs[1, 1] = -c1 / (β * c0 ^ 2) + 2k * μ / (θ * c0 ^ 3)
    dfocs[1, 2] = 1.0 / (β * c0)
    dfocs[1, 3] = -μ / (θ * c0 ^ 2)

    dfocs[2, 1] = -μ / (c0 ^ 2)
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

function ces_integrand(t, gpath, Y)
    # CES aggregator for output
    # gpath is the solution to the ODE system at Y 
    U, μ = gpath(t)
    ~, ~, k = alloc_single(U, μ, t, Y)
    ~, ft, ~ = fdist(t)
    return (t * k) ^ ((ϵ - 1) / ϵ) * ft
    
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

Yt = 1.
# test_ums = range(-3., -1., length = 20)
# testvals = [test_shoot(test_ums[i], Yt) for i in 1:length(test_ums)]
# plot(test_ums, testvals)

# tax_shoot(-0.9, Yt)

bkt = find_bracket(um -> tax_shoot(um, Yt), bkt0 = (-1.5, -1.4))
# @time Umin_opt = find_zero(x -> tax_shoot(x, Yt), bkt)
# u0 = [Umin_opt, 0.0]
# p = (Yt)
# tspan = (θ_min, θ_max)
# prob = ODEProblem(de_system!, u0, tspan, p)
# gpath = solve(prob, alg_hints = [:stiff]) 
# y_upd = gauss_leg(t -> ces_integrand(t, gpath, Yt), 20, θ_min, θ_max) ^ (ϵ / (ϵ - 1.))
# println("y1 = $y_upd")
# display(plot(gpath))

# # Optimal allocations along the solution 
# function opt_allocs(gpath, Y)
#     c0 = zeros(nt)
#     k = similar(c0)
#     c1 = similar(c0)

#     for i in 1:nt
#         t = tgrid[i]
#         U, mu = gpath(t)
#         c0[i], c1[i], k[i] = alloc_single(U, mu, t, Y)
#     end

#     return c0, c1, k
# end

# function solve_model(Y_0)
#     Y_1 = copy(Y_0)
#     mxit = 250
#     its = 1
#     diff = 10.0
#     tol = 1e-5
#     bkt_init = (-1.2, -1.1)
#     gpath = 0. 

#     print("\nSolving for Y \n")
#     print("-----------------------------\n")
#     print(" its     diff         y1\n")
#     print("-----------------------------\n")
    
#     while diff > tol && its < mxit
        
#         bkt = find_bracket(um -> tax_shoot(um, Y_0), bkt0 = bkt_init)
#         bkt_init = bkt
#         Umin_opt = find_zero(x -> tax_shoot(x, Y_0), bkt) 
#         u0 = [Umin_opt, 0.0]
#         p = (Y_0)
#         tspan = (θ_min, θ_max)
#         prob = ODEProblem(de_system!, u0, tspan, p)
#         gpath = solve(prob, alg_hints = [:stiff]) 
#         Y_1 = gauss_leg(t -> ces_integrand(t, gpath, Y_0), 20, θ_min, θ_max) ^ (ϵ / (ϵ - 1.))
#         diff = abs(Y_0 - Y_1)
#         if mod(its, 5) == 0
#             @printf("%3d %12.8f %12.8f\n", its, diff, Y_1)
#         end
#         Y_0 = copy(Y_1)
#         its += 1
#     end
    
#     return Y_1, bkt_init, gpath
# end

# y0 = 1.3
# ts = @elapsed yopt, bkt, gpath_test = solve_model(y0) 

# Umin_opt = find_zero(x -> tax_shoot(x, yopt), bkt)
# u0 = [Umin_opt, 0.0]
# p = (yopt)
# tspan = (θ_min, θ_max)
# prob = ODEProblem(de_system!, u0, tspan, p)
# gpath = solve(prob, alg_hints = [:stiff]) 
# ppath = plot(gpath, label = [L"U(\theta)" L"\mu(\theta)"],
#     title = "DE Solution", xlab = L"\theta", 
#     legend = :bottomright)

# c0, c1, k = opt_allocs(gpath, yopt)
# pt0 = plot(tgrid, [c0 k w .- c0 .- k], 
#     label = [L"c_0(\theta)" L"k(\theta)" L"b(\theta)"],
#     title = "t = 0", xlab = L"\theta", legend = :bottomright)
# pt1 = plot(tgrid, [c1 tgrid .* k],
#     label = [L"c_1(\theta)" L"\theta k(\theta)"],
#     title = "t = 1", xlab = L"\theta", legend = :topleft)
# palloc = plot(pt0, pt1, layout = (2, 1))
# display(plot(ppath, palloc, layout = (1, 2)))

# # Check budget constraints
# function bc_integrand_0(t, gpath, Y)
#     U, mu = gpath(t)
#     c0, c1, k = alloc_single(U, mu, t, Y)
#     Ft, ft, fpt = fdist(t)
#     return (c0 + k) * ft
# end

# function bc_integrand_1(t, gpath, Y)
#     U, mu = gpath(t)
#     c0, c1, k = alloc_single(U, mu, t, Y)
#     Ft, ft, fpt = fdist(t)
#     return c1 * ft
# end

# bc0 = gauss_leg(t -> bc_integrand_0(t, gpath, yopt), 20, θ_min, θ_max) - w;
# bc1 = gauss_leg(t -> bc_integrand_1(t, gpath, yopt), 20, θ_min, θ_max) - yopt;
# println([bc0, bc1])

# # Wedges at optimum 
# τ_k = 1. .- c1 ./ (c0 .* β) .* (ϵ / (ϵ - 1.)) .* (k .^ ((1. - ϵ) / ϵ)) .* 
#     ((tgrid ./ yopt) .^ (1. / ϵ))
# τ_b = 1. .- c1 ./ (β * R * c0)

# pwedge = plot(tgrid, [τ_k τ_b], 
#     label = [L"\tau_k(\theta)" L"\tau_b(\theta)"],
#     title = "Wedges", xlab = L"\theta",
#     legend = :bottomright)

# display(plot(pt0, pt1, ppath, pwedge))