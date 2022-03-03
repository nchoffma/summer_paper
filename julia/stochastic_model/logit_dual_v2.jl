
# Solves the dual, with allocations to non-investors solved in closed form


# Solves the Cost Minimization Problem in the FPM 

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, QuadGK, FastGaussQuadrature
gr()

println("*** logit_dual_v2.jl ***")

# Parameters
α = 0.5         # Pr(y>0)
β = 0.95        # discounting
θ_min = 1.4     # up from 3.2
θ_max = 2.8     # up from 4.6
w = 1.2         
R = 1.3

# Promise-Keeping
Ustar = -0.7         # minimum total utility 
γ = 1.0             # multiplier on PKC (3.5)

#= 

- Example in current draft:
α = 0.5, R = 1.5, β = 0.95, θ_min = 3.2, θ_max = 4.6

- Why are the allocations reversed?
- Why do these not satisfy global incentive constraints?

=#

# Distribution for θ (Pareto)
function par(x; bd = false)
    # F(x), f(x), f′(x) for Pareto (bounded/unbounded)
    
    # a = 1.5 # shape parameter
    
    # if bd 
    #     L = θ_min
    #     H = θ_max 
    #     den = 1.0 - (L / H) ^ a 
        
    #     cdf = (1.0 - L ^ a * x ^ (-a)) / den 
    #     pdf = a * L ^ a * x ^ (-a - 1.0) / den 
    #     fpx = a * (-a - 1.0) * L ^ a * x ^ (-a - 2.0)
    # else
    #     xm = θ_min
    #     cdf = 1.0 - (xm / x) ^ a
    #     pdf = a * xm ^ a / (x ^ (a + 1.0))
    #     fpx = (-a - 1.0) * a * xm ^ a / (x ^ (a + 2.0))
    # end
    
    # Uniform
    a = θ_min
    b = θ_max
    cdf = (x - a) / (b - a)
    pdf = 1.0 / (b - a)
    fpx = 0.0
    
    return cdf, pdf, fpx

end

# LF allocations, as initial guess
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 0.6 * ones(nt)
klf = 0.5 * ones(nt)
blf = 0.1 * ones(nt)
c1y_lf = tgrid .* klf
c1u_lf = R * blf
Ulf = log.(c0_lf) + β * (α * log.(c1y_lf) + (1 - α) * log.(c1u_lf))

μ_lf = -0.1 * ones(nt)
μ_lf[[1 end]] .= 0.0

function foc_kp!(focs, x, U, μ, θ, Um)
    c0, k, c1y, c1u, ϕ = x
    focs[1] = 1.0 - c1y / (β * R * c0) + ϕ / (c0 + k) - μ * k / (θ * c0 ^ 2)    # c0
    focs[2] = c1y - c1u - β * R * ϕ / (1.0 - α)                                 # c1u 
    focs[3] = 1.0 + μ / (θ * c0) + ϕ / (c0 + k) - α * θ / R                     # k 
    focs[4] = log(c0) + β * (α * log(c1y) + (1.0 - α) * log(c1u)) - U           # η
    focs[5] = log(c0 + k) + β * log(c1u) - Um                                   # ϕ
    return focs
    
end

function foc_k0_cf(U)
    c0 = (exp(U) / (R * β) ^ β) ^ (1.0 / (1.0 + β))
    c1 = R * β * c0 
    return [c0 c1]'
end

function foc_newton_kp(U, μ, θ, i, Um)
    x0 = [c0_lf[i] klf[i] c1y_lf[i] c1u_lf[i] 0.5]'
    mxit = 1000
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0
    
    while diff > tol && its < mxit
        focs = foc_kp!(similar(x0), x0, U, μ, θ, Um)
        diff = maximum(abs.(focs))
        dfoc = ForwardDiff.jacobian(x -> foc_kp!(similar(x), x, U, μ, θ, Um), x0)
        d = dfoc \ focs
        while minimum(x0 - d) .< 0.0 # avoid moving to a negative point
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
    if its == mxit && diff > tol
        fail = 1
    end
    return x0, fail
    
end

function alloc_single(U, μ, t, Um)
    # Solves for allocations c0, k, c1y, c1u, and ϕ for a single (t, U(t), μ(t))
    # given Um = U(θ̲)
    
    it = findmin(abs.(tgrid .- t))[2] # for finding initial guess
    
    # println((it, U))
    x, f = foc_newton_kp(U, μ, t, it, Um)
    if f == 1
        x0 = foc_k0_cf(U)
        c0, c1u = x0
        c1y = c1u
        k = 0.0
        ϕ = 0.0
    else
        c0, k, c1y, c1u, ϕ = x 
    end
    return c0, k, c1y, c1u, ϕ
    
end

function de_system!(du, u, p, t)
    U, μ = u    # current guess
    Um = p[1]   # make sure this is a number
    c0, k, c1y, c1u, ϕ = alloc_single(U, μ, t, Um)
    
    du[1] = k / (c0 * t)
    
    ~, ft, fpt = par(t)
    if t == θ_min
        du[2] = γ - c1y / (β * R) - μ * fpt / ft + ϕ
    else
        du[2] = γ - c1y / (β * R) - μ * fpt / ft
    end

end

# Shooting algorithm 
function tax_shoot(Umin)
    u0 = [Umin 0.0]
    p = (Umin)  # pass this as parameter and bound
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan, p)
    gpath = solve(prob, Trapezoid(), alg_hints = [:stiff])
    return gpath.u[end][2] # want to get μ(θ̲) = 0

end

function find_bracket(f)
    # Finds the bracket (x, x + 0.1) containing the zero of 
    # the function f

    a, b = (-4.2, -4.1)
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
function test_shoot(Um)
    ep = try
        tax_shoot(Um)
    catch y 
        if isa(y, UndefVarError)
            NaN
        elseif isa(y, LAPACKException)
            NaN
        elseif isa(y, DomainError)
            NaN
        end
    end
    return ep
end

test_ums = range(-5.0, 5.0, length = 250)
testvals = [test_shoot(test_ums[i]) for i in 1:length(test_ums)]
plot(test_ums, testvals)

# @time bkt = find_bracket(tax_shoot)
# println("Bracket found:", bkt, ", going to root finder")
# @time Umin_opt = find_zero(tax_shoot, bkt, Bisection())
# end_opt = tax_shoot(Umin_opt)

# println("Root found, val = ", end_opt, ", going to allocations")
# # Solve the system, given Umin_opt
# u0 = [Umin_opt 0.0]
# p = (Umin_opt)
# tspan = (θ_min, θ_max)
# prob = ODEProblem(de_system!, u0, tspan, p)
# gpath_opt = solve(prob, alg_hints = [:stiff])

# function opt_allocs(gpath)
#     # Allocations at opt 
#     # takes solution to ODEProblem

#     c0 = zeros(length(gpath.t))
#     k = copy(c0)
#     c1y = copy(c0)
#     c1u = copy(c0)
#     ϕ = copy(c0)
#     # i = 1
#     # for t in gpath.t 
#     #     U, μ = gpath(t)
#     #     c0[i], k[i], c1y[i], c1u[i], ϕ[i] = alloc_single(U, μ, t, Umin_opt)
#     #     i += 1
#     # end 
#     for i in 1:length(gpath.t)
#         U, μ = gpath.u[i]
#         t = gpath.t[i]
#         c0[i], k[i], c1y[i], c1u[i], ϕ[i] = alloc_single(U, μ, t, Umin_opt)
#     end
    
#     return c0, k, c1y, c1u, ϕ
# end

# c0, k, c1y, c1u, ϕ = opt_allocs(gpath_opt)
# b = w .- (c0 + k) # borrowing (looks strange b/c the budget doesn't clear)

# # Plot solution
# pt0 = plot(gpath_opt.t, [c0 k b],
#     label = [L"c_0(\theta)" L"k(\theta)" L"b(\theta)"],
#     title = L"t = 0", xlab = L"\theta", legend = :bottomright)
# pt1 = plot(gpath_opt.t, [c1u c1y ϕ],
#     label = [L"c_1^0(\theta)" L"c_1^y(\theta)" L"\phi(\theta)"],
#     title = L"t = 1", xlab = L"\theta", legend = :topright)
# pU = plot(gpath_opt, vars = (0, 1), title = L"U(\theta)",
#     xlab = L"\theta", legend = false)
# pμ = plot(gpath_opt, vars = (0, 2), title = L"\mu(\theta)",
#     xlab = L"\theta", legend = false)
# p_alloc = plot(pt0, pt1, pU, pμ, layout = (2, 2))
# # savefig("julia/stochastic_model/allocs_dual")

# # Total utility
# function integrand_u(t, gpath)
#     U, ~ = gpath(t)
#     ~, ft, ~ = par(t)
#     return U * ft 
# end

# function gauss_leg(f, n, a, b)
#     # Uses Gauss-Legendre quadrature with n nodes over [a,b]
#     # Upshot: can be quicker
#     # Downside: no idea how accurate the solution is (not adaptive)

#     # Get nodes and weights
#     xi, ωi = gausslegendre(n)

#     # Compute approximation 
#     x_new = (xi .+ 1) * (b - a) / 2.0 .+ a # change of variable
#     approx = (b - a) / 2.0 * (ωi' * f.(x_new))
#     return approx
# end

# total_u = gauss_leg(t -> integrand_u(t, gpath_opt), 1000, θ_min, θ_max)
# @show [total_u, Ustar]

# # Wedges
# τ_k = 1.0 .- (1.0 ./ c0) ./ (α * β * gpath_opt.t .* (1.0 ./ c1y))
# τ_b = 1.0 .- (1.0 ./ c0) ./ (β * R * (α ./ c1y + (1.0 - α) ./ c1u))
# ptk = plot(gpath_opt.t, τ_k, title = L"\tau_k(\theta)",
#     xlabel = L"\theta",
#     legend = false)
# ptb = plot(gpath_opt.t, τ_b, title = L"\tau_b(\theta)",
#     xlabel = L"\theta",
#     legend = false)
# plot(ptk, ptb)
# # savefig("julia/stochastic_model/wedges_dual")

# plot(pt0, pt1, ptk, ptb, layout = (2, 2))
# # savefig("julia/stochastic_model/results_dual")
