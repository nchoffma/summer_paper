
# Attempting to solve the problem as posed by Chris

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, QuadGK, FastGaussQuadrature
gr()

println("*** logit_mod.jl ***")

# Parameters
α = 0.5         # Pr(y>0)
β = 0.95        # discounting
θ_min = 1.0     # works with 3.8
θ_max = 2.8     # looks ok with 4.6     
w = 1.2

# Interest rate
R = 1.0821
λ_1 = 1.3217          # works with 0.5
λ_0 = R * λ_1
println([λ_0 λ_1])

# Idea: λ_t is *something* like the (relative) price of consumption in period t 
# So, if resources used at time t are too low, can lower λ_t, and if they are too high,
# lower λ_t. 

#=

One that works: 
R = 1.5, λ_1 = 0.45, θ_min = 3.2, θ_max = 4.6 (uniform):
    [bc0 bc1] = [-1.897119154115152 2.282652316233833]

Curious: this looks better with R = 2 and the bounds 3.2 and 4.6...

R = 2.0, λ_1 = 1.0, θ_min = 3.2, θ_max = 4.6 (uniform):
    [bc0 bc1] = [0.6986322055948295 -0.8840478907580817]
    total_u = -0.8045697793584983

    9/28
    - This example works even if we move around the bounds (without θ̄ going too high...)
    - The problem is that R is too high, and λ_1 too low 

R = 1.3, λ_1 = 1.2, θ_min = 1.6, θ_max = 3.0 (uniform):
    [bc0 bc1] = [0.5576964496795255 -0.7538807879367222]
    total_u = -0.709355150772456

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
    focs[1] = c1y / (β * c0) - μ * k / (λ_1 * θ * c0 ^ 2) - ϕ / (λ_1 * (c0 + k)) - R
    focs[2] = c1y / c1u - (β * ϕ) / (λ_1 * (1.0 - α) * c1u) - 1
    focs[3] = α * θ + μ / (λ_1 * c0 * θ) - ϕ / (λ_1 * (c0 + k)) - R
    focs[4] = U - log(c0) - β * (α * log(c1y) + (1 - α) * log(c1u))
    focs[5] = Um - log(c0 + k) - β * log(c1u)
    return focs
    
end

function foc_k0!(focs, x, U)
    c0, c1 = x 
    focs[1] = log(c0) + β * log(c1) - U
    focs[2] = c1 / (β * c0) - R
    return focs 

end

function foc_newton_kp(U, μ, θ, i, Um)
    x0 = [c0_lf[i] klf[i] c1y_lf[i] c1u_lf[i] 0.5]'
    mxit = 500
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0
    
    while diff > tol && its < mxit
        focs = foc_kp!(similar(x0), x0, U, μ, θ, Um)
        diff = maximum(abs.(focs))
        dfoc = ForwardDiff.jacobian(x -> foc_kp!(similar(x), x, U, μ, θ, Um), x0)
        d = dfoc \ focs
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

function foc_newton_k0(U, i)
    x0 = [c0_lf[i] c1u_lf[i]]'
    mxit = 500
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0
    
    while diff > tol && its < mxit
        focs = foc_k0!(similar(x0), x0, U)
        diff = maximum(abs.(focs))
        dfoc = ForwardDiff.jacobian(x -> foc_k0!(similar(x), x, U), x0)
        d = dfoc \ focs
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
    if its == mxit && diff > tol
        fail == 1
    end
    return x0, fail
end

function alloc_single(U, μ, t, Um)

    it = findmin(abs.(tgrid .- t))[2] # for finding initial guess
    
    x, f = foc_newton_kp(U, μ, t, it, Um)
    if f == 1
        x0, f0 = foc_newton_k0(U, it)
        if f0 == 1
            println((t, U))
            println("neither converged")
        else
            c0, c1u = x0
            c1y = c1u
            k = 0.0
            ϕ = 0.0
        end
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
        du[2] = λ_1 / β * c1y - μ * fpt / ft - 1 - ϕ
    else
        du[2] = λ_1 / β * c1y - μ * fpt / ft - 1
    end
    
end

# Shooting algorithm
function tax_shoot(Umin)
    u0 = [Umin 0.0]
    p = (Umin)  # pass this as parameter and bound
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan, p)
    gpath = solve(prob, alg_hints = [:stiff]) 
    return gpath.u[end][2] # want to get μ(θ̲) = 0

end

function find_bracket(f)
    # Finds the bracket (x, x + 0.1) containing the zero of 
    # the function f

    a, b = (-3.3, -3.2)
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

test_ums = range(-10.0, 10.0, length = 250)
testvals = [test_shoot(test_ums[i]) for i in 1:length(test_ums)]
plot(testvals)

# @time bkt = find_bracket(tax_shoot)
# println("Bracket found:", bkt, " going to root finder")
# @time Umin_opt = find_zero(tax_shoot, bkt)

# # tax_shoot(Umin_opt)
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
#     i = 1
#     for t in gpath.t 
#         U, μ = gpath(t)
#         c0[i], k[i], c1y[i], c1u[i], ϕ[i] = alloc_single(U, μ, t, Umin_opt)
#         i += 1
#     end 
#     return c0, k, c1y, c1u, ϕ
# end

# c0, k, c1y, c1u, ϕ = opt_allocs(gpath_opt)
# b = w .- (c0 + k)

# # Plot solution
# pt0 = plot(gpath_opt.t, [c0 k b],
#     label = [L"c_0(\theta)" L"k(\theta)" L"b(\theta)"],
#     title = L"t = 0", xlab = L"\theta", legend = :topleft)
# pt1 = plot(gpath_opt.t, [c1u c1y ϕ],
#     label = [L"c_1^0(\theta)" L"c_1^y(\theta)" L"\phi(\theta)"],
#     title = L"t = 1", xlab = L"\theta", legend = :topleft)
# pU = plot(gpath_opt, vars = (0, 1), title = L"U(\theta)",
#     xlab = L"\theta", legend = false)
# pμ = plot(gpath_opt, vars = (0, 2), title = L"\mu(\theta)",
#     xlab = L"\theta", legend = false)
# p_alloc = plot(pt0, pt1, pU, pμ, layout = (2, 2))
# savefig("julia/stochastic_model/allocations")

# # Budget constraint
# # Idea: given R, there is a value of λ_1 (and thus λ_0) that 
# # leads to budget-clearing allocations 
# println("Moving to BC")

# function integrand0(t, gpath, Umin)
#     # Budget constraint, t = 0
#     # ∫[c₀ + k]fdθ ≤ w 
#     # gpath is the solution to the ODE system at Umin 
#     U, μ = gpath(t)
#     c0, k, ~, ~, ~ = alloc_single(U, μ, t, Umin)
#     ~, ft, ~ = par(t)
#     return (c0 + k) * ft 

# end

# function integrand1(t, gpath, Umin)
#     # Budget constraint, t = 1
#     # ∫[α(θk - c₁ʸ) + (1 - α)c₁ᵘ]fdθ ≥ 0
#     # gpath is the solution to the ODE system at Umin 
#     U, μ = gpath(t)
#     c0, k, c1y, c1u, ~ = alloc_single(U, μ, t, Umin)
#     ~, ft, ~ = par(t)
#     return (α * (t * k - c1y) - (1.0 - α) * c1u) * ft

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

# spend0_gl = gauss_leg(t -> integrand0(t, gpath_opt, Umin_opt), 1000, θ_min, θ_max)
# bc0 = w - spend0_gl
# bc1 = gauss_leg(t -> integrand1(t, gpath_opt, Umin_opt), 1000, θ_min, θ_max)
# @show [bc0 bc1];

# # # Wedges
# # τ_k = 1.0 .- c1y ./ (α * β * gpath_opt.t .* c0)
# # τ_b = 1.0 .- (1.0 ./ c0) ./ (β * R * (α ./ c1y + (1.0 - α) ./ c1u))
# # θ_c = gpath_opt.t[findfirst(k .> 0.0)]
# # plot(gpath_opt.t, [τ_k τ_b], label = [L"\tau_k(\theta)" L"\tau_b(\theta)"],
# #     title = "Wedges", xlabel = L"\theta",
# #     legend = :bottomright)
# # savefig("julia/stochastic_model/wedges")

# # Calculate total utility
# # Total utility
# function integrand_u(t, gpath)
#     U, ~ = gpath(t)
#     ~, ft, ~ = par(t)
#     return U * ft 
# end

# total_u = gauss_leg(t -> integrand_u(t, gpath_opt), 1000, θ_min, θ_max)
# @show total_u
# p_alloc