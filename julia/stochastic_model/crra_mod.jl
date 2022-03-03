
# Attempting to solve the problem with CRRA utility 

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, QuadGK, FastGaussQuadrature
gr()

println("*** crra_mod.jl ***")

# Parameters
α = 0.5         # Pr(y>0)
β = 0.95        # discounting
θ_min = 1.0     # works with 3.8
θ_max = 2.8     # looks ok with 4.6     
w = 1.2         # starting endowment (wealth)
σ = 2.0         # 

# Interest rate
R = 1.1
λ_1 = 1.0          # works with 0.5
λ_0 = R * λ_1
println([λ_0 λ_1])

# utility function
function util(c)
    (c^(1.0 - σ)) / (1.0 - σ)
end

# Distribution for θ 
function par(x; bd=false)
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
tgrid = range(θ_min, θ_max, length=nt)
c0_lf = 0.6 * ones(nt)
klf = 0.5 * ones(nt)
blf = 0.1 * ones(nt)
c1y_lf = tgrid .* klf
c1u_lf = R * blf
Ulf = util.(c0_lf) + β * (α * util.(c1y_lf) + (1 - α) * util.(c1u_lf))

μ_lf = -0.1 * ones(nt)
μ_lf[[1 end]] .= 0.0

function foc_kp!(focs, x, U, μ, θ, Um)
    c0, k, c1y, c1u, ϕ = x
    focs[1] = 1.0 / β * (c1y / c0)^σ - σ * μ * k / (λ_1 * θ * c0^(σ + 1)) - ϕ / (λ_1 * (c0 + k)) - R
    focs[2] = (c1y / c1u)^σ - (β * ϕ) / (λ_1 * (1.0 - α) * c1u^σ) - 1.0
    focs[3] = α * θ + μ / (λ_1 * θ * c0^σ) - ϕ / (λ_1 * (c0 + k)^σ) - R
    focs[4] = U - util(c0) - β * (α * util(c1y) + (1 - α) * util(c1u))
    focs[5] = Um - util(c0 + k) - β * util(c1u)
    return focs
    
end

function foc_k0_cf(U)
    c0 = (U * (1.0 - σ) / (1.0 + (β * R ^ (1.0 - σ)) ^ (1.0 / σ))) ^ (1.0 / (1.0 - σ))
    c1 = c0 * (β * R) ^ (1.0 / σ)
    return [c0 c1]'
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

function alloc_single(U, μ, t, Um)

    it = findmin(abs.(tgrid .- t))[2] # for finding initial guess
    
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
    
    du[1] = k / (t * c0 ^ σ)

    ~, ft, fpt = par(t)
    if t == θ_min
        du[2] = λ_1 / β * (c1y ^ σ) - μ * fpt / ft - 1 - ϕ
    else
        du[2] = λ_1 / β * (c1y ^ σ) - μ * fpt / ft - 1
    end
    
end

# Shooting algorithm
function tax_shoot(Umin)
    u0 = [Umin 0.0]
    p = (Umin)  # pass this as parameter and bound
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan, p)
    gpath = solve(prob, alg_hints=[:stiff]) 
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

test_ums = range(-10.0, 10.0, length=250)
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