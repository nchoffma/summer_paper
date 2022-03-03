
# Testing FOCs in planner's problem

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    QuadGK, NLsolve, ForwardDiff, Optim, LaTeXStrings

# Parameters
α = 0.5         # Pr(y>0)
β = 0.95        # discounting
θ_min = 2.3
θ_max = 4.5
w = 1.2

# Interest rate
R = 1.1
λ_1 = 1.0
λ_0 = R * λ_1

# LF allocations, as initial guess
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 0.6 * ones(nt)
klf = 0.5 * ones(nt)
blf = 0.1 * ones(nt)
c1y_lf = tgrid .* klf
c1u_lf = R * blf
Ulf = log.(c0_lf) + β * (α * log.(c1y_lf) + (1 - α) * log.(c1u_lf))
μ_lf = -0.3 * ones(nt)
μ_lf[[1 end]] .= 0.0

Umin = Ulf[1]

function foc_kp!(focs, x, U, μ, θ)
    c0, k, c1y, c1u, ϕ = x
    focs[1] = c1y / (β * c0) - μ * k / (λ_1 * θ * c0 ^ 2) - ϕ / (λ_1 * (c0 + k)) - R
    focs[2] = c1y / c1u - (β * ϕ) / (λ_1 * (1.0 - α) * c1u) - 1.0
    focs[3] = α * θ + μ / (λ_1 * c0 * θ) - ϕ / (λ_1 * (c0 + k)) - R
    focs[4] = U - log(c0) - β * (α * log(c1y) + (1 - α) * log(c1u))
    focs[5] = Umin - log(c0 + k) - β * log(c1u)
    return focs

end

function foc_k0!(focs, x, U)
    c0, c1 = x 
    focs[1] = log(c0) + β * log(c1) - U
    focs[2] = c1 / (β * c0) - R
    return focs 
end

function foc_k0_cf(U)
    c0 = (exp(U) / (R * β) ^ β) ^ (1.0 / (1.0 + β))
    c1 = R * β * c0 
    return [c0 c1]'
end

function foc_newton_kp(i)
    x0 = [c0_lf[i] klf[i] c1y_lf[i] c1u_lf[i] 0.5]'
    mxit = 100
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0

    while diff > tol && its < mxit
        focs = foc_kp!(similar(x0), x0, Ulf[i], μ_lf[i], tgrid[i])
        diff = maximum(abs.(focs))
        dfoc = ForwardDiff.jacobian(x -> foc_kp!(similar(x), x, Ulf[i], μ_lf[i], tgrid[i]), x0)
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

# function foc_newton_k0(i)
#     x0 = [c0_lf[i] c1u_lf[i]]'
#     mxit = 100
#     tol = 1e-6
#     diff = 10.0
#     its = 0
#     fail = 0

#     while diff > tol && its < mxit
#         focs = foc_k0!(similar(x0), x0, Ulf[i])
#         diff = maximum(abs.(focs))
#         dfoc = ForwardDiff.jacobian(x -> foc_k0!(similar(x), x, Ulf[i]), x0)
#         d = dfoc \ focs
#         while minimum(x0 - d) .< 0.0
#             d = d / 2.0
#             if maximum(d) < tol
#                 fail = 1
#                 break
#             end
#         end
#         if fail == 1
#             break
#         end
#         x0 = x0 - d
#         its += 1

#     end
#     if its == mxit && diff > tol
#         fail == 1
#     end
#     return x0, fail
# end

i = 47
x1, f = foc_newton_kp(i)
x10 = foc_k0_cf(Ulf[i])

# function opt_allocs()
#     c0 = zeros(nt)
#     k = copy(c0)
#     c1y = copy(c0)
#     c1u = copy(c0)
#     ϕ = copy(c0)
#     for i = 1:nt
#         x, f = foc_newton_kp(i)
#         if f == 1
#             x0, f0 = foc_newton_k0(i)
#             if f0 == 1
#                 println(i)
#                 println("neither converged")
#             else
#                 c0[i], c1u[i] = x0
#                 c1y[i] = c1u[i]
#             end
#         else
#             c0[i], k[i], c1y[i], c1u[i], ϕ[i] = x 
#         end
#     end

#     return c0, k, c1y, c1u, ϕ
# end

function opt_allocs()
    c0 = zeros(nt)
    k = copy(c0)
    c1y = copy(c0)
    c1u = copy(c0)
    ϕ = copy(c0)
    for i = 1:nt
        x, f = foc_newton_kp(i)
        if f == 1
            x0 = foc_k0_cf(Ulf[i])
            c0[i], c1u[i] = x0
            c1y[i] = c1u[i]
        else
            c0[i], k[i], c1y[i], c1u[i], ϕ[i] = x 
        end
    end
    
    return c0, k, c1y, c1u, ϕ
end

c0, k, c1y, c1u, ϕ = opt_allocs()

pt0 = plot(tgrid, [c0 k], label = [L"c_0(\theta)" L"k(\theta)"],
    title = "t = 0", xlab = L"\theta", legend = :topleft)
pt1 = plot(tgrid, [c1u c1y ϕ],
    label = [L"c_1^0(\theta)" L"c_1^y(\theta)" L"\phi(\theta)"],
    title = "t = 1", xlab = L"\theta", legend = :topleft)
plot(pt0, pt1, layout = (2, 1))

