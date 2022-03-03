
#= 
Solves two-period, two-type case 
Two models in here: one with mobile capital, one without

The output is the vector 
x = [c(θ_L), c(θ_H), k(θ_L), k(θ_H)]′

=#

using LinearAlgebra, Optim, ForwardDiff


# Parameters
β = 0.95            # discounting
R = 1.0 / β - 0.01  # interest rate 
θ = [1.01, 1.06]    # productivity
π = [0.6, 0.4]       # probabilities of types
w = 1.2             # starting wealth
N = 2               # no. of types
E = 0.0             # government expenditures

# Objective and constraints, plus 1st and 2nd derivatives
function plan_obj(x)
    c = x[1:N]
    k = x[(N + 1):2N]
    obj = -dot(π, log.(w .- k) + β * log.(c))
end

function plan_grad!(g, x)
    g = ForwardDiff.gradient(plan_obj, x)
end

function plan_hess!(h, x)
    h = ForwardDiff.hessian(plan_obj, x)
end

function constr!(c, x)
    c = x[1:N]
    k = x[(N + 1):2N]

    # Resource constraint
    rc = E - dot(π, θ .* k .- c)

    # Incentive constraints
    ics = zeros(N * (N - 1), 1)
    p = 1 
    for i in 1:N
        for j in 1:N # type i mimicking type j 
            if i == j
                continue
            else
                if θ[j] / θ[i] * k[j] >= w 
                    ics[p] = -30.0 - abs(w - θ[j] / θ[i] * k[j]) + β * log(c[j]) - 
                        log(w - k[i]) - β * log(c[i])
                else
                    ics[p] = log(w - θ[j] / θ[i] * k[j]) + β * log(c[j]) - 
                        log(w - k[i]) - β * log(c[i])
                end
                p += 1
            end
        end
    end
    c = [rc; ics]
end

x0 = ones(2N) * 0.3
plan_obj(x0)
# ForwardDiff.gradient(plan_obj, x0)
# ForwardDiff.hessian(plan_obj, x0)

function constr_jac!(j, x)
    c_dum = (y) -> constr!(zeros(N * (N + 1), 1), y)
    j = ForwardDiff.jacobian(c_dum, x)
end

c_dum = (y) -> constr!(zeros(N * (N + 1), 1), y)
ForwardDiff.jacobian(c_dum, x0) # ForwardDiff has trouble with the constraints

function constr_hess(h, x, λ)
    c_dum = (y) -> constr_jac!(zeros(N * (N + 1), 1), y)
    # h += λ .* ForwardDiff.hessian(c_dum, x) 
end

# # Build initial guess
# k0 = [0.25 0.75]
# rc0 = dot(π, k0 .* θ)
# x0 = vec([k0 .* rc0 k0])

# # Box constraints
# lx = zeros(length(x0)) # c ≥ 0, k ≥ 0
# ux = [Inf, Inf, w, w]  # no bound on c, k ≤ w 

# # Nonlinear constraints (output of constr())
# # All are defined such that cᵢ(x) ≤ 0
# lc = fill(-Inf, 3)
# uc = zeros(3)

# # Set up problem 
# func = TwiceDifferentiable(plan_obj, plan_grad!, plan_hess!, x0)
# cons = TwiceDifferentiableConstraints(constr!, constr_jac!, constr_hess!, lx, ux, lc, uc)

# res = optimize(func, cons, x0, IPNewton())
