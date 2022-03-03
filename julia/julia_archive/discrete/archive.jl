# # Objective and constraints, plus 1st and 2nd derivatives
# function plan_obj(x)
#     obj = 0.0
#     for i in 1:N
#         obj += π[i] * (log(w - x[i + N]) + β * log(x[i]))
#     end
#     obj
# end

# function plan_grad!(g, x)
#     for i in 1:N
#         g[i] = β * π[i] / x[i]
#         g[i + N] = -π[i] / (w - x[i + N])
#     end
#     g
# end

# function plan_hess!(h, x)
#     for i in 1:N
#         h[i, i] = -β * π[i] / (x[i]^2)
#         h[i + N, i + N] = -π[i] / ((w - x[i + N]) ^ 2)
#     end
#     h
# end

# function constr!(c, x)
#     # Resource Constraint
#     res = 0.0
#     for i in 1:N
#         res += π[i] * (θ[i] * x[i + N] - x[i])
#     end
#     rc = E - res 

#     # Incentive constaints
#     ic1 = log(w - θ[1] / θ[2] * x[3]) + β * log(x[1]) - 
#         log(w - x[4]) - β * log(x[2])
#     ic2 = log(w - θ[2] / θ[1] * x[4]) + β * log(x[2]) - 
#         log(w - x[3]) - β * log(x[1])

#     c = [rc, ic1, ic2]
# end

# function constr_jac!(j, x)
#     # Resource constaint
#     j[1, 1:2] = π
#     j[1, 3:4] = -π .* θ

#     # IC₁
#     j[2, 1] = β / x[1]
#     j[2, 2] = -β / x[2]
#     j[2, 3] = -(θ[1] / θ[2]) / (w - θ[1] / θ[2] * x[3])
#     j[2, 4] = 1.0 / (w - x[4])

#     # IC₂
#     j[3, 1] = -β / x[1]
#     j[3, 2] = β / x[2]
#     j[3, 3] = 1.0 / (w - x[3])
#     j[3, 4] = -(θ[2] / θ[1]) / (w - θ[2] / θ[1] * x[4])
#     j
# end

# function constr_hess!(h, x, λ)
#     # Hessian of RC is all 0, as RC is linear 
#     h .+= λ[1] * 0.0 

#     # IC₁
#     h[1, 1] += -λ[2] * β / (x[1] ^ 2)
#     h[2, 2] += λ[2] * β / (x[2] ^ 2)
#     h[3, 3] += -λ[2] / ((w - θ[1] / θ[2] * x[3]) ^ 2) * (θ[1] / θ[2]) ^ 2
#     h[4, 4] += -λ[2] / ((w - x[4]) ^ 2)

#     # IC₂
#     h[1, 1] += λ[3] * β / (x[1] ^ 2)
#     h[2, 2] += -λ[3] * β / (x[2] ^ 2)
#     h[3, 3] += -λ[3] / ((w - x[3]) ^ 2)
#     h[4, 4] += -λ[3] / ((w - θ[2] / θ[1] * x[4]) ^ 2) * (θ[2] / θ[1]) ^ 2
#     return h
# end