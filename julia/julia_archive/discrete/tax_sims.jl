
#= 
Replicates numerical simulations of tax schedule from New Dynamic Public Finance: A User's Guide
Golosov, Tyvinski, and Werning
=#

using LinearAlgebra, Plots, Optim, NLopt

# Baseline Parameters
R1 = 1.0        # interest rate
R2 = 1.0        # t = 2
β = 1.0         # discounting
N1 = 3          # no. of types (should be 50, but start small)
N2 = 2          # N₂(i) for each i in 1:N1
π_1 = 1.0 / N1  # type probability
γ = 2.0         # liesuer disutility
G = 0.0         # gov't spending (both pds)
K1 = 0.0        # endowed capital
σ = 1.0         # CES utility

# Initial skills
θ_min = 0.1
θ_max = 1.0
θ_1 = Array(range(θ_min, θ_max, length = N1))

# period 2 shocks
α_1 = 1.0       # these multiply θ₁
α_2 = 0.5       # each w/p 1/2

function util(c)
    if c < 0
        u = -999.99 - abs(c)
    elseif σ == 1.0
        u = log(c)
    else
        u = (c ^ (1.0 - σ)) / (1.0 - σ)
    end
end

vl(l) = -l ^ γ # disutility of labor

function ic_rhs(X)
    # Computes the rhs of the IC for all (i, iᵣ, jᵣ) combns,
    # rhs[i, iᵣ, jᵣ] gives payoff to being type i, and claiming 
    # iᵣ at t = 0 and jᵣ at t = 1. 

    # Unpack guess 
    C = X[:, 1:N1]
    Y = X[:, (N1 + 1):end]

    # Compute RHS of IC 
    rhs = zeros(N1, N1, N2)
    for ic in 1:N1          # i (indexes actual types θ, which determine effort to fake)
        for i in 1:N1       # iᵣ
            for j in 1:N2   # jᵣ
                pay0 = util(C[i, 1]) + vl(Y[i, 1] / θ_1[ic])
                epay1 = β * 0.5 * ( util(C[i, j + 1]) + vl(Y[i, j + 1] / (α_1 * θ_1[ic])) + 
                    util(C[i, j + 1]) + vl(Y[i, j + 1] / (α_2 * θ_1[ic])) )
                rhs[ic, i, j] = pay0 + epay1
            end
        end
    end
    return rhs
end

function plan_obj(X)
    # Computes the value of planner's objective (total expected utility)

    # Unpack guess 
    C = X[:, 1:N1]
    Y = X[:, (N1 + 1):end]

    # Compute expected utility
    pay = 0.0
    for i in 1:N1
        pay0 = util(C[i, 1]) + vl(Y[i, 1] / θ_1[i])
        epay1 = β * 0.5 * ( util(C[i, 2]) + vl(Y[i, 2] / (α_1 * θ_1[i])) + 
            util(C[i, 3]) + vl(Y[i, 3] / (α_2 * θ_1[i])) )
        pay += pay0 + epay1
    end
    return pay / N1 # equal probability of each type 

end

function resource_const(X)
    # Computes the value of (RC)
    # want this to be ≤ 0 (if < 0, c < y on avg.)

    # Unpack guess 
    C = X[:, 1:N1]
    Y = X[:, (N1 + 1):end]

    rc = 0.0
    for i in 1:N1
        rc += C[i, 1] - Y[i, 1] +           # t = 1
            0.5 * (C[i, 2] - Y[i, 2]) +     # t = 2, j = 1
            0.5 * (C[i, 3] - Y[i, 3])       # t = 2, j = 2
    end
    return rc / N1 

end

function incent_const(X)
    # Computes the value of (IC), using the computed 3D matrix
    # from ic_rhs
    # Assume LHS - RHS, so this should be ≥ 0 (vector)

    # Unpack guess 
    C = X[:, 1:N1]
    Y = X[:, (N1 + 1):end]

    # Compute RHS
    incents = ic_rhs(X)

    # Compute IC 
    ics = zeros(N1, N1, N2)
    for i in 1:N1
        for ir in 1:N1
            for jr in 1:N2
                plan_pay0 = util(C[i, 1]) + vl(Y[i, 1] / θ_1[i])
                plan_epay1 = β * 0.5 * (util(C[i, 2]) + vl(Y[i, 2] / (α_1 * θ_1[i])) + 
                    util(C[i, 3]) + vl(Y[i, 3] / (α_2 * θ_1[i])))
                ics[i, ir, jr] = plan_pay0 + plan_epay1 - incents[i, ir, jr]
            end
        end
    end
    return vec(ics) # order should not matter 

end

# Penalty method

function penalty_obj(X, P)
    obj = -plan_obj(X)
    rc = resource_const(X)
    ic = incent_const(X)

    return -obj + 0.5 * P * (rc ^ 2 + sum(min.(ic, 0.0) .^ 2)) # ic can be ≥ 0 
end

# Guesses
C = repeat(θ_1, 1, 3)
Y = C ./ 2.0
# K = C[:, 1] - Y[:, 1] # can just back out
X0 = [C Y] # full soln. 

# result = optimize(X -> penalty_obj(X, 100.0), X0, LBFGS()) # does not converge

# Second attempt: try NLopt
# Problem: this requires vector input, not matrix, so 
# constraints/objective need to be transformed
