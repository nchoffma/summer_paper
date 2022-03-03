
# Problem 3: Continuous State
# Idea: want V(X) for arbitrary X∈[X̲, X̄]

using Optim, LinearAlgebra, ApproxFun, Plots

# Parameters
const β = 0.96
const ε = 10E-7
const q_ss = 0.573468954520529 # already found in part 1
const X_ss = 100q_ss
const xmin = 0.5X_ss
const xmax = 1.5X_ss

# Functions
function payoff(X, q)
    return q - 0.5 * (q ^ 2) - q * ((X + 1.0) ^ -0.2)
end

function cheb_nodes(m_cheb, n_cheb, a, b)
    # Returns nodes for Chebyshev approx in interval [a,b]
    # using m_cheb nodes and polynomial of degree n_cheb
    z(j) = -cos(((2j - 1) / 2m_cheb) * pi)
    zk = [z(j) for j in 1:m_cheb]
    xk = (b - a) / 2.0 .* (zk .+ 1.0) .+ a
    return zk, xk
end

function cheb_approx(m_cheb, n_cheb, fvals; a = xmin, b = xmax)
    # n_cheb is the polynomial order
    # m_cheb is the number of nodes (m_cheb ≥ n_cheb + 1)
    
    # fvals must be of length m_cheb
    if length(fvals) != m_cheb
        error("fvals must have length m_cheb")
    end
    
    zk, xk = cheb_nodes(m_cheb, n_cheb, a, b)
    # already have yk (fvals)
    
    # Chebyshev Polynomials
    Tz = zeros(m_cheb, n_cheb + 1)
    Tz[:, 1] .= 1.0
    Tz[:, 2] = zk
    for i in 1:m_cheb
        for j in 3:(n_cheb + 1)
            Tz[i, j] = 2.0 * zk[i] * Tz[i, j - 1] - Tz[i, j - 2]
        end
    end
    
    # Coefficients
    if m_cheb == (n_cheb + 1)
        ai = Tz \ fvals # if m = n + 1, can solve directly 
    else
        ai = (Tz' * Tz) \ Tz' * fvals # otherwise, need least squares
    end
    
    # Approximate function
    function fhat(x)
        z = -1.0 + 2.0 * (x - a) / (b - a)
        T = zeros(n_cheb + 1)
        T[1] = 1.0
        T[2] = z
        for i in 3:(n_cheb + 1)
            T[i] = 2.0 * z * T[i - 1] - T[i - 2]
        end
        return ai' * T
        
    end
    
    return xk, ai, fhat # return the nodes, coeffs, and function 
    
end # this function works

# Bellman operator
function bellman(xp, x, vhat)
    # vhat is now a function, not an array 
    q = xp - 0.99x 
    return payoff(x, q) + β * vhat(xp)

end

# VFI
function vfi_cont(fhat0, m_cheb, n_cheb)
    # fhat0 is the initial guess for the value at nodes  
    # the value function 
    fhat1 = copy(fhat0) # updated values
    xopt = copy(fhat0)  # policy function X′(X)
    diff = 10.0
    its = 0
    mxit = 1_000
    ~, xg0 = cheb_nodes(m_cheb, n_cheb, xmin, xmax)

    while diff > ε && its < mxit
        ~, ac, vhat0 = cheb_approx(m_cheb, n_cheb, fhat0) # get the x grid and V̂(X)

        for i in 1:m_cheb
            res = maximize(xp -> bellman(xp, xg0[i], vhat0), 0.99 * xg0[i], xmax) # not letting it select q < 0
            fhat1[i] = Optim.maximum(res)
            xopt[i] = Optim.maximizer(res) # policy fxn is X′ vals, not indexes
        end

        diff = norm(fhat1 - fhat0, Inf)
        fhat0 = copy(fhat1) # update
        its += 1
    end

    ~, ~, xpol = cheb_approx(m_cheb, n_cheb, xopt)
    ~, ~, vhat = cheb_approx(m_cheb, n_cheb, fhat0)
    return vhat, xpol, its, diff

end
 
# 1) Report starting coeffs 
m_cheb, n_cheb = 5, 4 # nodes, order 
z0, x0 = cheb_nodes(m_cheb, n_cheb, xmin, xmax)
vstart = log.(x0)
~, a0, f0 = cheb_approx(m_cheb, n_cheb, vstart)

# 2) Solve using this method
@time vhat, xp, its, diff = vfi_cont(vstart, m_cheb, n_cheb)

xpts = range(xmin, xmax, length = 501)
qpol = xp.(xpts) - (0.99 * xpts)

p1 = plot(xpts, vhat.(xpts), 
    title = "Value Function",
    legend = false,
    xlabel = "X",
    ylabel = "V(X)")
p2 = plot(xpts, qpol,
    title = "Policy Function",
    legend = false,
    xlabel = "X",
    ylabel = "q(X)")
plot(p1, p2, legend = false)
# savefig("assignments/assign_4/q3_vfi_cheb.png")

# 3) Plot this pol fxn with previous
using DelimitedFiles
qpol_disc = readdlm("assignments/assign_4/q2_pol_fxn.txt", '\t', Float64, '\n')
plot(xpts, [qpol qpol_disc],
    label = ["Continuous State" "Discrete State"],
    title = "Policy Functions",
    legend = :topleft,
    xlabel = "X", ylabel = "q(X)")
savefig("assignments/assign_4/q3_pol_comp.png")

# 4) Solve with m_cheb = 10
m_cheb, n_cheb = 10, 4 # nodes, order 
z0, x0 = cheb_nodes(m_cheb, n_cheb, xmin, xmax)
vstart = log.(x0)

@time vhat10, xp10, its10, diff10 = vfi_cont(vstart, m_cheb, n_cheb)

qpol10 = xp10.(xpts) - (0.99 * xpts)
p3 = plot(xpts, vhat10.(xpts), 
    title = "Value Function",
    legend = false,
    xlabel = "X",
    ylabel = "V(X)")
p4 = plot(xpts, qpol10,
    title = "Policy Function",
    legend = false,
    xlabel = "X",
    ylabel = "q(X)")
plot(p3, p4, legend = false)
savefig("assignments/assign_4/q3_vfi_10p.png")