
# Problem 2: discrete-state dynamic programming

using LinearAlgebra, Roots, Plots

# Parameters
const β = 0.96
const ε = 10E-7

function payoff(X, q)
    return q - 0.5 * (q ^ 2) - q * ((X + 1.0) ^ -0.2)
end

# Solve for steady-state values
function euler_q(q)
    return 1.0 - q - (100.0q + 1) ^ (-0.2) +
        (0.2β * (100.0q + 1) ^ (-1.2) * q) /
        (1.0 - 0.99β) # should be equal to 0
end

q_ss = find_zero(euler_q, (0.01, 1.0), Bisection())
X_ss = 100.0q_ss

# Grids
xmin = 0.5X_ss
xmax = 1.5X_ss
nxpts = 501
x_grid = range(xmin, xmax, length = nxpts)

# Value function iteration (grid search)

# Binary Search algorithm, to save on function evaluations
function binary_search(f, xgrid, imin::Int64, imax::Int64)    
    
    while imax - imin > 2
        il = Int(floor((imin + imax) / 2))
        iu = il + 1
        if f(xgrid[iu]) > f(xgrid[il])
            imin = copy(il)
        else
            imax = copy(iu)
        end
    end
    
    icands = [imin imin + 1 imax]
    fcands = f.(xgrid[icands])
    fmax = maximum(fcands)
    iopt = icands[fcands .== fmax]
    return iopt[1], fmax
    
end

# Bellman Equation
function bellman(Xprime, X, V_curr)
    q = Xprime - 0.99X
    if q < 0.0
        return -99_999.0 - abs(q)
    else
        pay = payoff(X, q) 
        Xpt_prime = findall(x_grid .== Xprime)
        return pay + β * V_curr[Xpt_prime[1]]
    end
end

# VFI 
function val_fun_it(V0)
    V1 = copy(V0) # updated guess at policy function 
    XP = zeros(Int64, size(V0))
    diff = 10.0
    its = 0
    mxit = 100_000

    while diff > ε && its < mxit
        istar = 1
        for i in 1:nxpts
            qpi, vi = binary_search(xp -> bellman(xp, x_grid[i], V0), x_grid, istar, nxpts)
            V1[i] = vi
            XP[i] = qpi 
            istar = copy(qpi) 
        end

        diff = norm(V0 - V1, Inf) # sup norm 
        V0 = copy(V1)
        its += 1
    end
    return V0, XP, its 

end

# Initial guess
V0 = ones(nxpts) .* payoff(X_ss, q_ss)
@time vf, pf, its = val_fun_it(V0) #  ~11-12 seconds, 294 iterations

# Plotting
p1 = plot(x_grid, vf, legend = false, title = "Value Function",
        xlabel = "X", ylabel = "V(X)")

qf = x_grid[pf] - (0.99 * x_grid)
p2 = plot(x_grid, qf, legend = false, title = "Policy Function",
        xlabel = "X", ylabel = "q(x)")
plot(p1, p2, legend = false)
savefig("assignments/assign_4/q2_vfi.png")


# Policy Function Iteration 

function mod_pfi(V0; npol = 5)
    V1 = copy(V0) # updated guess at policy function 
    w0 = copy(V0)
    w1 = copy(V0) # for PFI
    XP = zeros(Int64, size(V0))
    diff = 10.0
    its = 0
    mxit = 100_000

    while diff > ε && its < mxit
        istar = 1
        for i in 1:nxpts
            qpi, vi = binary_search(xp -> bellman(xp, x_grid[i], V0), x_grid, istar, nxpts)
            V1[i] = vi
            XP[i] = qpi
            istar = copy(qpi)
        end

        # Modified PFI goes here
        w0 = copy(V1)
        for τ in 0:npol
            for i in 1:nxpts
                w1[i] = payoff(x_grid[i], x_grid[XP[i]] - 0.99 * x_grid[i]) + β * w0[XP[i]]
            end # this version works
            w0 = copy(w1)
        end
        V1 = copy(w0)

        diff = norm(V0 - V1, Inf) # sup norm 
        V0 = copy(V1)
        its += 1
    end
    return V0, XP, its 

end

V0 = ones(nxpts) .* payoff(X_ss, q_ss)
@time vf2, pf2, its2 = mod_pfi(V0) #  ~1-2 seconds, 52 iterations 
@time vff, pff, itsf = mod_pfi(V0, npol = 30) # about half a second (!!), 13 iterations

# Plotting
p3 = plot(x_grid, vf2, legend = false, title = "Value Function",
    xlabel = "X", ylabel = "V(X)")

qf2 = x_grid[pf2] - (0.99 * x_grid)
p4 = plot(x_grid, qf2, legend = false, title = "Policy Function",
    xlabel = "X", ylabel = "q(X)")
plot(p3, p4, legend = :false)
savefig("assignments/assign_4/q2_pfi.png")

plot(x_grid, x_grid[pf2])
# Value and policy functions are the same 

# Write policy function to text file 
using DelimitedFiles

open("assignments/assign_4/q2_pol_fxn.txt", "w") do io
    writedlm(io, qf2)
end
