
# Problem 1: Computation of Hugget Model ---------------------------------------------

using LinearAlgebra, Plots, SparseArrays

# Parameters
napts = 1000
nypts = 3
nrpts = 6 # fewer, since we need to solve for V(a,y;r) for all of these
β = 0.9
amin = -0.3
amax = 10.0 # may need to change later

# Grids
ygrid = Array(exp.([-0.1, 0.0, 0.1]))
agrid = Array(range(amin, amax, length = napts))
rgrid = Array(range(-1.0, 0.1, length = nrpts))

# Transition probs
prl = 0.25 / 3.0
prh = 0.75 + prl
P = [prh prl prl;
    prl prh prl;
    prl prl prh]

# Problem A: Using VFI (grid search: pick next agrid point to go to)
function binary_search(f, xgrid, imin, imax)    

    while imax - imin > 2
        il = Int(floor((imin + imax) / 2))
        iu = il + 1
        if f(xgrid[iu]) > f(xgrid[il])
            imin = copy(il)
        else
            imax = copy(iu)
        end
    end

    if (imin + 1) > imax
        return imax, f(xgrid[imax])
    else
        icands = [imin imin + 1 imax]
        fcands = f.(xgrid[icands])
        fmax = maximum(fcands)
        iopt = icands[fcands .== fmax]
        return iopt[1], fmax
    end
    
end

function bellman(anext, apt, ypt, V0, r)
    c = ygrid[ypt] + agrid[apt] * (1.0 + r) - anext
    if c < 0
        return -99_999_999.0 - abs(c) # penalty
    else
        apt_next = findall(agrid .== anext)
        return log(c) + β * (P[ypt, :]' * V0[apt_next[1], :])
    end
end

function vfi_grid(V0, r; ε = 1E-8)
    V1 = copy(V0)
    kp = zeros(Int64, size(V0))
    diff = 10.0
    its = 0
    mxit = 100_000

    while diff > ε && its < mxit
        for j in 1:nypts
            kstar = 1
            for i in 1:napts
                k_opt, v_opt = binary_search(anext -> bellman(anext, i, j, V0, r), agrid, kstar, napts)
                kp[i, j] = k_opt
                V1[i, j] = v_opt
                kstar = copy(k_opt)
            end
        end

        diff = norm(V0 - V1, Inf)
        V0 = copy(V1)
        its += 1
    end

    return V1, kp, diff, its

end 

function q1a()
    # Grids to store solutions
    vfs = zeros(napts, nypts, nrpts)
    pfs = zeros(Int64, size(vfs))
    its = zeros(Int64, nrpts)

    # Initial starting guess
    V0 = zeros(napts, nypts)

    # Solve for each value of r 
    for k in 1:nrpts
        vf, pf, ~, it = vfi_grid(V0, rgrid[k])
        vfs[:, :, k] = vf
        pfs[:, :, k] = pf
        its[k] = it
        V0 = copy(vf) # use previous soln. as guess for next iteration
        println("done with iteration $k")
    end

    return vfs, pfs, its
    
end

@elapsed vfs, pfs, its = q1a() # ~10 mins, but works!

# Get stationary distribution for one rval

function calc_stationary_dist(pf)
    # Calculates the (discrete) stationary distribution ϕ
    # pf is the policy function (indeces)
    nm = napts * nypts
    Pdist = zeros(nm, nm)
    for i = 1:napts
        for j = 1:nypts
            r = (i - 1) * nypts + j
            c1 = (pf[i, j] - 1) * nypts + 1
            cm = (pf[i, j] - 1) * nypts + nypts
            Pdist[r, c1:cm] = P[j, :]
        end
    end
    
    vals, vecs = eigen(Pdist')
    vec_ϕ = real(vecs[:, end])
    vec_ϕ = vec_ϕ ./ sum(vec_ϕ)
    
    # Unpacking
    ϕ = zeros(napts, nypts)
    for i in 1:napts
        for j in 1:nypts
            ϕ[i, j] = vec_ϕ[(i - 1) * nypts + j]
        end
    end
    ϕ_star = sum(ϕ, dims = 2)
    return ϕ_star
    
end

# dtest = calc_stationary_dist(pfs[:, :, 5])

# anets = zeros(nrpts)
# for i in 1:nrpts
#     dist = calc_stationary_dist(pfs[:, :, i])
#     anets[i] = dot(dist, agrid)
# end # all r values lead to too little assets

# r⋆ ∈[0.1, 0.111], found via trial and error
function net_assets(r)
    ~, pf, ~, ~ = vfi_grid(zeros(napts, nypts), r)
    dist = calc_stationary_dist(pf)
    return dot(dist, agrid)
end

using Roots

# r_star = find_zero(net_assets, (0.1, 0.11), Bisection()) # TAKES A LONG TIME
r_star = 0.10720380520169005

# β * (1 + r_star)

# Plot stationary distribution and policy functions for r = 0.1 (rgrid[end])
pf = pfs[:, :, end]
dist = calc_stationary_dist(pf)

plot(agrid, agrid[pf], legend = :bottomright,
    title = "Policy Functions, r = 0.1",
    xlab = "a", ylab = "a_prime")
savefig("assignment_writeups/02_ps2/q1_pol_funs.png")

plot(agrid, dist, legend = false,
    xlims = (-0.5, 1.0),
    title = "Stationary Distribution of Assets",
    xlab = "a")
savefig("assignment_writeups/02_ps2/q1_dist.png")