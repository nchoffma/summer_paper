
# Problem 2: Huggett meets McCall

using LinearAlgebra, Plots, Distributions, Arpack

# Parameters
β = 0.96
lam = 0.48  # contact rate
lam_p = 0.10 # separation rate

sig1 = 0.2
sig2 = 0.4

b = 0.4 # unemp benefit

# Construct a grid 
# Using a geometrically-spaced grid 
amin = -0.3
amax = 10.0
napts = 100
Δ = (amax - amin) / ((napts - 1) ^ 2)
agrid = zeros(napts)
for i in 1:napts
    agrid[i] = amin + Δ * ((i - 1) ^ 2)
end

# Bounds on r 
rmin = 0.0
rmax = (1.0 / β) - 1.0

# Tauchen Parameters
rho_w = 0.0
sig_w = sig1
mu_w = 0.0
nwpts = 30

# Discretize offer grid using Tauchen
function normcdf(x) # for convenience
    cdf.(Normal(), x)
end

function tauchen(rho, mu, sig_eps, npts; λ = 2)
    # Tauchen (1986) method for discretizing continuous AR(1) process
    # Default width for the grid is ±2*σ_z
    
    # Bounds on grid 
    sig_z = sig_eps / sqrt(1.0 - rho ^ 2.0)
    zmin = mu - λ * sig_z
    zmax = mu + λ * sig_z
    
    zgrid = Array(range(zmin, zmax, length = npts))     # discrete grid
    mgrid = (zgrid[2:end] + zgrid[1:(end - 1)]) ./ 2.0  # midpoints
    
    # Transition probs
    pi = zeros(npts, npts); 
    for i in 1:npts
        pi[i, 1] = normcdf((mgrid[1] - (1.0 - rho) * mu - rho * zgrid[i]) / sig_eps)
        pi[i, npts] = 1.0 - normcdf((mgrid[npts - 1] - (1.0 - rho) * mu - rho * zgrid[i]) / sig_eps)
        for j in 2:(npts - 1)
            pi[i, j] = normcdf((mgrid[j] - (1.0 - rho) * mu - rho * zgrid[i]) / sig_eps) - 
                normcdf( (mgrid[j - 1] - (1.0 - rho) * mu - rho * zgrid[i]) / sig_eps)
        end
    end

    # Unconditional distribution
    pi_u = pi ^ 10_000
    pi_star = pi_u[1, :]

    return zgrid, pi, pi_star

end

# Tauchen process
# Really what we want is pi_star, the unconditional distribution, which will be 
# the discretized f(w)
wgrid, pi_w, pi_star = tauchen(rho_w, mu_w, sig_w, nwpts, λ = 3)
wgrid = exp.(wgrid) # b/c it was log-normal

# Set up value function iteration
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

function bellman_e(aprime, apt, wpt, ve_0, vu_0, r)
    c = wgrid[wpt] + agrid[apt] * (1.0 + r) - aprime
    if c < 0.0
        return -99_999_999.0 - abs(c) # penalty
    else 
        apt_next = findall(agrid .== aprime)[1]
        return log(c) + β * ((1 - lam_p) * ve_0[apt_next, wpt] + 
            lam_p * vu_0[apt_next])
    end

end

function bellman_u(aprime, apt, ve_0, vu_0, r)
    c = b + agrid[apt] * (1.0 + r) - aprime
    if c < 0.0
        return -99_999_999.0 - abs(c) 
    else 
        apt_next = findall(agrid .== aprime)[1]
        return log(c) + β * ((1 - lam) * vu_0[apt_next] + 
            lam * maximum( [vu_0[apt_next], dot(ve_0[apt_next, :], pi_star)] )) # ∫Max{V(a',w'), V(a',b)}dF(w)
    end
end

function vfi_search(r; ε = 1E-6)

    # Value Functions
    ve_0 = zeros(napts, nwpts) # employed
    vu_0 = zeros(napts, 1)     # unemployed
    ve_1 = copy(ve_0)
    vu_1 = copy(vu_0)

    # Policy Functions
    kp_e = zeros(Int64, size(ve_0))
    kp_u = zeros(Int64, size(vu_0))
    rw = zeros(Int64, napts) # Reservation wages

    # Solver 
    its = 0
    mxit = 1_000
    diff = 10.0

    while diff > ε && its < mxit

        # Update value function for employed 
        for j in 1:nwpts
            kstar_e = 1
            for i in 1:napts
                ap, v = binary_search(ap -> bellman_e(ap, i, j, ve_0, vu_0, r), agrid, kstar_e, napts) 
                ve_1[i, j] = v 
                kp_e[i, j] = ap 
                kstar_e = copy(ap) # starting point for next search 
            end
        end

        # Update value function for unemployed
        kstar_u = 1
        for i in 1:napts
            ap, v = binary_search(ap -> bellman_u(ap, i, ve_1, vu_0, r), agrid, kstar_u, napts)
            kp_u[i] = ap
            vu_1[i] = v 
            kstar_u = copy(ap) # assumes V(a, b) is increasing in a
        end

        # Get the reservation wage
        for i in 1:napts
            valdiff = ve_1[i, :] .- vu_1[i]
            if all(valdiff .< 0.0)
                rw[i] = nwpts
            else
                rw[i] = findfirst(valdiff .> 0.0)
            end
        end

        diff = maximum([norm(ve_0 - ve_1, Inf) norm(vu_0 - vu_1, Inf)])
        its += 1
        ve_0 = copy(ve_1)
        vu_0 = copy(vu_1)

    end
    
    return ve_0, kp_e, vu_0, kp_u, rw, its, diff
end

# Get stationary dist
function calc_stat_dist(kp, rw) 
    
    nypts = nwpts + 1
    nm = napts * nypts
    Pdist = zeros(nm, nm) # this thing is huge 
    # Employed worker
    for i in 1:napts
        for h in 2:nypts
            r = (i - 1) * nypts + h
            c_stay = (kp[i, h] - 1) * nypts + h 
            c_sep = (kp[i, h] - 1) * nypts + 1
            Pdist[r, c_stay] = 1.0 - lam_p
            Pdist[r, c_sep] = lam_p
        end
    end
    
    # Unemployed Worker 
    for i in 1:napts # h = 1
        r = (i - 1) * nypts + 1
        c_unemp = (kp[i, 1] - 1) * nypts + 1
        Pdist[r, c_unemp] = (1.0 - lam) + lam * sum(pi_star[1:(rw[i] - 1)])
        c_emp_st = (kp[i, 1] - 1) * nypts + rw[i] + 1
        c_emp_en = (kp[i, 1] - 1) * nypts + nypts
        Pdist[r, c_emp_st:c_emp_en] .= lam .* pi_star[rw[i]:end]
    end
    
    val, vec = eigs(Pdist', nev = 1, which = :LM) # much quicker method with Arpack
    vec_ϕ = real(vec)
    vec_ϕ = vec_ϕ ./ sum(vec_ϕ)
    
    ϕ = zeros(napts, nypts)
    for i in 1:napts
        for j in 1:nypts
            ϕ[i, j] = vec_ϕ[(i - 1) * nypts + j]
        end
    end
    
    # ϕ_star = sum(ϕ, dims = 2)
    return ϕ
end

# Solve for market-clearing interest rate

# @elapsed ve_0, kp_e, vu_0, kp_u, rw, its, diff = vfi_search(rmin)
# With napts = 100 and nwpts = 30, this takes ~15-20 secs

function net_assets(r)
    ~, kp_e, ~, kp_u, rw, ~, ~ = vfi_search(r)
    kp = [kp_u kp_e]
    dist = calc_stat_dist(kp, rw)
    adist = sum(dist, dims = 2)
    return dot(agrid, adist)
end

using Roots

# r_star = find_zero(net_assets, (-0.05, 0.05))
# r_star = -0.029563028383787934 # odd that it's negative, but it clears the market. 
# r_star = -0.01030969294392331  # with b = 0.4


# Equilibrium soln.
# ve, kp_e, vu, kp_u, rw, ~, ~ = vfi_search(r_star)
# kp = [kp_u kp_e]
# eq_dist = calc_stat_dist(kp, rw)

#= Making nice plots
pve = plot(agrid, ve, legend = false, 
        title = "Value Functions (emp.)",
        xlab = "a")
pvu = plot(agrid, vu, legend = false, 
        title = "Value Function (unemp.)",
        xlab = "a")
plot(pve, pvu, layout = (1, 2), legend = false)
# savefig("assignment_writeups/02_ps2/q2_vfs.png")

pae = plot(agrid, agrid[kp_e], 
    title = "Policy Functions (emp.)",
    xlab = "a")
pau = plot(agrid, agrid[kp_u], 
    title = "Policy Function (unemp.)",
    xlab = "a")
plot(pae, pau, layout = (1, 2), legend = false)
# savefig("assignment_writeups/02_ps2/q2_pfs.png")
=#

# adist = sum(eq_dist, dims = 2)
# ydist = sum(eq_dist, dims = 1)

# pad = plot(agrid, adist, title = "Asset Distribution",
#     xlab = "a")
# pyd = plot(wgrid, ydist[2:end], 
#     title = "Income Distribution",
#     xlab = "y")
# plot(pad, pyd, legend = false)
# # savefig("assignment_writeups/02_ps2/q2_dists.png")

# plot(agrid, wgrid[rw], legend = false, 
#     title = "Reservation wage w*(a)",
#     xlab = "a")
# # savefig("assignment_writeups/02_ps2/q2_rws.png")

# urate = ydist[1]

# Compare reservation wage policies
r_star1 = -0.029563028383787934 # odd that it's negative, but it clears the market. 
r_star2 = -0.01030969294392331  # with b = 0.4

ve1, kp_e1, vu1, kp_u1, rw1, ~, ~ = vfi_search(r_star1)
ve2, kp_e2, vu2, kp_u2, rw2, ~, ~ = vfi_search(r_star2)

kp1 = [kp_u1 kp_e1]
kp2 = [kp_u2 kp_e2]

eq_dist1 = calc_stat_dist(kp1, rw1)
eq_dist2 = calc_stat_dist(kp2, rw2)

adist1 = sum(eq_dist1, dims = 2)
ydist1 = sum(eq_dist2, dims = 1)
adist2 = sum(eq_dist1, dims = 2)
ydist2 = sum(eq_dist2, dims = 1)

plot(agrid, [adist1 adist2], 
    label = ["b = 0.3" "b = 0.4"])

plot(wgrid, [ydist1[2:end] ydist2[2:end]])

plot(agrid, [wgrid[rw1] wgrid[rw2]],
    label = ["b = 0.3" "b = 0.4"],
    legend = :bottomright,
    title = "Reservation Wages")
savefig("assignment_writeups/02_ps2/q2_altb.png")

