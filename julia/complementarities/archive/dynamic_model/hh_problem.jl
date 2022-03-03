
#= Summary

Uses value function iteration to solve the household's problem, given a tax function T 
- T may depend on:
    - investment income y
    - interest income Rb
    - wealth w 

=#

using LinearAlgebra, Plots, ApproxFun, Optim, Interpolations, LaTeXStrings, 
    Arpack, Roots, DelimitedFiles

println("*** hh_problem.jl ***")

# Parameters #

# Constants
α = 0.5         # Pr(y>0)
β = 0.95        # discounting
θ_min = 2.0     # down from 2.2
θ_max = 4.0     # down from 4.0
nt = 10
nw = 150        # works with nw = 150
π = ones(nt) / nt
wmin = 0.1
wmax = 12.0
blim = -3.0     # down from -2.0

# Tax Parameters
δ1 = 0.3
δ2 = 0.3
ψ1 = 1.0
ψ2 = 1.0
δ3 = 0.0        # wealth tax (should be same as before now)
υ = 0.2

# Grids
tgrid = Array(range(θ_min, θ_max, length = nt))
wgrid = Array(range(wmin, wmax, length = nw))       # may need to adjust bounds

# Tax
function tax_wp(y, rb, R)
    # Calculates next-period wealth, given the tax structure
    b = rb / R
    if rb >= blim
        if y > 0
            tax = δ1 * y ^ ψ1 + δ2 * (max(rb, 0.0)) ^ ψ2
            wp = (1.0 - δ3) * (y + rb - tax)
        else
            wp = (1.0 - δ3) * (rb + υ - δ2 * (max(rb, 0.0)) ^ ψ2)
        end
    else
        wp = 0.0 # 100% tax on everything
    end
    return wp 

end

function bellman(x, w, i, v0, R)
    # Bellman equation 
    kp, bp = x
    c = w - kp - bp
    θ = tgrid[i]

    # Next-period wealth, given tax 
    w_0 = tax_wp(0.0, R * bp, R)
    w_y = tax_wp(θ * kp, R * bp, R)

    if c < 0.0
        return 99_999.0 + abs(c) # minimizing, so penalty is positive
    elseif kp < 0.0
        return 99_999.0 + abs(kp)
    elseif w_0 < wmin
        return 99_999.0 + abs(w_0)
    elseif w_y < wmin 
        return 99_999.0 + abs(w_y) 
    else
        v_int = LinearInterpolation((tgrid, wgrid), v0,
            extrapolation_bc = Interpolations.Flat()) # create interpolation function
        bel = log(c) + β * dot(α * v_int.(tgrid, w_y) + (1.0 - α) * v_int.(tgrid, w_0), π)
        return -bel # for minimizer
    end 
end

function vfi_int(R; npol = 0)
    # Value and policy functions 
    v0 = 0.1 * ones(nt, nw)
    v1 = similar(v0)
    gk = similar(v0)
    gb = similar(v0)
    
    # Stopping criteria
    diff = 10.0
    its = 0
    mxit = 500
    tol = 1e-6

    # Do the VFI 
    while diff > tol && its < mxit
        for i in 1:nt 
            for j in 1:nw 
                x0 = [0.05, 0.05]
                res = optimize(x -> bellman(x, wgrid[j], i, v0, R), x0)
                v1[i, j] = -res.minimum
                gk[i, j], gb[i, j] = res.minimizer 
            end
        end

        # (Modified) Policy function iteration
        if npol > 0
            w0 = copy(v1)
            w1 = copy(w0)
            for l in 0:npol
                w_int = LinearInterpolation((tgrid, wgrid), w0, 
                            extrapolation_bc = Interpolations.Flat()) # indep. of i, j
                for i in 1:nt
                    for j in 1:nw 

                        # Policy functions
                        kp = gk[i, j]
                        bp = gb[i, j]
                        w = wgrid[j]
                        θ = tgrid[i]
                        w_0 = tax_wp(0.0, R * bp, R)
                        w_y = tax_wp(θ * kp, R * bp, R)

                        # Value from following policy functions 
                        w1[i, j] = log(w - kp - bp) + 
                            β * dot(π, α * w_int.(tgrid, w_y) + (1.0 - α) * w_int.(tgrid, w_0))
                    end
                end
                w0 = copy(w1)
            end
            v1 = copy(w0)
        end

        # println((its, diff))
        diff = norm(v0 - v1, Inf)
        v0 = copy(v1) 
        its += 1
    end

    return v1, gk, gb, diff, its

end

# Stationary distribution
function cal_stationary_dist(gk, gb, R)
    # Calculates the stationary distribution, given 
    # policy functions gk = k′ and gb = b′
    nm = nt * nw
    Pdist = zeros(nm, nm)
    for i in 1:nw
        for j in 1:nt
            
            # Next wealth-grid points, given policy functions
            kp = gk[j, i]
            bp = gb[j, i]
            w_y = tax_wp(tgrid[j] * kp, R * bp, R)
            w_0 = tax_wp(0.0, R * bp, R)

            # Add to Pdist 
            r = (i - 1) * nt + j # row
            if w_y > wmax
                wp_y = nw
                c1_y_h = (wp_y - 1) * nt + 1
                cm_y_h = (wp_y - 1) * nt + nt
                Pdist[r, c1_y_h:cm_y_h] += π .* α
            elseif w_y < wmin
                wp_y = 1
                c1_y_h = (wp_y - 1) * nt + 1
                cm_y_h = (wp_y - 1) * nt + nt
                Pdist[r, c1_y_h:cm_y_h] += π .* α
            else
                wp_y = findfirst(wgrid .> w_y) - 1
                wgt1_y = (w_y - wgrid[wp_y]) / (wgrid[wp_y + 1] - wgrid[wp_y])
                c1_y_l = (wp_y - 1) * nt + 1
                cm_y_l = (wp_y - 1) * nt + nt
                c1_y_h = wp_y * nt + 1
                cm_y_h = wp_y * nt + nt
                Pdist[r, c1_y_l:cm_y_l] += π .* (wgt1_y * α)
                Pdist[r, c1_y_h:cm_y_h] += π .* ((1.0 - wgt1_y) * α)

            end # done with case where y > 0

            if w_0 < wmin
                wp_0 = 1
                c1_0_l = (wp_0 - 1) * nt + 1 
                cm_0_l = (wp_0 - 1) * nt + nt
                Pdist[r, c1_0_l:cm_0_l] += π .* (1.0 - α)
            elseif w_0 > wmax
                wp_0 = nt
                c1_0_l = (wp_0 - 1) * nt + 1 
                cm_0_l = (wp_0 - 1) * nt + nt
                Pdist[r, c1_0_l:cm_0_l] += π .* (1.0 - α)
            else
                wp_0 = findfirst(wgrid .> w_0) - 1 # lower grid points 
                wgt1_0 = (w_0 - wgrid[wp_0]) / (wgrid[wp_0 + 1] - wgrid[wp_0]) # weight to lower grid points
                c1_0_l = (wp_0 - 1) * nt + 1
                cm_0_l = (wp_0 - 1) * nt + nt
                c1_0_h = wp_0 * nt + 1
                cm_0_h = wp_0 * nt + nt
                Pdist[r, c1_0_l:cm_0_l] += π .* (wgt1_0 * (1.0 - α))
                Pdist[r, c1_0_h:cm_0_h] += π .* ((1.0 - wgt1_0) * (1.0 - α))
            end # done with case where y = 0

        end
    end

    # Get invariant distribution
    val, vec = eigs(Pdist', nev = 1, which = :LM)
    vec_ϕ = real(vec)
    vec_ϕ = vec_ϕ ./ sum(vec_ϕ)

    ϕ = zeros(nw, nt)
    for i in 1:nw
        for j in 1:nt
            ϕ[i, j] = vec_ϕ[(i - 1) * nt + j]
        end
    end

    return ϕ

end

# R = 1.2
# @time v1, gk, gb, diff, its = vfi_int(R, npol = 30)
# @show diff, its
# dist = cal_stationary_dist(gk, gb, R)
# plot(wgrid, sum(dist, dims = 2))

# open("julia/dynamic_model/vf.txt", "w") do io
#     writedlm(io, v1)
# end

function net_bor(R)
    v1, gk, gb, diff, its = vfi_int(R, npol = 15)
    if diff > 1e-6 || its == 500
        println("value function did not converge")
    end
    dist = cal_stationary_dist(gk, gb, R)
    return sum(gb' .* dist)
end
# @time net_bor(1.156)

function find_bracket(f)
    # Finds the bracket (x, x + 0.1) containing the zero of 
    # the function f

    a, b = (1.0, 1.1)
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

bkt = find_bracket(net_bor)
@time rstar = find_zero(net_bor, bkt, atol = 1e-3, rtol = 1e-3) # approx ok

v1, gk, gb, diff, its = vfi_int(rstar, npol = 35)
dist = cal_stationary_dist(gk, gb, rstar)

# Back out c 
cpol = repeat(wgrid', nt, 1) - gk - gb

pv = plot(wgrid, v1', legend = false,
    title = L"V(\theta,w)", xlab = L"w")

pk = plot(wgrid, gk', legend = false,
    title = L"k^\prime(\theta,w)", xlab = L"w")

pb = plot(wgrid, gb', legend = false,
    title = L"b^\prime(\theta,w)", xlab = L"w")

pc = plot(wgrid, cpol', legend = false,
    title = L"c(\theta,w)", xlab = L"w")
plot(pv, pk, pb, pc, layout = (2, 2))
savefig("julia/dynamic_model/hh_problem_n150")

# Steady state allocations
# want: for each θ, what are the allocations at the stationary distributions?

k_steady = sum(gk' .* dist, dims = 1) .* 10.0
b_steady = sum(gb' .* dist, dims = 1) .* 10.0
c_steady = sum(cpol' .* dist, dims = 1) .* 10.0 # I think these need to be 10x, to correct for theta dist

plot(tgrid, [c_steady' k_steady' b_steady'], 
    label = [L"c(\theta)" L"k(\theta)" L"b(\theta)"],
    xlab = L"\theta",
    title = "Steady-State Allocations")
savefig("julia/dynamic_model/ss_allocs_n150")

# # Steady-state continuation values (something like promised utility)

# #= 
# Thought:
#     w₁′ ≈ ∫V(θ′,y)dF(θ)
#     w₀′ ≈ ∫V(θ′,0)dF(θ)
# =#

# v_int = LinearInterpolation((tgrid, wgrid), v1, extrapolation_bc = Interpolations.Flat())
# for i in 1:nt
#     for j in 1:nw
#         kp = gk[i, j]
#         bp = gb[i, j]

#     end
# end