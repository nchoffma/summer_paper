#= 

Solves the planner's infinite-horixon CMP, 
with θ∼N(μ,σ) 

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, NLsolve, 
    ForwardDiff, Optim, LaTeXStrings, Roots, 
    FastGaussQuadrature, Printf, DelimitedFiles, ApproxFun

gr()
println("******** cpp_inf_horizon.jl ********")

# Parameters
β = 0.9         # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.
# ϵ = 5.9 # highest successful is 5.8
# ϵ = 5.8

R = 1.1

# Analytically, c(θ,0) = ctest, and A(0) = ctest / (1. - β)
ctest = (1. / (β * R)) ^ (1. / (1. - β))
Atest =  ctest /  (1. - β)

# Path to write results 
solnpath = "julia/complementarities/results/normal/eps_experiments/"
# solnpath = "julia/complementarities/results/normal/"

# Infinite horizon
shft = 1. - β          # shifter (inifinite horizon) on allocations
shft_w = 1.            # shifter on promise utility

p_tilde(x) = exp(-1. / ϵ * shft * shft_w * x) # continuation state 

# Autaurky allocations
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 1. / (1. + β) * w * ones(nt)
klf = β / (1. + β) * w * ones(nt)
c1_lf = tgrid .* klf
wp_lf = log.(c1_lf)
Ulf = log.(c0_lf) + β * wp_lf
μlf = 0.05 * ones(nt)
μlf[[1 end]] .= 0. 

# State space for p̄
pL = 0.     # absorbing barrier 
# pH = 0.5  # works here
# pH = 0.6  # here too 
# pH = 0.7  # yes
pH = 0.8  # yes 
# pH = 0.9 # yes
# pH = 0.9  # yes
# pH = 0.91 # no

m_cheb = 5
S0 = Chebyshev(pL..pH)
p0 = points(S0, m_cheb)

# Functions for truncated normal dist 
tmean = (θ_max + θ_min) / 2
tsig = 0.3
function tnorm_pdf(x)
    pdf.(truncated(Normal(tmean, tsig), θ_min, θ_max), x)
end

function tnorm_cdf(x) # for convenience
    cdf.(truncated(Normal(tmean, tsig), θ_min, θ_max), x)
end

function tnorm_fprime(x)
    ForwardDiff.derivative(tnorm_pdf, x)
end 

# Distribution for output 
function fdist(x)

    # # Uniform
    # L = θ_min
    # H = θ_max 
    # cdf = (x - L) / (H - L)
    # pdf = 1. / (H - L)
    # fpx = 0.

    # Truncated normal
    cdf = tnorm_cdf(x)
    pdf = tnorm_pdf(x)
    fpx = tnorm_fprime(x)
    
    return cdf, pdf, fpx
    
end

# Gauss-legendre quadrature
function gauss_leg(f, n, a, b)
    # Uses Gauss-Legendre quadrature with n nodes over [a,b]
    # Upshot: can be quicker
    # Downside: no idea how accurate the solution is (not adaptive)
    
    # Get nodes and weights
    xi, ωi = gausslegendre(n)
    
    # Compute approximation 
    x_new = (xi .+ 1) * (b - a) / 2.0 .+ a # change of variable
    approx = (b - a) / 2.0 * (ωi' * f.(x_new))
    return approx
    
end

function find_bracket(f; bkt0 = (-0.3, -0.2), step = 0.1, print_every = false)
    # Finds the bracket (x, x + step) containing the zero of 
    # the function f
    
    a, b = bkt0
    its = 0
    mxit = 50_000

    if print_every
        println([a, f(a), b, f(b)])
    end
    
    while f(a) * f(b) > 0.0 && its < mxit
        if f(a) > 0.0           # f(a), f(b) positive
            if f(a) > f(b)      # f decreasing
                a = copy(b)
                b += step
            else                # f increasing
                b = copy(a)
                a -= step
            end
        else                    # f(a), f(b) negative
            if f(a) > f(b)      # f increasing
                b = copy(a)
                a -= step
            else                # f decreasing
                a = copy(b)
                b += step
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

function eval_pkc(γ, pbar, A0, ubkt; 
    return_all = false, 
    extrap = true,
    return_fxns = false)

    # Build the necessary functions

    # Extrapolation function 
    function extrap_pbar(pb)
        # If p̄ > pH (only applies if this is the case!), we 
        # linearly extrapolate At near pH to get the right slope

        del = 0.05 # distance to determine slope
        slp = (A0(pH) - A0(pH - del)) / del
        ext_val = A0(pH) + slp * (pb - pH)
        return ext_val 
    end

    # FOCs
    function foc_k(x, U, μ, θ)
        
        c, wp, k = x
        ptilde = p_tilde(wp)
        phat = (θ * k) ^ (-1. / ϵ)
        pnext = ptilde * pbar
    
        if extrap && pnext > pH 
            Apn = extrap_pbar(pnext)
        else
            Apn = A0(pnext)
        end
    
        # FOC vector
        focs = zeros(eltype(x), 3)
        focs[1] = 1. - Apn * shft * exp(shft * wp) / (β * R * c) - μ * k / (θ * c ^ 2)
        focs[2] = 1. - 1. / R * pbar * phat * θ + μ / (θ * c)
        focs[3] = log(c) + β * wp - U 
        
        return focs
        
    end
    
    function newton_k(U, μ, θ, i)
        x0 = [c0_lf[i] wp_lf[i] klf[i]]'
        mxit = 500
        tol = 1e-10
        diff = 10.0
        its = 0
        fail = 0
        
        while diff > tol && its < mxit
            focs = foc_k(x0, U, μ, θ)
            dfocs = ForwardDiff.jacobian(x -> foc_k(x, U, μ, θ), x0)
            diff = norm(focs, Inf)
            d = dfocs \ focs 
            while minimum(x0[[1 3]] - d[[1 3]]) .< 0.0 # c0 and k need to be positive, w need not be
                d = d / 2.0
                if maximum(d) < tol
                    fail = 1
                    println("warning: newton failed")
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
    
    function alloc_single(U, μ, t)
        
        it = findmin(abs.(tgrid .- t))[2] # for finding initial guess
        
        x, f = newton_k(U, μ, t, it)
        c0, wp, k = x 
        return c0, wp, k
        
    end
    
    function de_system!(du, u, p, t)
        U, μ = u
        c0, wp, k = alloc_single(U, μ, t)
        ptilde = p_tilde(wp)
        phat = (t * k) ^ (-1. / ϵ)
        ~, ft, fpt = fdist(t)
        pnext = pbar * ptilde

        if extrap && pnext > pH 
            Apn = extrap_pbar(pnext)
        else
            Apn = A0(pnext)
        end

        du[1] = k / (c0 * t)
        du[2] = γ - Apn / (β * R) * shft * exp(shft * wp) - fpt / ft * μ
    end
    
    function tax_shoot(U_0)
        u0 = [U_0 0.0]
        tspan = (θ_min, θ_max)
        prob = ODEProblem(de_system!, u0, tspan)
        gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
        return gpath.u[end][2] # want to get μ(θ̲) = 0
        
    end


    # Promise-keeping (U* = 0)
    function pkc_integrand(t, gpath)
        U, mu = gpath(t)
        Ft, ft, fpt = fdist(t)
        return U * ft
    end

    # Optimal allocations along the path
    function opt_allocs(gpath)
        cp = zeros(nt)
        kp = similar(cp)
        wp = similar(cp)
        mup = similar(cp)
        for i in 1:nt
            U, mu = gpath(tgrid[i])
            cp[i], wp[i], kp[i] = alloc_single(U, mu, tgrid[i])
            mup[i] = mu

        end
        return cp, wp, kp, mup
    end

    # Value of A_t-1
    function at_integrand(t, gpath)
        U, mu = gpath(t)
        Ft, ft, fpt = fdist(t)
        ct, wpt, kt = alloc_single(U, mu, t)
        phat = t * kt ^ (-1. / ϵ)
        pnext = pbar * phat
        ptilde = p_tilde(wpt)
        pbp = pbar * ptilde

        if extrap && pbp > pH 
            atp1 = extrap_pbar(pbp)
        else
            atp1 = A0(pbar * ptilde)
        end
        
        return ft * (ct + kt + 1 / R * ( atp1 * exp(shft * wpt) - t * pnext * kt ) )
    end

    # bkts = ubkt
    stp = ubkt[2] - ubkt[1]
    # println((ubkt, stp))
    bkt = find_bracket(um -> tax_shoot(um), bkt0 = ubkt, step = stp)
    Umin_opt = find_zero(x -> tax_shoot(x), bkt)

    u0 = [Umin_opt 0.0]
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan)
    gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 

    pkc = gauss_leg(t -> pkc_integrand(t, gpath), 50, θ_min, θ_max)

    if return_all

        # Return all elements of the solution 
        csol, wsol, ksol, musol = opt_allocs(gpath)
        aupd = gauss_leg(t -> at_integrand(t, gpath), 50, θ_min, θ_max)
        return gpath, csol, wsol, ksol, aupd, musol, bkt
    elseif return_fxns

        # Return functions built along the way (used for PFI step)
        # println("got to fxns")
        return gpath, extrap_pbar, foc_k, newton_k, alloc_single, bkt 
    else

        # Just return the pkc value (used for zeroing)
        return pkc
    end

end

function update_at(At, gam_brackets, ubkt1;
    extp_t = true,
    update_ubkt = true)

    # If length of gam_brackets is one, this is the initial bracket
    # If length of gam_brackets is m_cheb, it gives the bracket for each pbar

    at1 = zeros(m_cheb)
    gams = similar(at1)
    csol = zeros(nt, m_cheb)
    ksol = similar(csol)
    wsol = similar(csol)
    musol = similar(csol)

    if length(gam_brackets) == 1
        gam_bkt = gam_brackets[1]
    end
    ubkt = ubkt1
    
    for i in 1:m_cheb

        # Build current pbar function
        fpb(gm) = eval_pkc(gm, p0[i], At, ubkt, extrap = extp_t)

        # Find optimal γ and get allocations
        if length(gam_brackets) > 1
            gam_bkt = gam_brackets[i]
        end

        # println("ip = $i, γ bracket = $gam_bkt")
        stp = gam_bkt[2] - gam_bkt[1]
        bkt_int = find_bracket(fpb, bkt0 = gam_bkt, step = stp)
        gstar = find_zero(fpb, bkt_int, 
            atol = 1e-6, rtol = 1e-6,
            xatol = 1e-6, xrtol = 1e-6,
            Roots.A42()) 
        
        gams[i] = gstar
        gpath, csol[:, i], wsol[:, i], ksol[:, i], aupd, musol[:, i], ubkt_p = eval_pkc(gstar, 
            p0[i], At, ubkt, return_all = true, extrap = extp_t)
        gprint = round(gstar, digits = 4)
        ubprint = round.(ubkt_p, digits = 4)
        println("pbar value $i: γ* = $gprint, Ubkt = $ubprint")
        at1[i] = aupd

        if update_ubkt
            ubkt = ubkt_p .- ucush # give some cushion
        end
        
        if length(gam_brackets) == 1
            gam_bkt = bkt_int
        end
        
    end
    return gams, csol, wsol, ksol, at1, musol
end

function pfi_step(gstars, a1, a0f, ubkt0; npol = 20, extrap = false)
    # Uses modified PFI (HIA) to get updated A(⋅) function 
    # Default is 20 steps 
    # Note: a0f is the prior starting function, which is needed to recover the functions

    # a1 is the values at the m_cheb Chebyshev nodes
    # ahat0 and ahat1 will be the same idea

    ahat0 = copy(a1)
    ahat1 = similar(ahat0)
    for hk in 1:npol
        ahat1 = zeros(m_cheb)
        Ahat = Fun(S0, ApproxFun.transform(S0, ahat0))
        ubkt = ubkt0
        for j in 1:m_cheb
            pbar = p0[j]
            gpath, extrap_pbar, foc_k, newton_k, alloc_single, bkt_n = eval_pkc(
                    gstars[j], pbar, a0f, ubkt, return_fxns = true) # get the functions
            
            function extrap_pbar_hia(pb)
                # Analogous to the extrapolation function above, but 
                # uses the interim function Ahat for consistency

                del = 0.05 # distance to determine slope
                slp = (Ahat(pH) - Ahat(pH - del)) / del
                ext_val = Ahat(pH) + slp * (pb - pH)
                return ext_val 
            end

            function at_integrand_step(t, gpath)

                # Analogous to the at_integrand used in solution, but instead uses 
                # intermediate approximation Ahat 
                U, mu = gpath(t)
                Ft, ft, fpt = fdist(t)
                ct, wpt, kt = alloc_single(U, mu, t)
                phat = t * kt ^ (-1. / ϵ)
                pnext = pbar * phat
                ptilde = p_tilde(wpt)
                pbp = pbar * ptilde
        
                if extrap && pbp > pH 
                    atp1 = extrap_pbar_hia(pbp)
                else
                    atp1 = Ahat(pbar * ptilde)
                end
                
                return ft * (ct + kt + 1 / R * ( atp1 * exp(shft * wpt) - t * pnext * kt ) )
            end

            ahat1[j] = gauss_leg(t -> at_integrand_step(t, gpath), 50, θ_min, θ_max)
            ubkt = bkt_n

        end
        ahat0 = copy(ahat1)

    end

    return ahat0

end

function get_gbkts(gs)
    gbs = [floor.(gs, digits = 3) ceil.(gs, digits = 3)]
    gbks = Tuple(Tuple(gbs[i,:]) for i in 1:m_cheb)

    return gbks
end

# Iteration
function iterate_at(a0, gam_bkt0, Ubkt0; 
    updateU = true,
    npfi = 25,
    gams_to_update = "first")

    a1 = similar(a0)
    gstars = zeros(m_cheb)
    csol = zeros(nt, m_cheb)
    wsol = similar(csol)
    ksol = similar(csol)

    norm_A = 10.
    tol = 1e-6
    its = 0
    mxit = 1000

    while norm_A > tol && its < mxit
        println("Iteration $its")

        # Update 
        Afunc = Fun(S0, ApproxFun.transform(S0, a0))
        gstars, csol, wsol, ksol, a1, ~ = update_at(Afunc, gam_bkt0, Ubkt0,
            extp_t = true, update_ubkt = updateU)

        # Find the new brackets for U0, γ 
        Usol = log.(csol) + β * wsol 
        Ubkt0 = (floor(Usol[1,1], digits = 2), 
                ceil.(Usol[1,1], digits = 2)) .- ucush # need to ensure we're approaching root from below

        gbs_upd = [floor.(gstars, digits = 3) ceil.(gstars, digits = 3)]

        if gams_to_update == "first"
            gam_bkt0 = (Tuple(gbs_upd[1,:]), )
        elseif gams_to_update == "all"
            gam_bkt0 = Tuple(Tuple(gbs_upd[i,:]) for i in 1:m_cheb)
        else
            error("gams_to_update must be one of 'first' or 'all'. ")
        end

        if npfi > 0
            println("Going to PFI")
            Ubkt_pfi = (floor(Usol[1,1], digits = 2), 
                ceil.(Usol[1,1], digits = 2))
            a1 = pfi_step(gstars, a1, Afunc, Ubkt_pfi, npol = npfi, extrap = true)
        end

        # Evaluate convergence
        norm_A = norm(a1 - a0, Inf) 
        println("norm_A = $norm_A")
        if norm_A > 0.2
            rdA = round.(a1, digits = 4)
            println("a1: $rdA")
        end

        # if npfi == 0 && norm_A < 1e-4 # go to PFI once we're very close
        #     npfi = 25
        # end

        a0 = copy(a1)
        its += 1

    end

    return gstars, csol, wsol, ksol, a1

end

# Read in prior solution
at0 = vec(readdlm(solnpath * "norm_a1c_0.9_1.1_0.8_0.3_4.0.txt"))
gs0 = readdlm(solnpath * "norm_gstars_0.9_1.1_0.8_0.3_4.0.txt") 

if length(at0) != m_cheb
    println("Interpolating starting guess")
    A0s = Fun(S0, ApproxFun.transform(S0, at0))
    at0 = A0s.(p0)
    g0s = Fun(S0, ApproxFun.transform(S0, gs0))
    gs0 = g0s.(p0)
else
    gbks0 = get_gbkts(gs0) 
end

# Ubkt_start = (-0.23, -0.22)
# Ubkt_start = (-0.33, -0.32) 
# Ubkt_start = (-0.47, -0.46) 
Ubkt_start = (-0.63, -0.62) 
# Ubkt_start = (-0.83, -0.82) 
# Ubkt_start = (-0.99, -0.98) 
# Ubkt_start = (-1.36, -1.35) 

ucush = 0.

# at0 = ones(m_cheb) * Atest
@time gstars_c, csol_c, wsol_c, ksol_c, a1_c = iterate_at(at0, gbks0, 
    Ubkt_start, gams_to_update = "all", npfi = 0);
Usol_c = log.(csol_c) + β * wsol_c
Ac = Fun(S0, ApproxFun.transform(S0, a1_c))

# Save solution for future starting guesses 
ac_fname = solnpath * "norm_a1c_" * string(β) * "_" * string(R) * "_" * string(pH) * "_" * string(tsig) * "_" * string(ϵ) * ".txt"
gs_fname = solnpath * "norm_gstars_" * string(β) * "_" * string(R) * "_" * string(pH) * "_" * string(tsig) * "_" * string(ϵ) * ".txt"
writedlm(ac_fname, a1_c)
writedlm(gs_fname, gstars_c)

# p̄'
function pbar_next(wpol)
    pbarp = zeros(nt, m_cheb)
    for i in 1:nt, j in 1:m_cheb
        ptilde = p_tilde(wpol[i, j])
        pbarp[i, j] = ptilde * p0[j]
    end
    return pbarp
end

pbar_prime_c = pbar_next(wsol_c);

# Plot solutions
pU = plot(tgrid, Usol_c,
    xlab = L"\theta",
    title = "Utilities",
    ylab = L"U(\theta;\bar{p})",
    legend = false);

ppb = plot(tgrid, pbar_prime_c,
    xlab = L"\theta",
    title = "Next-period state",
    ylab = L"\bar{p}^\prime (\theta;\bar{p})",
    legend = false);

p0fine = range(pL, pH, length = 50)
pat1 = plot(p0fine, Ac.(p0fine),
    title = L"A(\bar{p})",
    xlab = L"\bar{p}",
    legend = false);

pc = plot(tgrid, csol_c, 
    title = L"c(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false);

pk = plot(tgrid, ksol_c, 
    title = L"k^\prime(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false);

pw = plot(tgrid, wsol_c, 
    title = L"w^\prime(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false);

figpath = "julia/complementarities/results/normal/eps_experiments/figures/eps4/"
display(plot(plot(ppb, pat1), pU, layout = (2,1)))
savefig(figpath * "inf_soln_tnorm.png")

display(plot(plot(pc, pk), pw, layout = (2, 1)))
savefig(figpath * "inf_allocs_tnorm.png")

# Additional plots 

# Rates of return 
p_hat(t, k) = (t * k) ^ (-1. / ϵ)

RoRs = [p0[j] * tgrid[i] * p_hat(tgrid[i], ksol_c[i, j]) 
    for i in 1:nt, j in 1:m_cheb] # p̄p̂θ

pR = plot(tgrid, RoRs,
    title = L"\bar{p}\hat{p}(\theta)\theta",
    xlab = L"\theta",
    legend = false);
display(pR)
savefig(figpath * "inf_rors_tnorm.png")

# Wedges and allocations with w≠0
wgrid = range(-10., 10., length = 201)
# pfix = 2 # fix p̄₀ = p0[pfix]
# pbar0 = p0[pfix]

# Calculating E[c(θ',p̄p̃)]

function cprime(tp, it, pj, csol, wsol)
    # Calculates c(θ',p̄p̃) for the pair (θ, p̄)
    # it is the index for θ
    # tp is the value of θ'
    # pj is the index for p̄

    # Get p̄′
    pbp = p0[pj] * p_tilde(wsol[it, pj])

    # Find indeces that bracket θ'
    itp = findlast(tgrid .< tp)
    itp1 = itp + 1
    tpl, tph = tgrid[[itp, itp1]]

    # Get the interpolation values
    csmooth_t = Fun(S0, ApproxFun.transform(S0, csol[itp, :]))
    csmooth_t1 = Fun(S0, ApproxFun.transform(S0, csol[itp1, :]))
    ctp_ptp = csmooth_t(pbp)
    ctp1_ptp = csmooth_t1(pbp)

    # Interpolate between them (linear)
    c_int = ctp_ptp + (ctp1_ptp - ctp_ptp) / (tph - tpl) * (tp - tpl)

    return c_int 

end

# cprime(1.002, 1, 1, csol_c, wsol_c)

function cpr_inv_f(tp, it, pj, csol, wsol)
    # Calculates (1. / cprime) * ft, for expectation

    Ft, ft, ftp = fdist(tp)
    return (1. / cprime(tp, it, pj, csol, wsol)) * ft
end

function wedges(i, j)
    # Given current values θ (index i) and p̄ (index j),
    # returns τ_k(θ, p̄) and τ_b(θ, p̄)
    # Note: both independent of w separate from p̄

    wp = wsol_c[i, j]
    cc = csol_c[i, j]
    th = tgrid[i]
    ecprime = gauss_leg(tp -> cpr_inv_f(tp, i, j, csol_c, wsol_c), 50, θ_min, θ_max)
    phat = p_hat(tgrid[i], ksol_c[i, j])
    
    τ_b = 1. - exp((1. - β) * wp) / (β * R * cc * ecprime)
    τ_k = 1. - exp((1. - β) * wp) / (β * th * p0[j] * phat * cc * ecprime)

    return [τ_b, τ_k]

end

# Savings wedge: depends only on θ
println("Calculating Wedges")
τb = [wedges(i, j)[1] for i in 1:nt, j in 1:m_cheb] 

# Investing wedge: depends on θ, pbar
τk = [wedges(i, j)[2] for i in 1:nt, j in 1:m_cheb]

p_τb = plot(tgrid, τb,
    title = L"\tau_b(\theta, \bar{p})",
    xlabel = L"\theta",
    legend = false);

# Plotting 
p_τk = plot(tgrid, τk,
    title = L"\tau_k(\theta, \bar{p})",
    xlabel = L"\theta",
    legend = false);
# p_τk = plot!(twinx(), tgrid, τk[:, 1],
#     xticks = :none,
#     label = L"\tau_k(\theta,\bar{p}_1)\;\;\textrm{ (Right\;axis)}",
#     legend = :left)
# plot!(right_margin = 12Plots.mm)

display(plot(p_τb, p_τk, layout = (1, 2) ) )
savefig(figpath * "inf_wedges_norm.png")

# Save k, w', RoR, wedges for comparison plots 
writedlm(figpath * "csol_c.txt", csol_c)
writedlm(figpath * "ksol_c.txt", ksol_c)
writedlm(figpath * "wsol_c.txt", wsol_c)
writedlm(figpath * "RoRs.txt", RoRs)
writedlm(figpath * "tk.txt", τk)
writedlm(figpath * "tb.txt", τb)

# plot(tgrid, [τb[:, 3] τk[:, 3]]) # same pattern as static in all cases

# Sample paths for wedges 
tdist = truncated(Normal(tmean, tsig), θ_min, θ_max)
capT = 100
thetas = rand(tdist, capT)
# plot(1:capT, thetas)

# pb_start = 0.5


function polfuns_smooth(th, pb)
    # For any θ and p̄, returns c, k, w' 
    # TODO: incorporate this into the wedges above 

    # Establish brackets for theta
    it = findlast(tgrid .< th)
    if isnothing(it)
        it = 1
    end
    it1 = it + 1
    tlo, thi = tgrid[[it, it1]]

    # Use Chebyshev approximations for policy functions 
    c_cheb_l = Fun(S0, ApproxFun.transform(S0, csol_c[it, :]))
    c_cheb_h = Fun(S0, ApproxFun.transform(S0, csol_c[it1, :]))
    c_l = c_cheb_l(pb)
    c_h = c_cheb_h(pb)

    k_cheb_l = Fun(S0, ApproxFun.transform(S0, ksol_c[it, :]))
    k_cheb_h = Fun(S0, ApproxFun.transform(S0, ksol_c[it1, :]))
    k_l = k_cheb_l(pb)
    if k_l < 0.
        k_l = k_cheb_l(0.)
    end
    k_h = k_cheb_h(pb)
    if k_h < 0.
        k_h = k_cheb_h(0.)
    end
    # If these dip below zero bc of added curvature, assume all the low guys
    # invest the same 

    w_cheb_l = Fun(S0, ApproxFun.transform(S0, wsol_c[it, :]))
    w_cheb_h = Fun(S0, ApproxFun.transform(S0, wsol_c[it1, :]))
    w_l = w_cheb_l(pb)
    w_h = w_cheb_h(pb)

    # Interpolate 
    cpol = c_l + (c_h - c_l) / (thi - tlo) * (th - tlo)
    kpol = k_l + (k_h - k_l) / (thi - tlo) * (th - tlo)
    wpol = w_l + (w_h - w_l) / (thi - tlo) * (th - tlo)

    return cpol, kpol, wpol

end

function wedges_smooth(th, pb)
    # Gets wedges for any (th, pb combination)

    c, k, wp = polfuns_smooth(th, pb)
    
    pbp = pb * p_tilde(wp)

    function cpr_inv_f_th(tp)
        cp, ~, ~ = polfuns_smooth(tp, pbp)
        ~, ft, ~ = fdist(tp)
        return (1. / cp) * ft
    end

    ecprime = gauss_leg(cpr_inv_f_th, 50, θ_min, θ_max)
    if k<0.
        println([th, pb])
        error("k<0")
    end
    phat = p_hat(th, k)

    τ_b = 1. - exp((1. - β) * wp) / (β * R * c * ecprime)
    τ_k = 1. - exp((1. - β) * wp) / (β * th * pb * phat * c * ecprime)

    return τ_b, τ_k

end

# polfuns_smooth(thetas[1], pb_start)
# wedges_smooth(thetas[1], pb_start)

# Works 
# tk_test = [wedges_smooth(tgrid[i], p0[j])[2] for i in 1:nt, j in 1:m_cheb]
# plot(tgrid, tk_test)

function sim_wedges(capT, pb_start)
    
    # Draw θₜ
    thetas = rand(tdist, capT)

    # Initialize 
    tkpath = zeros(capT)
    tbpath = similar(tkpath)
    pbpath = similar(tkpath)
    cpath = similar(tkpath)
    kpath = similar(tkpath)
    wpath = similar(tkpath)
    wp_path = similar(tkpath) # need to distinguish between accumulated w, and w'
    rorpath = similar(tkpath)

    pbpath[1] = pb_start
    wpath[1] = 0. # can start wherever 

    for i in 1:capT
        c, k, wp = polfuns_smooth(thetas[i], pbpath[i]) # allocations for (θ, p̄) pair 
        tkpath[i], tbpath[i] = wedges_smooth(thetas[i], pbpath[i])

        # Rate of Return 
        rorpath[i] = thetas[i] * pbpath[i] * p_hat(thetas[i], k)

        # Allocations for (θ, w, p̄) triple 
        cpath[i] = c * exp(shft * wpath[i])
        kpath[i] = k * exp(shft * wpath[i])
        wp_path[i] = wp

        if i < capT # update state 
            wpath[i + 1] = wpath[i] + wp 
            pbpath[i + 1] = pbpath[i] * p_tilde(wp)
        end
        
        if mod(i, 1_000) == 0
            println("t = $i")
        end

    end

    return tbpath, tkpath, pbpath, cpath, kpath, wp_path, wpath, rorpath

end

# capT = 10_000
# tb, tk, pb = sim_wedges(capT, 0.05)
# display(plot(1:capT, [tb tk])) 
# display(plot(1:capT, pb))

function simpaths(pb_starts, capT)
    tb_paths = zeros(capT, length(pb_starts))
    tk_paths = similar(tb_paths)
    pb_paths = similar(tb_paths)
    c_paths = similar(tb_paths)
    k_paths = similar(tb_paths)
    w_paths = similar(tb_paths)
    wp_paths = similar(tb_paths)
    ror_paths = similar(tb_paths)


    for i in 1:length(pb_starts)
        tb_paths[:, i], tk_paths[:, i], pb_paths[:, i], c_paths[:, i], k_paths[:, i],
            wp_paths[:, i], w_paths[:, i], ror_paths[:, i] = sim_wedges(capT, pb_starts[i])
        
    end

    return tb_paths, tk_paths, pb_paths, c_paths, k_paths, wp_paths, w_paths, ror_paths
end

# Sim with various pstart, see where it ends up
# Do same for wedges
pb_starts = range(0.05, pH, length = 8)
capT = 15_000

println("Simulating Paths")
@elapsed tb_paths, tk_paths, pb_paths, c_paths, k_paths, 
    wp_paths, w_paths, ror_paths = simpaths(pb_starts, capT)

# Plot p̄, wedges 
plot(pb_paths, 
    title = "State p̄",
    xlabel = L"t",
    ylabel = L"\bar{p}_t",
    legend = false);
pt_pbar = plot!(2000:capT, pb_paths[2000:end, :],
    inset = (1,  bbox(0.05, 0.15, 0.5, 0.35, :bottom, :right)),
    ylabel = L"\bar{p}_t",
    legend = false);

plot(tk_paths, 
    title = "Investing Wedge",
    xlabel = L"t",
    ylabel = L"\tau_k(\theta_t, \bar{p}_t)",
    legend = false);
pt_tauk = plot!(2000:capT, tk_paths[2000:end, :],
    inset = (1,  bbox(0.05, 0.15, 0.5, 0.35, :bottom, :right)),
    ylabel = L"\tau_k(\theta_t, \bar{p}_t)",
    legend = false);

plot(tb_paths, 
    title = "Savings Wedge",
    xlabel = L"t",
    ylabel = L"\tau_b(\theta_t, \bar{p}_t)",
    legend = false);
pt_taub = plot!(2000:capT, tb_paths[2000:end, :],
    inset = (1,  bbox(0.05, 0.15, 0.5, 0.35, :bottom, :right)),
    ylabel = L"\tau_b(\theta_t, \bar{p}_t)",
    legend = false);

display(plot(pt_pbar, pt_tauk, pt_taub, layout = (3, 1), size = (700, 800)))
savefig(figpath * "time_paths_wedges.png")

# It certainly looks like there is an ergodic set for p̄
# Does this hold with higher variance in returns? 
#  - Plot pre-tax rates of return, allocations 

# Plotting allocations and RoRs 

pc_pbar = plot(c_paths, 
    title = "Consumption",
    xlabel = L"t",
    ylabel = L"c_t(\theta^t)",
    legend = false);

pk_pbar = plot(k_paths, 
    title = "Investment",
    xlabel = L"t",
    ylabel = L"k_t(\theta^t)",
    legend = false);

pwp_pbar = plot(wp_paths, 
    title = "Promise Utility w′",
    xlabel = L"t",
    ylabel = L"w^\prime_t(\theta_t, \bar{p}_t)",
    legend = false);

pw_pbar = plot(w_paths, 
    title = "Promise Utility w (state)",
    xlabel = L"t",
    ylabel = L"w_t(\theta^{t-1})",
    legend = false); # replace with w'(θ, p̄)

plot(ror_paths, 
    title = "Rates of Return p̄p̂(θ)θ",
    xlabel = L"t",
    ylabel = L"\bar{p}_t \hat{p}(\theta_t)\theta_t",
    legend = false);
pror_pbar = plot!(2000:capT, ror_paths[2000:end, :],
    inset = (1,  bbox(0.05, 0.15, 0.5, 0.35, :bottom, :right)),
    ylabel = L"\bar{p}_t \hat{p}(\theta_t)\theta_t",
    legend = false);

display(
    plot(
        plot(pc_pbar, pk_pbar, pw_pbar, pwp_pbar, 
        layout = (2,2)),
    pror_pbar,
    layout = (2, 1),
    size = (1000, 1000)
    )
)
savefig(figpath * "time_paths_allocs.png")

# plot(p_tilde.(pb_paths)) 
psamp = vec(pb_paths[5000:end, :])
writedlm(solnpath * "psamp.txt", psamp)

using KernelDensity
kd = kde(psamp)
pspace = range(0.46, 0.48, length = 10_000)
plot(pspace, x -> pdf(kd, x))