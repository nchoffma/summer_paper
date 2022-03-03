using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles, ApproxFun

gr()
println("******** cpp_inf_horizon.jl ********")

# Parameters
β = 0.8         # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

R = 1.5

# Analitically, c(θ,0) = ctest, and A(0) = ctest / (1. - β)
ctest = (1. / (β * R)) ^ (1. / (1. - β))
Atest =  ctest /  (1. - β)

# Path to write results 
solnpath = "julia/complementarities/results/"

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
# pH = 0.8  # yes 
pH = 0.9  # does not work here

m_cheb = 5
S0 = Chebyshev(pL..pH)
p0 = points(S0, m_cheb)

# Functions for truncated normal dist 
tmean = (θ_max + θ_min) / 2
tsig = 0.2
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

    # Uniform
    L = θ_min
    H = θ_max 
    cdf = (x - L) / (H - L)
    pdf = 1. / (H - L)
    fpx = 0.

    # # Truncated normal
    # cdf = tnorm_cdf(x)
    # pdf = tnorm_pdf(x)
    # fpx = tnorm_fprime(x)
    
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
        println("pbar value $i: γ* = $gstar, Ubkt = $ubkt_p")
        # println(ubkt_p)
        at1[i] = aupd

        if update_ubkt
            ubkt = ubkt_p .- 0.5 # give some cushion
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
    pfi = true,
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
    mxit = 100

    while norm_A > tol && its < mxit
        println("Iteration $its")

        # Update 
        Afunc = Fun(S0, ApproxFun.transform(S0, a0))
        gstars, csol, wsol, ksol, a1, ~ = update_at(Afunc, gam_bkt0, Ubkt0,
            extp_t = true, update_ubkt = updateU)

        # Find the new brackets for U0, γ 
        Usol = log.(csol) + β * wsol 
        Ubkt0 = (floor(Usol[1,1], digits = 2), 
                ceil.(Usol[1,1], digits = 2)) .- 0.5 # need to ensure we're approaching root from below

        gbs_upd = [floor.(gstars, digits = 3) ceil.(gstars, digits = 3)]

        # if its > 1
        #     gams_to_update = "all"
        # else
        #     gams_to_update = "first"
        # end

        if gams_to_update == "first"
            gam_bkt0 = (Tuple(gbs_upd[1,:]), )
        elseif gams_to_update == "all"
            gam_bkt0 = Tuple(Tuple(gbs_upd[i,:]) for i in 1:m_cheb)
        else
            error("gams_to_update must be one of 'first' or 'all'. ")
        end

        # PFI, if desired
        if pfi
            println("Going to PFI")
            Ubkt_pfi = (floor(Usol[1,1], digits = 2), 
                ceil.(Usol[1,1], digits = 2))
            a1 = pfi_step(gstars, a1, Afunc, Ubkt_pfi, npol = npfi, extrap = true)
        end

        # Evaluate convergence
        norm_A = norm(a1 - a0, Inf) 
        println("norm_A = $norm_A")

        a0 = copy(a1)
        its += 1

    end

    return gstars, csol, wsol, ksol, a1

end

# Ubkt_start = (-0.23, -0.22) # works up to pH = 0.8
Ubkt_start = (-0.63, -0.62) 
at0 = ones(m_cheb) * Atest
@time gstars_c, csol_c, wsol_c, ksol_c, a1_c = iterate_at(at0, ((0.416, 0.417), ), 
    Ubkt_start, gams_to_update = "all", npfi = 5);
Usol_c = log.(csol_c) + β * wsol_c
Ac = Fun(S0, ApproxFun.transform(S0, a1_c))

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

figpath = "julia/complementarities/results/"
display(plot(plot(ppb, pat1), pU, layout = (2,1)))
savefig(figpath * "inf_soln.png")

# display(plot(plot(pc, pk), pw, layout = (2, 1)))
# savefig(figpath * "inf_allocs.png")
