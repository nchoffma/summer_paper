#= 

Solves the planner's infinite-horixon CMP, 
with θ∼N(μ,σ) and households as monopolists

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, NLsolve, 
    ForwardDiff, Optim, LaTeXStrings, Roots, 
    FastGaussQuadrature, Printf, DelimitedFiles, ApproxFun

gr()
println("******** cpp_inf_mpl.jl ********")

# Parameters
β = 0.9         # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

R = 1.1

κ = 1.
# println("\nκ = $κ\n")

#= κ notes 

Seems to work with low values of κ, down to 0.5 (so far) and up to 1.

For κ = 1.01, works. For κ = 1.02, run into boundary issues. These are easily
fixed, we just need to be consistent. 

=#

solnpath = "julia/complementarities/monopolists/results/"
figpath = solnpath * "figures/"

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

function full_info_soln(wL, wH, wp0, cpts0,
    print_every = 50)

    println("\nSolving full-info problem for w∈[$wL, $wH], κ = $κ\n")

    # Build the space and approximation
    C0 = Fun(S0, ApproxFun.transform(S0, cpts0))

    # Calculate constant terms in C 
    tterm = gauss_leg(x -> (x ^ (ϵ - 1.)) * tnorm_pdf(x), 50, θ_min, θ_max)
    cons1 = (κ * (ϵ - 1.) / (ϵ * R)) ^ ϵ * tterm 
    cons2 = (κ / R) * (κ * (ϵ - 1.) / (ϵ * R)) ^ (ϵ - 1.) * tterm 

    function get_wp(w, Cf; w0 = w)

        function fr(wp)
            if wp > wH
                Cp_wp = Cf'(wH)
            elseif wp < wL
                Cp_wp = Cf'(wL)
            else
                Cp_wp = Cf'(wp)
            end
    
            return Cp_wp - R * β * exp(w - β * wp)
        end
    
        bkt0 = (floor(w0 - 0.01, digits = 1), ceil(w0 + 0.01, digits = 1))
        stp = round(bkt0[2] - bkt0[1], digits = 5)
        bkt = find_bracket(fr, bkt0 = bkt0, step = stp) 
        wpstar = find_zero(fr, bkt,
            atol = 1e-6, rtol = 1e-6,
            xatol = 1e-6, xrtol = 1e-6) 
    
        return wpstar 

        
    end

    function iterate_c_full(cpts0;
        ω = 1.)
    
        diff = 10
        tol = 1e-6
        its = 0
        mxit = 5000
        cpts1 = copy(cpts0)
        wpvals = copy(wp0)
        cwp = zeros(m_cheb)  
    
        while diff > tol && its < mxit
            C0 = Fun(S0, ApproxFun.transform(S0, cpts0))
            
            for i in 1:m_cheb
    
                # Get w' values 
                get_wp_curr(w) = get_wp(w, C0, w0 = wpvals[i])
                wpvals[i] = get_wp_curr(wp0[i])
    
                # Get C(w') values, interpolating where needed 
                wp = wpvals[i]
                if wp > wH
                    cwp[i] = C0(wH) + C0'(wH) * (wp - wH)
                elseif wp < wL 
                    cwp[i] = C0(wL) + C0'(wL) * (wp - wL)
                else
                    cwp[i] = C0(wp)
                end
    
            end
    
            cpts1 = exp.(wp0 - β * wpvals) + cwp / R .+ cons1 .- cons2
    
            diff = norm(cpts0 - cpts1, Inf)
            rd = round(diff, digits = 5)
            rdwp = round.(wpvals, digits = 2)
            if mod(its, print_every) == 0
                println("iteration $its: diff = $rd, w' = $rdwp")
            end
            
            its += 1
            cpts0 = ω * cpts1 + (1. - ω) * cpts0
        end
        return cpts1, wpvals
    end
    
    cpts, wps = iterate_c_full(cpts0)
    gamcs = exp.(wp0 - β * wps)         # these are the c values, which equal γ

    # Plot and Return
    wgrid = range(wL, wH, length = 100)
    Cfull = Fun(S0, ApproxFun.transform(S0, cpts))

    pC = plot(wgrid, Cfull.(wgrid),
        title = "Full Info Solution",
        label = "C(w)",
        legend = :bottomright,
        xlabel = "w")

    pwp = plot(wp0, [wp0 wps],
        title = "Full-info w'",
        label = ["w'=w" "w'"],
        legend = :bottomright,
        xlabel = "w")

    display(plot(pC, pwp, layout = (1, 2)))

    return cpts, gamcs, wps

end

function eval_pkc(γ, C0, ubkt; 
    return_all = false,
    return_fxns = false)
    
    # Build the necessary functions of C0, γ -------------------------------------

    function foc_k(x, U, μ, θ)
        
        c, wp, k = x
    
        if wp > wH
            Cp_wp = C0'(wH)
        elseif wp < wL
            Cp_wp = C0'(wL)
        else
            Cp_wp = C0'(wp)
        end
    
        # FOC vector
        focs = zeros(eltype(x), 3)
        focs[1] = 1. - Cp_wp / (β * R * c) - μ * k / (θ * c ^ 2)
        focs[2] = 1. - κ * (ϵ - 1.) / (R * ϵ) * θ ^ (1. - 1. / ϵ) * k ^ (-1. / ϵ) + 
            μ / (θ * c)
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
        ~, ft, fpt = fdist(t)
    
        if wp > wH
            Cp_wp = C0'(wH)
        elseif wp < wL
            Cp_wp = C0'(wL)
        else
            Cp_wp = C0'(wp)
        end
    
        du[1] = k / (c0 * t)
        du[2] = γ - Cp_wp / (β * R) - fpt / ft * μ
    
    end

    function tax_shoot(U_0)
        u0 = [U_0 0.0]
        tspan = (θ_min, θ_max)
        prob = ODEProblem(de_system!, u0, tspan)
        gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
        return gpath.u[end][2] # want to get μ(θ̲) = 0
        
    end

    # Promise-keeping
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

    # Value of C1
    function C_integrand(t, gpath)
        
        U, mu = gpath(t)
        Ft, ft, fpt = fdist(t)
        ct, wpt, kt = alloc_single(U, mu, t)

        tr = round(t, digits = 2)
        wpr = round(wpt, digits = 2)

        # Extrapolation (if necessary)
        if wpt > wH
            println("Warning: w' > wH in C_integrand (θ = $tr, w' = $wpr)")
            C_wp = C0(wH) + C0'(wH) * (wpt - wH)
        elseif wpt < wL
            println("Warning: w' < wL in C_integrand (θ = $tr, w' = $wpr)")
            C_wp = C0(wL) + C0'(wL) * (wpt - wL)
        else
            C_wp = C0(wpt)
        end
        
        return ft * (ct + kt + 1 / R * (C_wp - κ * (t * kt) ^ (1. - 1. / ϵ)) )

    end

    # Solve the system and evaluate promise-keeping ------------------------------

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
        Cupd = gauss_leg(t -> C_integrand(t, gpath), 50, θ_min, θ_max)
        return gpath, csol, wsol, ksol, Cupd, musol, bkt
    elseif return_fxns

        # Return functions built along the way (used for PFI step)
        # println("got to fxns")
        return gpath, foc_k, newton_k, alloc_single, bkt 
    else

        # Just return the pkc value (used for zeroing)
        return pkc
    end

end

function update_C(C0, gam_bkts, ubkts)
    # Updates C once 
    # 
    # Note: unlike the price-taking case, we need to supply a different U0 
    # bracket for each w*, because they are substantially different 

    cpts1 = zeros(m_cheb)
    gams = similar(cpts1)
    csol = zeros(nt, m_cheb)
    ksol = similar(csol)
    wsol = similar(csol)
    musol = similar(csol)

    for i in 1:m_cheb
        
        # Get the current brackets, w* val
        ubkt = ubkts[i]
        gam_bkt = gam_bkts[i]
        wstar = wp0[i]

        # Build the current w* function 
        fwstar(gm) = eval_pkc(gm, C0, ubkt) - wstar

        # And zero it out 
        stp = gam_bkt[2] - gam_bkt[1]
        bkt_int = find_bracket(fwstar, bkt0 = gam_bkt, step = stp)
        gstar = find_zero(fwstar, bkt_int, 
            atol = 1e-6, rtol = 1e-6,
            xatol = 1e-6, xrtol = 1e-6,
            Roots.A42())

        # Fill in the rest of the solution 
        gams[i] = gstar
        gpath, csol[:, i], wsol[:, i], ksol[:, i], Cupd, musol[:, i], ubkt_p = eval_pkc(gstar, 
            C0, ubkt, return_all = true)
        gprint = round(gstar, digits = 4)
        ubprint = round.(ubkt_p, digits = 4)
        println("w value $i: γ* = $gprint, Ubkt = $ubprint")
        cpts1[i] = Cupd

    end
    return gams, csol, wsol, ksol, cpts1, musol

end

function get_bkts(xs; digs = 3)
    # Gets the initial brackets based on the values

    xbs = [floor.(xs, digits = digs) ceil.(xs, digits = digs)]
    xbks = Tuple(Tuple(xbs[i,:]) for i in 1:length(xs))

    return xbks
end

function pfi_step(gstars, c1, c0f, ubkts; npol = 20)
    # Uses modified PFI (HIA) to get updated C(⋅) function 
    # Default is 20 steps 
    # Note: c0f is the prior starting function, which is needed to recover the functions
    # c1 is the updated values at the nodes 

    chat0 = copy(c1)
    chat1 = similar(chat0)

    for hk in 1:npol
        chat1 = zeros(m_cheb)
        Chat = Fun(S0, ApproxFun.transform(S0, chat0))
        for j in 1:m_cheb
            wstar = wp0[j]
            ubkt = ubkts[j]
            γ = gstars[j]
            gpath, foc_k, newton_k, alloc_single, bkt_U0 = 
                eval_pkc(γ, c0f, ubkt, return_fxns = true)

            function C_integrand_HIA(t, gpath)
                # The same as the C_integrand() function in eval_pkc(), but 
                # uses the interim function Chat 

                U, mu = gpath(t)
                Ft, ft, fpt = fdist(t)
                ct, wpt, kt = alloc_single(U, mu, t)

                # Extrapolation (if necessary)
                if wpt > wH
                    C_wp = Chat(wH) + Chat'(wH) * (wpt - wH)
                elseif wpt < wL
                    C_wp = Chat(wL) + Chat'(wL) * (wpt - wL)
                else
                    C_wp = Chat(wpt)
                end
                
                return ft * (ct + kt + 1 / R * (C_wp - κ * (t * kt) ^ (1. - 1. / ϵ)) )
            end

            chat1[j] = gauss_leg(t -> C_integrand_HIA(t, gpath), 50, θ_min, θ_max)
            ubkt = bkt_U0
        end
        chat0 = copy(chat1)

    end

    return chat0

end

function iterate_C(cpts0, gbkts0, ubkts0; 
    ω = 1.,
    npfi = 0,
    mxit = 1_000,
    plot_progress = true)

    # Allocate solutions 
    cpts1 = similar(cpts0)
    gstars = zeros(m_cheb)
    csol = zeros(nt, m_cheb)
    wsol = similar(csol)
    ksol = similar(csol)

    # Iteration parameters
    norm_C = 10.
    tol = 1e-6
    its = 0

    # Visualizing progress
    if plot_progress
        wspace = range(wL, wH, length = 100)
        Cstart = Fun(S0, ApproxFun.transform(S0, cpts0))
        display(plot(wspace, Cstart.(wspace), legend = false))
    end

    while norm_C > tol && its < mxit
        println("\nIteration $its\n")
        
        # Update C
        Cfunc = Fun(S0, ApproxFun.transform(S0, cpts0))
        if plot_progress
            display(plot!(wspace, Cfunc.(wspace), legend = false))
        end
        gstars, csol, wsol, ksol, cpts_u, musol = update_C(Cfunc, gbkts0, ubkts0)

        # Find the updated brackets
        Usol = log.(csol) + β * wsol 
        gbkts0 = get_bkts(gstars .- gcush, digs = 3) 
        ubkts0 = get_bkts(Usol[1, :] .- ucush, digs = 2) 

        # Go to HIA, if desired 
        if npfi > 0
            println("Going to HIA")
            cpts_u = pfi_step(gstars, cpts_u, C0, ubkts0, npol = npfi)
            println("HIA complete")
        end

        # Update and check for convergence 
        norm_C = norm(cpts0 - cpts_u, Inf)
        cpts1 = ω * cpts_u + (1. - ω) * cpts0
        println("norm_C = $norm_C")
        println(round.(cpts1, digits = 3))

        cpts0 = copy(cpts1)
        its += 1

    end

    return  gstars, csol, wsol, ksol, cpts1

end

ucush = 0.
gcush = 0.

#= State space notes

Notes: 
- The full-info solution is a VERY good starting guess, for both the C 
  coefficients and the γ values 
- If the bounds are not violated on the first update, it's unlikely they 
  will ultimately be

w∈[-7., 1.] - substantial extrapolation at lower bound
w∈[-7., 4.] - substantial extrapolation at lower bound
w∈[-7., 5.] - moving UB seems to help
⋮
w∈[-7., 10.] - WORKS with β = 0.9, R = 1.1
w∈[-8., 10.] - WORKS with β = 0.9, R = 1.1

w∈[-12., 10] # w settles down 

=#

# State space w 
wL = -18.
wH = 24.

m_cheb = 7
S0 = Chebyshev(wL..wH)
wp0 = points(S0, m_cheb)

# Step one: Solve full info allocations
cpts0 = (wp0 .- wp0[end] .+ 0.5) .^ 1.5 

cpts_full, gamcs_full, wps_full = full_info_soln(wL, wH, wp0, cpts0);
C0 = Fun(S0, ApproxFun.transform(S0, cpts_full))

println("\n -------- Going to full Solution -------- \n")

gbkts1 = get_bkts(gamcs_full, digs = 3)
ubkts1 = get_bkts(wp0 .- 0.401, digs = 2)

# Update once 
gstars, csol, wsol, ksol, cpts_u, musol = update_C(C0, gbkts1, ubkts1);

# Try some iteration
gstars, csol, wsol, ksol, cpts1 = iterate_C(cpts_full, 
    gbkts1, ubkts1, plot_progress = false, ω = 0.5)

Usol = log.(csol) + β * wsol 

# Check global incentive constraints
ics_full = zeros(nt, nt, m_cheb)
for i in 1:nt, j in 1:nt
    for k in 1:m_cheb
        dev_c = csol[j, k] + ksol[j, k] - tgrid[j] / tgrid[i] * ksol[j, k]
        ics_full[i, j, k] = log(max(1e-15, dev_c)) + β * wsol[j, k] - Usol[i, k]
    end
end

# maximum(ics_full)
numviols = sum(ics_full .> 1e-14) # global ics satisfied! 
println("\n$numviols Global incentive constraints violated\n")

soln_suffix = "_" * string(κ) * "_" * string(wL) * "_" * string(wH) * ".txt"
fig_suffix = "_" * string(κ) * "_" * string(wL) * "_" * string(wH) * ".png"
writedlm(solnpath * "cpts" * soln_suffix, cpts1)
writedlm(solnpath * "gams" * soln_suffix, gstars)
writedlm(solnpath * "U0s" * soln_suffix, Usol[1, :])

# Full-info k', for comparison
k_fullinfo = (κ * (ϵ - 1.) / (ϵ * R)) ^ ϵ * tgrid .^ (ϵ - 1.)

# Plot the solution 
pU = plot(tgrid, Usol,
    xlab = L"\theta",
    title = "Utilities",
    ylab = L"U(\theta;w)",
    legend = false);

Cc = Fun(S0, ApproxFun.transform(S0, cpts1))
wgrid = range(wL, wH, length = 100)
pC = plot(wgrid, Cc.(wgrid),
    title = L"C(w)",
    xlab = L"w",
    legend = false);

pc = plot(tgrid, csol, 
    title = L"c(\theta,w)",
    xlab = L"\theta",
    legend = false);

plot(tgrid, ksol, 
    title = L"k^\prime(\theta,w)",
    xlab = L"\theta",
    legend = false);
pk = plot!(tgrid, k_fullinfo, linestyle = :dash)

pw = plot(tgrid, wsol, 
    title = L"w^\prime(\theta,w)",
    xlab = L"\theta",
    legend = false);

# Rate of return (θk)^(1 - 1/ϵ)
rors = κ * (repeat(tgrid, 1, m_cheb) .* ksol) .^ (1. - 1. / ϵ)
pror = plot(tgrid, rors, 
    title = "Rate of Return θ p(θ)",
    xlab = L"\theta",
    ylab = L"[\theta k(\theta)]^{1 - 1/\epsilon}",
    legend = false);

display(plot(pU, pC, pc, pk, pw, pror, layout = (3,2),
    size = (1000, 1000)))
savefig(figpath * "soln" * fig_suffix)

# Wedges 
function polfuns_smooth(th, w)
    # For any θ and w, returns c, k, w' 

    # Establish brackets for theta
    it = findlast(tgrid .< th)
    if isnothing(it)
        it = 1
    end
    it1 = it + 1
    tlo, thi = tgrid[[it, it1]]

    # Use Chebyshev approximations for policy functions 
    c_cheb_l = Fun(S0, ApproxFun.transform(S0, csol[it, :]))
    c_cheb_h = Fun(S0, ApproxFun.transform(S0, csol[it1, :]))
    c_l = c_cheb_l(w)
    c_h = c_cheb_h(w)

    k_cheb_l = Fun(S0, ApproxFun.transform(S0, ksol[it, :]))
    k_cheb_h = Fun(S0, ApproxFun.transform(S0, ksol[it1, :]))
    k_l = k_cheb_l(w)
    if k_l < 0.
        k_l = k_cheb_l(0.)
    end
    k_h = k_cheb_h(w)
    if k_h < 0.
        k_h = k_cheb_h(0.)
    end
    # If these dip below zero bc of added curvature, assume all the low guys
    # invest the same (shouldn't be an issue here)

    w_cheb_l = Fun(S0, ApproxFun.transform(S0, wsol[it, :]))
    w_cheb_h = Fun(S0, ApproxFun.transform(S0, wsol[it1, :]))
    w_l = w_cheb_l(w)
    w_h = w_cheb_h(w)

    # Interpolate 
    cpol = c_l + (c_h - c_l) / (thi - tlo) * (th - tlo)
    kpol = k_l + (k_h - k_l) / (thi - tlo) * (th - tlo)
    wpol = w_l + (w_h - w_l) / (thi - tlo) * (th - tlo)

    return cpol, kpol, wpol

end

# c, k, w = polfuns_smooth(1.23, 4.235) # seems to work 

# E[c(θ',w'(θ,w))^-1]
function Ecprime(th, w)
    # Given θ, calculates E[c(θ',w'(θ,w))]
    ~, ~, wp = polfuns_smooth(th, w)   # w'(θ, w) next-period state

    function cprime(tp, wp)
        cp, ~, ~ = polfuns_smooth(tp, wp)
        return cp
    end

    ecp = gauss_leg(t -> (1. / cprime(t, wp)), 50, θ_min, θ_max)

end

function wedges(th, w)
    # Calculate the wedges for any (θ, w) pair

    ecp = Ecprime(th, w)
    cc, kc, wp = polfuns_smooth(th, w)
    tauk = 1. - (ϵ / (ϵ - 1.)) / (β * cc * κ * (th * kc) ^ (1. - 1. / ϵ) * ecp)
    taub = 1. - 1. / (β * R * cc * ecp)

    return tauk, taub

end

println("Calculating Wedges")
tk = zeros(nt, m_cheb)
tb = similar(tk)
for i in 1:nt, j in 1:m_cheb
    tk[i, j], tb[i, j] = wedges(tgrid[i], wp0[j])
end

# Plotting 
p_τb = plot(tgrid, tb,
    title = L"\tau_b(\theta, w)",
    xlabel = L"\theta",
    legend = false);

p_τk = plot(tgrid, tk,
    title = L"\tau_k(\theta, w)",
    xlabel = L"\theta",
    legend = false);

display(plot(p_τb, p_τk, layout = (1, 2) ) )
savefig(figpath * "wedges" * fig_suffix)

# Save things for comparison plots 
writedlm(solnpath * "csol" * soln_suffix, csol)
writedlm(solnpath * "wsol" * soln_suffix, wsol)
writedlm(solnpath * "ksol" * soln_suffix, ksol)
writedlm(solnpath * "RoRs" * soln_suffix, rors)
writedlm(solnpath * "tk" * soln_suffix, tk)
writedlm(solnpath * "tb" * soln_suffix, tb)

# Simulating paths 
tdist = truncated(Normal(tmean, tsig), θ_min, θ_max)

function sim_wedges(capT, wstart)
    # Simulates a path for wedges, allocations, etc. 
    
    # Draw θₜ
    thetas = rand(tdist, capT)

    # Initialize 
    tkpath = zeros(capT)
    tbpath = similar(tkpath)
    cpath = similar(tkpath)
    kpath = similar(tkpath)
    rorpath = similar(tkpath)
    wpath = zeros(capT + 1)

    wpath[1] = wstart
    for i in 1:capT
        cpath[i], kpath[i], wpath[i + 1] = polfuns_smooth(thetas[i], wpath[i])
        tkpath[i], tbpath[i] = wedges(thetas[i], wpath[i])

        # Rate of return κ * [θ k] ^ {1 - 1/ϵ}
        rorpath[i] = κ * (thetas[i] * kpath[i]) ^ (1. - 1. / ϵ)

        if mod(i, 1_000) == 0
            println("t = $i")
        end

    end

    return tbpath, tkpath, cpath, kpath, wpath, rorpath

end

capT = 1_000
@time tbpath, tkpath, cpath, kpath, wpath, rorpath = sim_wedges(capT, 0.5 * (wL + wH));
plot(1:capT + 1, wpath) # some powerful immiseration here 

# Simulate paths for a number of different wstart values 
function simpaths(w_starts, capT)
    tb_paths = zeros(capT, length(w_starts))
    tk_paths = similar(tb_paths)
    c_paths = similar(tb_paths)
    k_paths = similar(tb_paths)
    ror_paths = similar(tb_paths)
    w_paths = zeros(capT + 1, length(wstarts))

    for i in 1:length(w_starts)
        tb_paths[:, i], tk_paths[:, i], c_paths[:, i], k_paths[:, i],
            w_paths[:, i], ror_paths[:, i] = sim_wedges(capT, w_starts[i])
        
    end

    return tb_paths, tk_paths, c_paths, k_paths, w_paths, ror_paths
end

println("Simulating Paths")
wstarts = range(wL, wH, length = 5)
capT = 5_000
@elapsed tb_paths, tk_paths, c_paths, k_paths, w_paths, 
    ror_paths = simpaths(wstarts, capT)

# Plot one: w, τ_k, τ_b 
pt_wp = plot(w_paths, 
    title = "State w",
    xlabel = "t",
    legend = :false);

pt_tauk = plot(tk_paths, 
    title = L"\tau_k(\theta_t, w_t)",
    xlabel = "t",
    legend = :false);

pt_taub = plot(tb_paths, 
    title = L"\tau_b(\theta_t, w_t)",
    xlabel = "t",
    legend = :false);

display(plot(pt_wp, pt_tauk, pt_taub, layout = (3, 1), size = (700, 800)))
savefig(figpath * "time_paths_wedges" * fig_suffix)

# Plot two: c, k, RoR 
pt_c = plot(c_paths, 
    title = L"c(\theta_t, w_t)",
    xlabel = "t",
    legend = :false);

pt_k = plot(k_paths, 
    title = L"k'(\theta_t, w_t)",
    xlabel = "t",
    legend = :false);

pt_ror = plot(ror_paths, 
    title = "Rate of Return θkp",
    xlabel = "t",
    legend = :false);

display(plot(pt_c, pt_k, pt_ror, layout = (3, 1), size = (700, 800)))
savefig(figpath * "time_path_allocs" * fig_suffix)