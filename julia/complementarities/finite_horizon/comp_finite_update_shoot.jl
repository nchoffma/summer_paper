#= 

Solves the problem at T-1, with knowledge that 
A_T(p) = 1 for *all* p∈R

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles, ApproxFun

gr()
println("******** comp_finite_vf_update_shoot.jl ********")

# Parameters
const β = 0.95        # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

R = 1.6

# Finite horizon
capT = 2
shft = (1. - β) / (1. - β ^ (capT + 1))             # shifter for finite horizon, t = 0,1,...,T on allocations
shft_w = (1. - β ^ capT) / (1. - β ^ (capT + 1) )   # shifter on promise utility

p_tilde(x) = exp(-1. / ϵ * shft * shft_w * x)

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

# # A_T 
# A0(x) = x ^ 0. # avoids nothing() derivative

# State space for p̄
pL = 0.     # absorbing barrier 
pH = 1.2

m_cheb = 8
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
    mxit = 1000

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

function eval_pkc(γ, pbar, A0; return_all = false, extrap = false)

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
        tol = 1e-6
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
        for i in 1:nt
            U, mu = gpath(tgrid[i])
            cp[i], wp[i], kp[i] = alloc_single(U, mu, tgrid[i])

        end
        return cp, wp, kp
    end

    # Value of A_t-1
    function at_integrand(t, gpath)
        U, mu = gpath(t)
        Ft, ft, fpt = fdist(t)
        ct, wpt, kt = alloc_single(U, mu, t)
        phat = (t * kt) ^ (-1. / ϵ)
        pnext = pbar * phat
        ptilde = p_tilde(wpt)
        atp1 = A0(pbar * ptilde)
        return ft * (ct + kt + 1 / R * ( atp1 * exp(shft * wpt) - t * pnext * kt ) )
    end

    if pbar < 1. # trial and error to determine this
        bkts = (-0.5, -0.4)
    elseif pbar > 1. && pbar < 1.5
        bkts = (-2.5, -2.4)
    else
        bkts = (-4.5, -4.4)
    end
    # println("bkts = $bkts")
    bkt = find_bracket(um -> tax_shoot(um), bkt0 = bkts)
    Umin_opt = find_zero(x -> tax_shoot(x), bkt)

    u0 = [Umin_opt 0.0]
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan)
    gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 

    pkc = gauss_leg(t -> pkc_integrand(t, gpath), 50, θ_min, θ_max)

    if return_all
        csol, wsol, ksol = opt_allocs(gpath)
        aupd = gauss_leg(t -> at_integrand(t, gpath), 50, θ_min, θ_max)
        return gpath, csol, wsol, ksol, aupd
    else
        return pkc
    end

end

# ip = 1
# γ = 0.052

# pkc = eval_pkc(γ, p0[ip])

function update_at(At, gam_brackets;
    extp_t = false)

    # If length of gam_brackets is one, this is the initial bracket
    # If length of gam_brackets is m_cheb, it gives the bracket for each pbar
    # TODO: if length is in between, use the supplied brackets, then update iteratively

    at1 = zeros(m_cheb)
    gams = similar(at1)
    csol = zeros(nt, m_cheb)
    ksol = similar(csol)
    wsol = similar(csol)

    if length(gam_brackets) == 1
        gam_bkt = gam_brackets[1]
    end
    
    for i in 1:m_cheb

        # Build current pbar function
        fpb(gm) = eval_pkc(gm, p0[i], At, extrap = extp_t)

        # Find optimal γ and get allocations
        if length(gam_brackets) > 1
            gam_bkt = gam_brackets[i]
        end

        println("ip = $i, γ bracket = $gam_bkt")
        bkt_int = find_bracket(fpb, bkt0 = gam_bkt, step = 0.001)
        gstar = find_zero(fpb, bkt_int)
        println("pbar value $i: γ* = $gstar")
        gams[i] = gstar
        gpath, csol[:, i], wsol[:, i], ksol[:, i], aupd = eval_pkc(gstar, p0[i], At, return_all = true, extrap = extp_t)
        at1[i] = aupd
        
        if length(gam_brackets) == 1
            gam_bkt = bkt_int
        end
        
    end
    return gams, csol, wsol, ksol, at1
end

# Aₜ
AT(x) = x ^ 0. # avoids nothing() derivative
@time gstars, csol, wsol, ksol, at1 = update_at(AT, ((0.339, 0.34), ) );

open("julia/complementarities/finite_horizon/results/a_t1.txt", "w") do io
    writedlm(io, at1)
end # write Chebyshev coefficients to file 

function pbar_next(wpol)
    pbarp = zeros(nt, m_cheb)
    for i in 1:nt, j in 1:m_cheb
        ptilde = p_tilde(wpol[i, j])
        pbarp[i, j] = ptilde * p0[j]
    end
    return pbarp
end

pbar_prime = pbar_next(wsol);
Usol = log.(csol) + β * wsol
# plot(tgrid, pbar_prime)
# plot(tgrid, csol)

# plot(p0, at1)

# Plotting
# palloc = plot(tgrid, [csol[:, 3] ksol[:, 3] wsol[:, 3]], 
#     title = "Allocations",
#     label = [L"c(\theta; \bar{p}_3)" L"k(\theta; \bar{p}_3)" L"w^\prime(\theta; \bar{p}_3)"],
#     xlab = L"\theta",
#     legend = :topleft)

pU = plot(tgrid, Usol,
    xlab = L"\theta",
    title = "Utilities",
    ylab = L"U_{T-1}(\theta;\bar{p})",
    legend = false);

ppb = plot(tgrid, pbar_prime,
    xlab = L"\theta",
    title = "Next-period state",
    ylab = L"\bar{p}_T (\theta;\bar{p})",
    legend = false);

pat1 = plot(p0, at1,
    title = L"A_{T-1}(\bar{p})",
    xlab = L"\bar{p}",
    legend = false);

savefig("julia/complementarities/finite_horizon/results/trial_soln_unif.png")

# Inspecting allocations 
pc = plot(tgrid, csol, 
    title = L"c_{T-1}(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false);

pk = plot(tgrid, ksol, 
    title = L"k_{T-1}(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false);

pw = plot(tgrid, wsol, 
    title = L"w^\prime_{T-1}(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false);

display(plot(plot(pc, pk), pw, layout = (2, 1)))
savefig("julia/complementarities/finite_horizon/results/trial_allocns_unif.png")

# Checking global incentive constraints
# There will be nt² of these for each value of pbar 
function check_global_ics()
    num_viols = 0
    for ip in 1:m_cheb
        for i in 1:nt, j in 1:nt
            devc = csol[j,ip] + ksol[j,ip] - tgrid[j] * ksol[j,ip] / tgrid[i]
            ic = log(max(devc, 1e-15)) + β * wsol[j, ip] - Usol[i, ip]
            
            if ic > 0 && i != j # i == j case is just numerical precision
                println([ip, i, j])
                println(ic)
                num_viols += 1
            end
        end
    end
    return num_viols
end

nvs = check_global_ics(); # all global ics hold 
if nvs > 0 println("nvs = $nvs") end

println([minimum(pbar_prime) maximum(pbar_prime)])

# Rates of return 
rors = [p0[j] * tgrid[i] * (ksol[i,j] * tgrid[i]) ^ (-1. / ϵ) for i in 1:nt, j in 1:m_cheb]

plot(tgrid, rors, 
    title = L"Rate of Return $\theta\bar{p}\hat{p}$ (R = 1.5)",
    legend = false,
    x = L"\theta") 
savefig("julia/complementarities/finite_horizon/results/trial_rors_unif.png")

# Euler equation (savings) residuals
lhs = 1. ./ csol 
cprime = exp.(wsol)
rhs = β * R * 1. ./ cprime

resids = lhs - rhs 

plot(tgrid, resids, 
    title = "Euler Equation Residuals",
    label = false,
    xlab = L"\theta")
savefig("julia/complementarities/finite_horizon/results/trial_resids_unif.png")

# Update again 
AT1 = Fun(S0, ApproxFun.transform(S0, at1))

# ip = 1
# fpb(gm) = eval_pkc(gm, p0[ip], AT1, extrap = true)
# gam_bkt = (0.226, 0.227)
# bkt_int = find_bracket(fpb, bkt0 = gam_bkt, step = 0.001)
# gstar = find_zero(fpb, bkt_int) 
# gpath = eval_pkc(gstar, p0[ip], AT1, return_all = true, extrap = true)

γ_bkts = (
    (0.226, 0.227),
    (0.274, 0.275),
    (0.337, 0.338),
    (0.383, 0.384),
    (0.402, 0.403),
    (0.407, 0.408),
    (0.407, 0.408),
    (0.407, 0.408)
) 

@time gstars1, csol1, wsol1, ksol1, at2 = update_at(AT1, γ_bkts, extp_t = true);

pbar_prime1 = pbar_next(wsol1);
Usol1 = log.(csol1) + β * wsol1;

pU = plot(tgrid, Usol1,
    xlab = L"\theta",
    title = "Utilities",
    ylab = L"U_{T-2}(\theta;\bar{p})",
    legend = false);

ppb = plot(tgrid, pbar_prime1,
    xlab = L"\theta",
    title = "Next-period state",
    ylab = L"\bar{p}_{T-1} (\theta;\bar{p})",
    legend = false);

pat2 = plot(p0, at2,
    title = L"A_{T-2}(\bar{p})",
    xlab = L"\bar{p}",
    legend = false);

display(plot(plot(ppb, pat2), pU, layout = (2,1)))
savefig("julia/complementarities/finite_horizon/results/trial_soln_T2.png")

# Inspecting allocations 
pc = plot(tgrid, csol1, 
    title = L"c_{T-2}(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false);

pk = plot(tgrid, ksol1, 
    title = L"k_{T-2}(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false);

pw = plot(tgrid, wsol1, 
    title = L"w^\prime_{T-2}(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false);

display(plot(plot(pc, pk), pw, layout = (2, 1)))
savefig("julia/complementarities/finite_horizon/results/trial_allocns_T2.png")
rtcpc