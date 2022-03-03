#= 

Solves the dynamic (T<∞) case using the Finite Element Method
This code uses the analytic derivatives

Note: the syntax A0' works for the derivative of the Chebyshev approximation

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles, ApproxFun, Zygote

gr()
println("******** comp_finite_vf.jl ********")

# Parameters
const β = 0.95        # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

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

ne = nt - 1     # number of intervals
nq = 5          # number of points for GL integration

# Shifters and state functions
capT = 2
shft = (1. - β) / (1. - β ^ (capT + 1))             # shifter for finite horizon, t = 0,1,...,T on allocations
shft_w = (1. - β ^ capT) / (1. - β ^ (capT + 1) )   # shifter on promise utility
p_tilde(w) = exp(-1. / ϵ * shft * shft_w * w)
p_hat(t, k) = (t * k) ^ (-1. / ϵ)

# Distribution for output 
function fdist(x)
    
    L = θ_min
    H = θ_max 
    
    # Uniform
    cdf = (x - L) / (H - L)
    pdf = 1. / (H - L)
    fpx = 0.
    
    return cdf, pdf, fpx
    
end

function qgausl(n, a, b)
    # Gauss-Legendre quadrature nodes and weights 
    # n nodes/weights on interval a, b

    xi, ωi = gausslegendre(n)                   # n nodes/weights on [-1, 1]
    x_new = (xi .+ 1.0) * (b - a) / 2.0 .+ a    # change interval 
    return x_new, ωi

end

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

function foc_k(x, U, μ, θ)

    c, wp, k = x
    ptilde = p_tilde(wp)
    phat = p_hat(θ, k)
    pnext = ptilde * pbar

    # # Force the next-period state to be in the domain
    # if pnext < pL
    #     pnext = pL 
    #     # println("trimming low")
    # elseif pnext > pH 
    #     pnext = pH
    #     # println("trimming high")
    # end

    # FOCs
    focs = zeros(3)
    focs[1] = 1. - A0(pnext) * shft * exp(shft * wp) / (β * R * c) - μ * k / (θ * c ^ 2)
    focs[2] = 1. - 1. / R * pbar * phat * θ + μ / (θ * c)
    focs[3] = log(c) + β * wp - U 

    # Jacobian
    dfocs = zeros(3, 3)
    gw = -A0'(pnext) * exp((shft - shft / ϵ) * wp) * pbar * shft ^ 2 / ϵ + 
        A0(pnext) * shft ^ 2 * exp(shft * wp)
    Gw = A0(pnext) * shft * exp(shft * wp)
    
    dfocs[1, 1] = 1 / (β * R * c ^ 2) * Gw + 2 * μ * k / (θ * c ^ 3)
    dfocs[1, 2] = -1 / (β * R * c) * gw 
    dfocs[1, 3] = -μ / (θ * c ^ 2)

    dfocs[2, 1] = -μ / (θ * c ^ 2)
    dfocs[2, 3] = 1 / (R * ϵ) * pbar * θ ^ (1. - 1. / ϵ) * k ^ (-1. / ϵ - 1.)

    dfocs[3, 1] = 1. / c
    dfocs[3, 2] = β

    return focs, dfocs

end

function newton_k(U, μ, θ, x0)
    mxit = 500
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0
    
    while diff > tol && its < mxit
        focs, dfocs = foc_k(x0, U, μ, θ)
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

function fem_newton(; ω = 1.)
    # Solves the ODE system, using the finite element method 
    # ω is the dampening/acceleration parameter 

    a0 = Ulf
    b0 = μlf

    tol = 1e-8
    diff = 10.0
    its = 0
    mxit = 100
    INTR = zeros(2nt, 1)
    dINTR = zeros(2nt, 2nt)

    print(" FEM Progress \n")
    print("----------------\n")
    print(" its        diff\n")
    print("----------------\n")

    while diff > tol && its < mxit
        # Residual matrices
        INTR = zeros(2nt, 1)
        dINTR = zeros(2nt, 2nt)

        for n in 1:ne
            
            # Get theta values for interval 
            x1 = tgrid[n]
            x2 = tgrid[n + 1]

            # Get nodes and weights 
            thq, wq = qgausl(nq, x1, x2)

            for i in 1:nq
                # Get weights, lengths, etc.
                th = thq[i]
                wth = wq[i]
                delta_n = x2 - x1 
                ep = 2.0 * (th - x1) / delta_n - 1.0
                bs1 = 0.5 * (1.0 - ep)
                bs2 = 0.5 * (1.0 + ep)

                # Approximations and allocations
                U = a0[n] * bs1 + a0[n + 1] * bs2
                mu = b0[n] * bs1 + b0[n + 1] * bs2
                Upr = (a0[n + 1] - a0[n]) / delta_n
                mupr = (b0[n + 1] - b0[n]) / delta_n

                xx = [c0_lf[n] wp_lf[n] klf[n]]'
                x, ~ = newton_k(U, mu, th, xx)
                c, wp, k = x
                ptilde = p_tilde(wp)
                phat = p_hat(th, k)
                pnext = ptilde * pbar

                # # Force the next-period state to be in the domain
                # if pnext < pL
                #     pnext = pL 
                #     # println("trimming low")
                # elseif pnext > pH 
                #     pnext = pH
                #     # println("trimming high")
                # end

                # Evauluating and updating residuals 
                ~, ft, fpt = fdist(th)
                FU = Upr - k / (th * c)
                ptilde = p_tilde(wp)
                pnext = pbar * ptilde
                mu_a = γ - A0(pnext) / (β * R) * shft * exp(shft * wp) 
                    - fpt / ft * mu
                Fmu = mupr - mu_a

                INTR[n] += bs1 * wth * FU
                INTR[n + 1] += bs2 * wth * FU
                INTR[nt + n] += bs1 * wth * Fmu
                INTR[nt + n + 1] += bs2 * wth * Fmu

                # Derivatives of allocs. w/r/t U and mu 
                ~, dfoc = foc_k(x, U, mu, th)
                dU = dfoc \ [0.; 0.; 1.]
                dmu = dfoc \ [k / (th * c ^ 2); -1.0 / (th * c); 0.]
                dc0_dU, dw_dU, dk_dU = dU
                dc0_dmu, dw_dmu, dk_dmu = dmu
                
                # Derivatives (of residuals) for Newton step

                gw = -A0'(pnext) * exp((shft - shft / ϵ) * wp) * pbar * shft ^ 2 / ϵ + 
                    A0(pnext) * shft ^ 2 * exp(shft * wp)
                Gw = A0(pnext) * shft * exp(shft * wp)
                
                dFU_da_n = -1. / delta_n + k / (th * c ^2) * dc0_dU * bs1 - 
                    1. / (th * c) * dk_dU * bs1
                dFU_db_n = k / (th * c ^2) * dc0_dmu * bs1 - 
                    1. / (th * c) * dk_dmu * bs1
                
                dFmu_da_n = 1 / (β * R) * gw * dw_dU * bs1 
                dFmu_db_n = -1 / delta_n + 1 / (β * R) * gw * dw_dmu * bs1 - fpt / ft * bs1

                dFU_da_n1 = 1 / delta_n + k / (th * c ^2) * dc0_dU * bs2 - 
                    1. / (th * c) * dk_dU * bs2
                dFU_db_n1 = k / (th * c ^2) * dc0_dmu * bs2 - 
                    1. / (th * c) * dk_dmu * bs2
                
                dFmu_da_n1 = 1 / (β * R) * gw * dw_dU * bs2 
                dFmu_db_n1 = 1 / delta_n + 1 / (β * R) * gw * dw_dmu * bs2 - fpt / ft * bs2

                # Fill in the Jacobian for the Newton step 
                dFU = zeros(2nt)
                dFmu = zeros(2nt)

                dFU[n]      = dFU_da_n;
                dFU[n + 1]  = dFU_da_n1;
                dFU[nt + n] = dFU_db_n;
                dFU[nt + n + 1] = dFU_db_n1;

                dFmu[n]          = dFmu_da_n;
                dFmu[n + 1]      = dFmu_da_n1;
                dFmu[nt + n]     = dFmu_db_n;
                dFmu[nt + n + 1] = dFmu_db_n1;

                # Add to derivative matrix
                for k in 1:2nt
                    dINTR[n, k] += wth * bs1 * dFU[k]
                    dINTR[n + 1, k] += wth * bs2 * dFU[k]
                    dINTR[nt + n, k] += wth * bs1 * dFmu[k]
                    dINTR[nt + n + 1, k] += wth * bs2 * dFmu[k]
                end

            end
        end

        diff = norm(INTR, Inf)
        its += 1

        # Newton update
        dstep = dINTR \ INTR
        a0 = a0 - dstep[1:nt] * ω
        b0 = b0 - dstep[nt + 1:end] * ω
        b0 = abs.(b0)                      # enforce positivity
        b0[[1 end]] .= 0.                  # enforce boundary conditions

        # Display progress 
        if mod(its, 10) == 0
            @printf("%2d %12.8f\n", its, diff) 
        end

    end

    return a0, b0, its, diff, INTR, dINTR
end

# State space for p̄
pL = 0.9
pH = 1.1
m_cheb = 5

R = 1.5

# Build initial guess 
S0 = Chebyshev(pL..pH)
p0 = points(S0, m_cheb)

# # Initial Guess
# a0_pts = ones(m_cheb) # analytically, A_T = 1 for all pbar
# A0 = Fun(S0, ApproxFun.transform(S0, a0_pts))

A0(x) = x ^ 0. # Avoid the derivative being nothing()

# Testing γ
ip = 3
pbar = p0[ip]
γ = 0.3

a0, b0, its, diff, INTR, ~ = fem_newton(ω = 0.5);

pu = plot(tgrid, a0, 
    legend = false,
    title = "U");
pmu = plot(tgrid, b0, 
    legend = false,
    title = "mu");
pdu = plot(tgrid, INTR[1:nt], 
    legend = false,
    title = "Uerr");
pdmu = plot(tgrid, INTR[nt+1:2nt], 
    legend = false,
    title = "muerr")
psys = plot(pu, pmu, pdu, pdmu)

function opt_allocs(U, mu)
    ct = zeros(nt)
    kt = zeros(nt)
    wpt = zeros(nt)
    for n in 1:nt
        x0 = [c0_lf[n] wp_lf[n] klf[n]]'
        x, ~ = newton_k(U[n], mu[n], tgrid[n], x0)
        ct[n], wpt[n], kt[n] = x
    end

    return ct, wpt, kt
end

ct, wpt, kt = opt_allocs(a0, b0);

palloc = plot(tgrid, [ct kt wpt],
    label = [L"c(\theta)" L"k(\theta)" L"w^\prime(\theta)"],
    title = "Allocations",
    xlab = L"\theta")
display(plot(psys, palloc, layout = (2, 1)))

# kt[end] / (tgrid[end] * ct[end])