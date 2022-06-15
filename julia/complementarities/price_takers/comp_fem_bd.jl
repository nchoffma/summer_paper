#= 

Solves the model with CES aggregation in production, using Roozbeh
Hosseini's Finite Element Method (FEM) code

This code: assumes bounded distribution, to reconcile the two methods

=#

using LinearAlgebra, Plots, LaTeXStrings, FastGaussQuadrature, Printf, ForwardDiff,
    ReverseDiff, Interpolations

gr()

# Parameters
β = 0.95
θ_min = 1.0
θ_max = 2.0     
w = 1.
ϵ = 4.
α = 3. # Pareto Distribution

# Distribution for output
function fdist(x)
    # F(x), f(x), f′(x) for bounded uniform
    
    L = θ_min
    H = θ_max 
    
    cdf = (x - L) / (H - L)
    pdf = 1. / (H - L)
    fpx = 0.
    return cdf, pdf, fpx
    
end

function par_inv(F)
    # Inverse of Uniform CDF 
    # For a given CDF value (F), calcluates the x that delivers F
    L = θ_min
    H = θ_max 
    x = F * (H - L) + L
    return x
end

# Build grid and LF allocns. 
# LF allocations, as initial guess
nt = 100
fmin = 0
fmax = 1             # percentile bounds
tmin, tmax = par_inv.([fmin fmax])
bigF = fmax - fmin
tgrid = Array(range(tmin, tmax, length = nt))
c0_lf = 0.5 * ones(nt)
klf = 0.5 * ones(nt)
c1_lf = tgrid .* klf
Ulf = log.(c0_lf) + β * log.(c1_lf)
μ_lf = -0.1 * ones(nt)
μ_lf[1] = 0.0

# Integration
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

# Functions for Newton, FEM
function foc_k(x, U, μ, θ, Y)
    
    c0, c1, k = x
    
    # FOC vector
    focs = zeros(3)
    focs[1] = c1 / (β * c0) - k / (λ_1 * θ * c0 ^ 2) * μ - R 
    focs[2] = (Y / k * θ ^ (ϵ - 1.)) ^ (1. / ϵ) + μ / (λ_1 * θ * c0) - R 
    focs[3] = log(c0) + β * log(c1) - U 
    
    # Jacobian matrix
    dfocs = zeros(3, 3)
    dfocs[1, 1] = -c1 / (β * c0 ^ 2) + 2k * μ / (λ_1 * θ * c0 ^ 3)
    dfocs[1, 2] = 1. / (β * c0)
    dfocs[1, 3] = -μ / (λ_1 * θ * c0 ^ 2)
    
    dfocs[2, 1] = -μ / (λ_1 * θ * c0 ^ 2)
    dfocs[2, 3] = - 1. / ϵ * (Y * θ ^ (ϵ - 1)) ^ (1. / ϵ) * k ^ (-1. / ϵ - 1.)
    
    dfocs[3, 1] = 1. / c0
    dfocs[3, 2] = β / c1 
    
    return focs, dfocs
    
end

function newton_k(U, μ, θ, x0, Y)
    
    mxit = 500
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0

    while diff > tol && its < mxit
        focs, dfocs = foc_k(x0, U, μ, θ, Y)
        diff = norm(focs, Inf)
        d = dfocs \ focs 
        while minimum(x0 - d) .< 0.0
            d = d / 2.0
            if maximum(d) < tol
                fail = 1
                break
            end
        end
        if fail == 1
            # println("Warning: Newton Failed")
            # println((U, μ, θ))
            break
        end

        x0 = x0 - d 
        its += 1

    end
    return x0, fail

end

function qgausl(n, a, b)
    # Gauss-Legendre quadrature nodes and weights 
    # n nodes/weights on interval a, b

    xi, ωi = gausslegendre(n)                   # n nodes/weights on [-1, 1]
    x_new = (xi .+ 1.0) * (b - a) / 2.0 .+ a    # change interval 
    return x_new, ωi

end

# FEM Parameters
ne = nt - 1     # number of intervals
nq = 5          # number of points for GL integration

# Main FEM function
function fem_newton(Y; ω = 1.0)
    # Solves the ODE system, using the finite element method 
    # ω is the dampening/acceleration parameter 

    a0 = Ulf
    b0 = μ_lf

    tol = 1e-8
    diff = 10.0
    its = 0
    mxit = 100
    INTR = zeros(2nt, 1)
    dINTR = zeros(2nt, 2nt)

    # print(" FEM Progress \n")
    # print("----------------\n")
    # print(" its        diff\n")
    # print("----------------\n")

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
                x0 = [c0_lf[n] c1_lf[n] klf[n]]'
                x, ~ = newton_k(U, mu, th, x0, Y)
                c0, c1, k = x

                # Derivatives of allocs. w/r/t U and mu 
                ~, dfoc = foc_k(x, U, mu, th, Y)
                dU = dfoc \ [0.; 0.; 1.]
                dmu = dfoc \ [k / (λ_1 * th * c0 ^ 2); -1.0 / (λ_1 * th * c0); 0.]
                dc0_dU, dc1_dU, dk_dU = dU
                dc0_dmu, dc1_dmu, dk_dmu = dmu

                # Evauluating and updating residuals 
                ~, ft, fpt = fdist(th) ./ bigF
                FU = Upr - k / (th * c0)
                Fmu = mupr - (λ_1 * c1 / β - fpt / ft * mu - 1.0)

                INTR[n] += bs1 * wth * FU
                INTR[n + 1] += bs2 * wth * FU
                INTR[nt + n] += bs1 * wth * Fmu
                INTR[nt + n + 1] += bs2 * wth * Fmu

                # Derivatives (of residuals) for Newton step

                dFU_da_n = -1.0 / delta_n - 1.0 / th * (1.0 / c0 * dk_dU * bs1 - 
                    k / (c0 ^ 2) * dc0_dU * bs1)
                dFU_db_n = - 1.0 / th * (1.0 / c0 * dk_dmu * bs1 - 
                    k / (c0 ^ 2) * dc0_dmu * bs1)

                dFmu_da_n = -λ_1 / β * dc1_dU * bs1
                dFmu_db_n = -1.0 / delta_n - (λ_1 / β * dc1_dmu * bs1 - 
                    bs1 * fpt / ft)

                dFU_da_n1 = 1.0 / delta_n - 1.0 / th * (1.0 / c0 * dk_dU * bs2 - 
                    k / (c0 ^ 2) * dc0_dU * bs2)
                dFU_db_n1 = - 1.0 / th * (1.0 / c0 * dk_dmu * bs2 - 
                    k / (c0 ^ 2) * dc0_dmu * bs2)

                dFmu_da_n1 = -λ_1 / β * dc1_dU * bs2
                dFmu_db_n1 = 1.0 / delta_n - (λ_1 / β * dc1_dmu * bs2 - 
                    bs2 * fpt / ft)

                # Put the residuals and derivatives in the right spot
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

                # Add to residual and derivative matrices 
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
        b0 = -abs.(b0)                      # enforce negativity
        b0[[1 end]] .= 0.                        # enforce boundary condition
        

        # # Display progress 
        # if mod(its, 5) == 0
        #     @printf("%2d %12.8f\n", its, diff) 
        # end

    end
    # print("----------------\n")

    return a0, b0, INTR
end

# Solve for one Y value 

# # Multipliers
λ_0 = 1.9491874662622317
λ_1 = 1.2727276699236132
R = λ_0 / λ_1

# Y0 = 1.
# Y0 = 0.26410989 # works with ω <= 0.7
# Y0 = 0.24996683
# a0, b0, errs  = fem_newton(Y0, ω = 0.5)

function opt_allocs(U, mu, Y)
    # Given values for U and mu (as well as output Y), calculates
    # optimal allocations 
    c0 = zeros(nt)
    c1 = copy(c0)
    k = copy(c0)

    for n in 1:nt
        x0 = [c0_lf[n] c1_lf[n] klf[n]]'
        x, ~ = newton_k(U[n], mu[n], tgrid[n], x0, Y)
        c0[n], c1[n], k[n] = x
    end

    return c0, c1, k 
end
c0, c1, k = opt_allocs(a0, b0, Y0);

# Calculating updated Y 
function calc_agg_y(kpath)
    # Given allocations k(theta) for theta in the (truncated) grid,
    # calculates aggregate Y 
    
    # k function
    tgridi = range(tmin, tmax, length = nt)
    kfunc = LinearInterpolation(tgridi, kpath)
    function ces_integrand(t)
        kt = kfunc(t)
        Ft, ft, fpt = fdist(t) ./ bigF
        return (t * kt) ^ ((ϵ - 1.) / ϵ) * ft 
    end

    ynew = gauss_leg(t -> ces_integrand(t), 100, tmin, tmax) ^ (ϵ / (ϵ - 1.))
end

# Y1 = calc_agg_y(k)
# @printf " Y1       = %10.8f\n" Y1
# @printf "|Y0 - Y1| = %10.8f\n" abs(Y0 - Y1)

# Function to iterate on Y 
function solve_model(Y0; 
    ω_Y = 1.,               # Dampening on Y
    ω_inner = 1.)           # Dampening on inner loop

    # Solves the model, given starting guess for Y 
        
    Y1 = copy(Y0)
    a0 = zeros(nt)
    b0 = zeros(nt)
    mxit = 250
    its = 1
    diff = 10.0
    tol = 1e-5

    print("\nSolving for Y \n")
    print("-----------------------------\n")
    print(" its     diff         y1\n")
    print("-----------------------------\n")

    while diff > tol && its < mxit

        a0, b0, errs  = fem_newton(Y0, ω = ω_inner)
        c0p, c1p, kp = opt_allocs(a0, b0, Y0)
        Y1 = calc_agg_y(kp)
        diff = abs(Y0 - Y1)
        
        if mod(its, 1) == 0
            @printf("%3d %12.8f %12.8f\n", its, diff, Y1)
        end
        # if diff > 0.01
        #     ω = 1.2 # a bit of acceleration, may make it unstable
        # else
        #     ω = 1.0
        # end

        Y0 = ω_Y * Y1 + (1. - ω_Y) * Y0
        its += 1
    end
    c0s, c1s, ks = opt_allocs(a0, b0, Y1);

    return Y1, a0, b0, c0s, c1s, ks

end

# Multipliers
λ_0 = 1.9491874662622317
λ_1 = 1.2727276699236132
R = λ_0 / λ_1

Y0 = 0.26410989
@time Y1, a0, b0, c0, c1, k = solve_model(Y0, ω_Y = 0.2, ω_inner = 0.4);

# # Wedges and plotting
# τ_k = 1. .- c1 ./ (c0 .* β) .* (tgrid .^ ((1. - ϵ) / ϵ)) .* 
#     ((k ./ Y0) .^ (1. / ϵ)) # price-takers
# τ_b = 1. .- c1 ./ (β * R * c0)

# pt0 = plot(tgrid, [c0 k], label = [L"c_0(\theta)" L"k(\theta)"],
#     title = "t = 0", xlab = L"\theta", legend = :topleft)
# pt1 = plot(tgrid, c1,
#     label = L"c_1(\theta)",
#     title = "t = 1", xlab = L"\theta", legend = :topleft)
# pde = plot(tgrid, [a0 b0], label = [L"U" L"\mu"], 
#     title = "DE Solution", xlab = L"\theta")
# pwedge = plot(tgrid, [τ_k τ_b], 
#     label = [L"\tau_k(\theta)" L"\tau_b(\theta)"],
#     title = "Wedges " *  "\$\\epsilon = $ϵ\$", xlab = L"\theta",
#     legend = :bottomright)

# display(plot(pt0, pt1, pde, pwedge))
# savefig("julia/complementarities/soln_unbounded.png")