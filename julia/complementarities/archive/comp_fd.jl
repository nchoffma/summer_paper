#= 

Testing the use of ForwardDiff (or ReverseDiff) to get the Jacobian 
of the residuals, to make sure that the derivative calculations are correct. 

=#

using LinearAlgebra, Plots, LaTeXStrings, FastGaussQuadrature, Printf, ForwardDiff, ReverseDiff

gr()

# Parameters
β = 0.95
θ_min = 1.0
θ_max = 4.0     
w = 1.
ϵ = 4.

# Multipliers
λ_0 = 1.95
λ_1 = 0.73
R = λ_0 / λ_1

# LF allocations, as initial guess
nt = 100
tgrid = Array(range(θ_min, θ_max, length = nt))
c0_lf = 0.5 * ones(nt)
klf = 0.5 * ones(nt)
c1_lf = tgrid .* klf
Ulf = log.(c0_lf) + β * log.(c1_lf)
μ_lf = -0.3 * ones(nt)
μ_lf[[1 end]] .= 0.0

# Distribution for output
function fdist(x; bd = false)
    # F(x), f(x), f′(x) for Pareto (bounded/unbounded)
    
    a = 1.5 # shape parameter
    
    if bd 
        L = θ_min
        H = θ_max 
        den = 1.0 - (L / H) ^ a 
        
        cdf = (1.0 - L ^ a * x ^ (-a)) / den 
        pdf = a * L ^ a * x ^ (-a - 1.0) / den 
        fpx = a * (-a - 1.0) * L ^ a * x ^ (-a - 2.0)
    else
        xm = θ_min
        cdf = 1.0 - (xm / x) ^ a
        pdf = a * xm ^ a / (x ^ (a + 1.0))
        fpx = (-a - 1.0) * a * xm ^ a / (x ^ (a + 2.0))
    end
    
    return cdf, pdf, fpx

end

function foc_k_fd(x, U, μ, θ, Y)
    
    c0, c1, k = x

    # FOC vector
    focs = zeros(eltype(x), 3)
    focs = [c1 / (β * c0) - k / (λ_1 * θ * c0 ^ 2) * μ - R , 
        (Y / k * θ ^ (ϵ - 1.)) ^ (1. / ϵ) + μ / (λ_1 * θ * c0) - R,
        log(c0) + β * log(c1) - U]

    return focs 

end

it = 12
x0 = [c0_lf[it], c1_lf[it], klf[it]]
fjac_fd = ForwardDiff.jacobian(x -> foc_k_fd(x, Ulf[it], μ_lf[it], tgrid[it], Yt), x0)

function newton_k_fd(U, μ, θ, x0, Y)
    
    mxit = 500
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0

    while diff > tol && its < mxit
        focs = foc_k_fd(x0, U, μ, θ, Y)
        dfocs = ForwardDiff.jacobian(x -> foc_k_fd(x, U, μ, θ, Y), x0)
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
            println("Warning: Newton Failed")
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

ne = nt - 1     # number of intervals
nq = 5          # number of points for GL integration

function fem_resids!(INTR, x0, Y)
    # Takes in pre-allocated residuals and guess x0,
    # returns new residuals

    a0 = x0[1:nt]
    b0 = x0[nt + 1:end]

    tol = 1e-8
    diff = 10.0
    its = 0
    mxit = 1 #10_000
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
            xx = [c0_lf[n] c1_lf[n] klf[n]]'
            x, ~ = newton_k_fd(U, mu, th, xx, Y)
            c0, c1, k = x

            # Evauluating and updating residuals 
            ~, ft, fpt = fdist(th)
            FU = Upr - k / (th * c0)
            Fmu = mupr - (λ_1 * c1 / β - fpt / ft * mu - 1.0)

            INTR[n] += bs1 * wth * FU
            INTR[n + 1] += bs2 * wth * FU
            INTR[nt + n] += bs1 * wth * Fmu
            INTR[nt + n + 1] += bs2 * wth * Fmu

        end    
    end
    return INTR
end

Yt = 1.
x0 = [Ulf; μ_lf]
resids = zeros(2nt)
errs_2 = fem_resids!(resids, x0, Yt)
derrs_a = ForwardDiff.jacobian(x -> fem_resids!(zeros(eltype(x), 2nt), x, Yt), x0)
# derrs_a = ReverseDiff.jacobian(x -> fem_resids!(zeros(eltype(x), 2nt), x, Yt), x0)