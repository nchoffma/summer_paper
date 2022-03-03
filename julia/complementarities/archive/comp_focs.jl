
# FOCs in the planner's problem with CES aggregator

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    QuadGK, NLsolve, ForwardDiff, Optim, LaTeXStrings, FastGaussQuadrature

# Parameters
β = 0.95        # discounting
θ_min = 1.1
θ_max = 4.0
w = 1.0
ϵ = 0.8

# Interest rate
R = 1.1
λ_1 = 1.0
λ_0 = R * λ_1

# LF allocations, as initial guess
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 0.5 * ones(nt)
klf = 0.5 * ones(nt)
c1_lf = tgrid .* klf
Ulf = log.(c0_lf) + β * log.(c1_lf)
μ_lf = -0.3 * ones(nt)
μ_lf[[1 end]] .= 0.0
Y_lf = 0.6 # may be able to do better, but it shouldn't matter that much 

# Distribution for output
function fdist(x)
    
    # Bounded Pareto
    a = 1.5 # shape parameter
    L = θ_min
    H = θ_max 
    den = 1.0 - (L / H) ^ a 
    
    cdf = (1.0 - L ^ a * x ^ (-a)) / den 
    pdf = a * L ^ a * x ^ (-a - 1.0) / den 
    fpx = a * (-a - 1.0) * L ^ a * x ^ (-a - 2.0)
    
    return cdf, pdf, fpx

end

function foc_k(x, U, μ, θ, Y)
    
    c0, k, c1 = x

    # FOC matrix
    focs = zeros(eltype(x), 3)
    focs[1] = c1 / (β * c0) - k / (θ * c0 ^ 2) * μ - R 
    focs[2] = (Y * θ ^ (ϵ - 1) / k) ^ (1.0 / ϵ) + μ / c0 - R 
    focs[3] = log(c0) + β * log(c1) - U 

    # Jacobian matrix
    dfocs = zeros(eltype(x), 3, 3)
    dfocs[1, 1] = -c1 / (β * c0 ^ 2) + 2k * μ / (θ * c0 ^ 3)
    dfocs[1, 2] = 1.0 / (β * c0)
    dfocs[1, 3] = -μ / (θ * c0 ^ 2)

    dfocs[2, 1] = -μ / (c0 ^ 2)
    dfocs[2, 3] = - 1.0 / ϵ * (Y * θ ^ (ϵ - 1)) ^ (1.0 / ϵ) * k ^ (-1.0 / ϵ - 1.0)

    dfocs[3, 1] = 1.0 / c0
    dfocs[3, 2] = β / c1 

    return focs, dfocs

end

i = 1
x0 = [c0_lf[i] c1_lf[i] klf[i]]'
foc_a, dfoc_a = foc_k(x0, Ulf[i], μ_lf[i], tgrid[i], Y_lf)
dfoc_fd = ForwardDiff.jacobian(x -> foc_k(x, Ulf[i], μ_lf[i], tgrid[i], Y_lf)[1], x0)
all(dfoc_a .≈ dfoc_fd)

function newton_k(U, μ, θ, i, Y)
    x0 = [c0_lf[i] klf[i] c1_lf[i]]'
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
            println("Warning: Newton Failed")
            break
        end

        x0 = x0 - d 
        its += 1

    end
    return x0, fail

end

function opt_allocs()
    c0 = zeros(nt)
    c1 = copy(c0)
    k = copy(c0)

    for i = 1:nt
        x, f = newton_k(Ulf[i], μ_lf[i], tgrid[i], i, Y_lf)
        c0[i], c1[i], k[i] = x 
    end

    return c0, c1, k
end

c0, c1, k = opt_allocs()

# pt0 = plot(tgrid, [c0 k], label = [L"c_0(\theta)" L"k(\theta)"],
#     title = "t = 0", xlab = L"\theta", legend = :topleft)
# pt1 = plot(tgrid, c1,
#     label = L"c_1(\theta)",
#     title = "t = 1", xlab = L"\theta", legend = :topleft)
# plot(pt0, pt1, layout = (2, 1))