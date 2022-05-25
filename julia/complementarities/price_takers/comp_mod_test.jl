#= 

Solves the model with complementarities
Testing script: without solving function that takes lambdas

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, Printf

gr()

println("******** comp_mod_TEST.jl ********")

# Parameters
const β = 0.9        # discounting
const θ_min = 1.0
const θ_max = 4.0

w = 1.
ϵ = 4.

# Multipliers (works with λ_1 = 1., λ_0 = 2.)
# The solution (and stability) is VERY sensitive to these. 
# Move in small increments!
# λ_1 = 0.581
# λ_0 = 1.25

# LF allocations, as initial guess for Newton 
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 1. / (1. + β) * w * ones(nt)
klf = β / (1. + β) * w * ones(nt)
c1_lf = tgrid .* klf

# Distribution for output

# Functions for truncated normal dist 
tmean = (θ_max + θ_min) / 2
tsig = 0.5
function tnorm_pdf(x)
    pdf.(truncated(Normal(tmean, tsig), θ_min, θ_max), x)
end

function tnorm_cdf(x) # for convenience
    cdf.(truncated(Normal(tmean, tsig), θ_min, θ_max), x)
end

function tnorm_fprime(x)
    ForwardDiff.derivative(tnorm_pdf, x)
end 

function fdist(x)
    
     # Truncated normal
     cdf = tnorm_cdf(x)
     pdf = tnorm_pdf(x)
     fpx = tnorm_fprime(x)
    
    return cdf, pdf, fpx
    
end


function foc_k(x, U, μ, θ, Y)
    
    c0, c1, k = x

    # FOC vector
    focs = zeros(3)
    focs[1] = c1 / (β * c0) - k / (θ * c0 ^ 2) * μ - R 
    focs[2] = (Y / k * θ ^ (ϵ - 1)) ^ (1.0 / ϵ) + μ / c0 - R 
    focs[3] = log(c0) + β * log(c1) - U 

    # Jacobian matrix
    dfocs = zeros(3, 3)
    dfocs[1, 1] = -c1 / (β * c0 ^ 2) + 2k * μ / (θ * c0 ^ 3)
    dfocs[1, 2] = 1.0 / (β * c0)
    dfocs[1, 3] = -μ / (θ * c0 ^ 2)

    dfocs[2, 1] = -μ / (c0 ^ 2)
    dfocs[2, 3] = - 1.0 / ϵ * (Y * θ ^ (ϵ - 1)) ^ (1.0 / ϵ) * k ^ (-1.0 / ϵ - 1.0)

    dfocs[3, 1] = 1.0 / c0
    dfocs[3, 2] = β / c1 

    return focs, dfocs

end

function newton_k(U, μ, θ, i, Y)
    x0 = [c0_lf[i] c1_lf[i] klf[i]]'
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
            break
        end

        x0 = x0 - d 
        its += 1

    end
    return x0, fail

end

function alloc_single(U, μ, t, Y)

    it = findmin(abs.(tgrid .- t))[2] # for finding initial guess
    
    x, f = newton_k(U, μ, t, it, Y)
    c0, c1, k = x 
    return c0, c1, k

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

function ces_integrand(t, gpath, Y)
    # CES aggregator for output
    # gpath is the solution to the ODE system at Y 
    U, μ = gpath(t)
    ~, ~, k = alloc_single(U, μ, t, Y)
    ~, ft, ~ = fdist(t)
    return (t * k) ^ ((ϵ - 1) / ϵ) * ft
    
end

function de_system!(du, u, p, t)
    U, μ = u
    Y = p[1]
    c0, c1, k = alloc_single(U, μ, t, Y)

    ~, ft, fpt = fdist(t)
    du[1] = k / (c0 * t)
    du[2] = λ_1 * c1 / β - μ * fpt / ft - 1
end

function tax_shoot(U_0, Y)
    u0 = [U_0 0.0]
    p = (Y)
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan, p)
    gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
    return gpath.u[end][2] # want to get μ(θ̲) = 0

end

function find_bracket(f; bkt0 = (-0.3, -0.2))
    # Finds the bracket (x, x + 0.1) containing the zero of 
    # the function f

    a, b = bkt0
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

# Try a few values
function test_shoot(Um, Y)
    println("U0 = $Um")
    ep = try
        tax_shoot(Um, Y)
    catch y # all the different ways it can fail 
        if isa(y, UndefVarError)
            NaN
        elseif isa(y, LAPACKException)
            NaN
        elseif isa(y, DomainError)
            NaN
        elseif isa(y, SingularException)
            NaN
        end
    end
    return ep
end

# Test params 
λ_0 = 1.9
λ_1 = 1.
R = λ_0 / λ_1 
@printf "\nλ_0 = %.4f\nλ_1 = %.4f\nR   = %.4f\n" λ_0 λ_1 R

Yt = 1.

# test_ums = range(-5., -0., length = 51)
# testvals = [test_shoot(test_ums[i], Yt) for i in 1:length(test_ums)]
# display(plot(test_ums, testvals))

bkt = find_bracket(um -> tax_shoot(um, Yt), bkt0 = (-3.5, -3.4))
println("bracket found")
Umin_opt = find_zero(x -> tax_shoot(x, Yt), bkt)
u0 = [Umin_opt, 0.0]
p = (Yt)
tspan = (θ_min, θ_max)
prob = ODEProblem(de_system!, u0, tspan, p)
gpath = solve(prob, alg_hints = [:stiff]) 
y_upd = gauss_leg(t -> ces_integrand(t, gpath, Yt), 20, θ_min, θ_max) ^ (ϵ / (ϵ - 1.))
