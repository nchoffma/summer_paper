#= Testing in the CPP =#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles, ApproxFun

gr()
println("******** cpp_inf_TEST.jl ********")

# Parameters
const β = 0.95        # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

# # LF allocations, as initial guess for Newton 
# nt = 30
# tgrid = range(θ_min, θ_max, length = nt)
# c0_lf = 0.5 * ones(nt)
# klf = 0.5 * ones(nt)
# c1_lf = tgrid .* klf
# wp_lf = log.(c1_lf) ./ (1. - β)
# Ulf = log.(c0_lf) + β * wp_lf
# μlf = 0.1 * ones(nt)
# μlf[[1 end]] .= 0. 

# Read in values from the discrete case, for initial guesses
nt = 50
tgrid = range(θ_min, θ_max, length = nt)
matpath = "julia/complementarities/discrete_results/"
c0_lf = readdlm(matpath * "csol.txt", '\t', Float64, '\n')
klf   = readdlm(matpath * "ksol.txt", '\t', Float64, '\n')
wp_lf = readdlm(matpath * "wsol.txt", '\t', Float64, '\n')
a0_ig = readdlm(matpath * "a0sol.txt", '\t', Float64, '\n')
Ulf = log.(c0_lf) + β * wp_lf
μlf = 0.1 * ones(nt)
μlf[[1 end]] .= 0. 

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

function cheb_approx(fvals, a, b)
    # Approximates the function defined by fvals on m_cheb nodes
    # over interval [a,b] using Chebyshev polynomials of order n_cheb
    # Uses the default order of n = m - 1, which is fine 

    m_cheb = length(fvals)
    S = Chebyshev(a..b)
    p = points(S, m_cheb)
    fx = Fun(S, ApproxFun.transform(S, fvals))
    return fx

end

function foc_k(x, U, μ, θ)
        
    c, wp, k = x
    ptilde = exp(-(1. - β) / ϵ * wp)
    phat = (θ * k) ^ (-1. / ϵ)
    pnext = ptilde * pbar

    # Force the next-period state to be in the domain
    if pnext < pL
        pnext = pL 
    elseif pnext > pH 
        pnext = pH
    end

    # FOC vector
    focs = zeros(eltype(x), 3)
    focs[1] = 1. - A0(pnext) * (1. - β) * exp((1. - β) * wp) / (β * R * c) - μ * k / (θ * c ^ 2)
    focs[2] = 1. - 1. / R * pbar * phat * θ + μ / (θ * c)
    focs[3] = log(c) + β * wp - U 
    
    return focs
    
end

function newton_k(U, μ, θ, i)
    ipr = m_cheb - ip + 1 # to get indexing direction right 
    x0 = [c0_lf[i, ipr] wp_lf[i, ipr] klf[i, ipr]]'
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
    ptilde = exp(-(1. - β) / ϵ * wp)
    phat = (t * k) ^ (-1. / ϵ)
    ~, ft, fpt = fdist(t)
    du[1] = k / (c0 * t)
    du[2] = γ - A0(pbar * ptilde) / (β * R) * (1. - β) * exp((1. - β) * wp) - fpt / ft * μ
end

function tax_shoot(U_0)
    u0 = [U_0 0.0]
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan)
    gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
    return gpath.u[end][2] # want to get μ(θ̲) = 0
    
end

function test_shoot(Um)
    println("U0 = $Um")
    ep = try
        tax_shoot(Um)
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

# State space for p̄
pL = 0.1
pH = 1.2
m_cheb = 8

R = 1.2

# Build initial guess 
S0 = Chebyshev(pL..pH)
p0 = points(S0, m_cheb)

# Initial guess for values comes from discrete (N = 30) case 
a0_pts = reverse(a0_ig[:, 1]) # to match with p0 
A0 = Fun(S0, ApproxFun.transform(S0, a0_pts))

# ip = m_cheb # γ∈[0.082, 0.083]
ip = 3
pbar = p0[ip] 
γ = 0.048

# Attempt to clear FOCs
function test_focs()
    ct = zeros(nt)
    wt = zeros(nt)
    kt = zeros(nt)
    ipr = m_cheb - ip + 1 # to get indexing direction right 


    for i in 1:nt
        # println(i)
        ct[i], wt[i], kt[i] = alloc_single(Ulf[i, ipr], μlf[i], tgrid[i])
    end

    return ct, wt, kt

end

# ct, wt, kt = test_focs()
# display(plot(tgrid, [ct wt kt]))

test_ums = range(-10., 0., length = 21)
testvals = [test_shoot(test_ums[i]) for i in 1:length(test_ums)]
display(plot(test_ums, testvals))

# bkt = find_bracket(um -> tax_shoot(um), bkt0 = (-3.5, -3.4))
# Umin_opt = find_zero(x -> tax_shoot(x), bkt) 

# u0 = [Umin_opt 0.0]
# tspan = (θ_min, θ_max)
# prob = ODEProblem(de_system!, u0, tspan)
# gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 

# # Promise-keeping (U* = 0)
# function pkc_integrand(t, gpath)
#     U, mu = gpath(t)
#     Ft, ft, fpt = fdist(t)
#     return U * ft
# end

# pkc = gauss_leg(t -> pkc_integrand(t, gpath), 20, θ_min, θ_max)