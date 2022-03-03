#= 

Solves the contraction mapping implied by w=0

A(p̄) = ∫[c + k + 1/R{A(p̄⋅p̃) - p̄⋅p̂θk}]dF(θ)

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, 
    Printf, DelimitedFiles, ApproxFun

gr()
println("******** cpp_inf.jl ********")

# Parameters
const β = 0.95
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.
R = 1.2


# LF allocations, as initial guess for Newton 
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 0.5 * ones(nt)
klf = 0.5 * ones(nt)
c1_lf = tgrid .* klf
wp_lf = log.(c1_lf) ./ (1. - β)
Ulf = log.(c0_lf) + β * wp_lf
μlf = 0.1 * ones(nt)
μlf[[1 end]] .= 0. 

function fdist(x)r
    
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

function find_bracket(f; bkt0 = (-0.3, -0.2), step_size = 0.1)
    # Finds the bracket (x, x + step_size) containing the zero of 
    # the function f
    
    a, b = bkt0
    its = 0
    mxit = 1000
    
    while f(a) * f(b) > 0.0 && its < mxit
        if f(a) > 0.0           # f(a), f(b) positive
            if f(a) > f(b)      # f decreasing
                a = copy(b)
                b += step_size
            else                # f increasing
                b = copy(a)
                a -= step_size
            end
        else                    # f(a), f(b) negative
            if f(a) > f(b)      # f increasing
                b = copy(a)
                a -= step_size
            else                # f decreasing
                a = copy(b)
                b += step_size
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

# Function to solve for γ*, given A0() and p̄ 
function solve_node(γ, A0, pbar; return_all = false)
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
                    # println("warning: newton failed")
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

    # Optimal allocations along the solution 
    function opt_allocs(gpath)
        c0p = zeros(nt)
        wpp = zeros(nt)
        kpp = zeros(nt)
        
        for i in 1:nt
            t = tgrid[i]
            U, mu = gpath(t)
            c0p[i], wpp[i], kpp[i] = alloc_single(U, mu, t)
        end
        
        return c0p, wpp, kpp
    end

    # Promise-keeping (U* = 0)
    function pkc_integrand(t, gpath)
        U, mu = gpath(t)
        Ft, ft, fpt = fdist(t)
        return U * ft
    end

    bkt = find_bracket(um -> tax_shoot(um), bkt0 = (-5.5, -5.4))
    Umin_opt = find_zero(x -> tax_shoot(x), bkt) 

    u0 = [Umin_opt 0.0]
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan)
    gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
    pkc = gauss_leg(t -> pkc_integrand(t, gpath), 20, θ_min, θ_max)

    if return_all
        # If we've found γ*, this can be used to calculate allocations and update
        c0p, wpp, kpp = opt_allocs(gpath)
        A_pbar_next = gauss_leg(t -> pkc_integrand(t, gpath), 50, θ_min, θ_max)
        return gpath, c0p, wpp, kpp, A_pbar_next
    else
        # While we're looking, don't need to calculate allocations
        return pkc
    end
end

# Function to perform the iteration 
function iterate_A(a0, m_cheb)
    # Iterates on the values of the Chebyshev approx. for A() at the 
    # m_cheb nodes 

    # a0 is the inital vector of points, must be of length m_cheb 
    a1 = copy(a0)
    diff = 10.
    its = 0
    mxit = 1 #_000 
    tol = 1e-6

    # Initialize the space 
    S0 = Chebyshev(pL..pH)
    pspace = points(S0, m_cheb)

    # Iterating
    while diff > tol && its < mxit

        # Build the current approximation
        A0 = Fun(S0, ApproxFun.transform(S0, a0))

        for i in 1:m_cheb
            pbar = pspace[i]
            println("i = $i")
            println("searching for bracket")
            γ_bkt = find_bracket(γ -> solve_node(γ, A0, pbar), bkt0 = (0.08, 0.09), step_size = 0.01)
            println("bracket = $γ_bkt")
            γ_star = find_zero(γ -> solve_node(γ, A0, pbar), γ_bkt)
            println("γ* = $γ_star")
            gpath, c0p, wpp, kpp, A_pbar_next = solve_node(γ_star, A0, pbar, return_all = true)
            a1[i] = A_pbar_next
        end

        diff = norm(a0 - a1, Inf)
        a0 = copy(a1)
        its += 1

    end

    return a1, diff
end

# Form space, initial guess 
pL = 0.01
pH = 1.2
m_cheb = 8

# Initial guess for values comes from discrete (N = 30) case
a0 = reverse([1.6593, 1.6575, 1.6428, 1.5592, 1.2811, 0.7052, -0.0533, -0.6145]) # to match with p0 

a1, diff = iterate_A(a0, m_cheb)
