#= 

Solves the recursive system for the baseline allocations Û, ĉ, k̂, ŵ

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    NLsolve, ForwardDiff, Optim, LaTeXStrings, Roots, FastGaussQuadrature, Printf, DelimitedFiles

gr()

println("******** dyn_comp.jl ********")

# Parameters
β = 0.95        # discounting
θ_min = 1.
θ_max = 2.
ϵ = 4.

# Distribution for output
# Note: this shooting method assumes that mu(theta_max) = 0, which 
# requires that the distribution be bounded 
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

# GOAL: Find some combination of values that will work, for an anchor 

function full_solve(v)

    # Given a vector v = [R κ], builds the DE machinery and 
    # solves the model 

    R, κ = v 

    function de_system!(du, u, p, t)
    
        # Unpack 
        A1, A2, U = u
        C1, C2 = p 
    
        # Interim variables
        ŵ = (1. / β) * (U - log(A1 / t))
        kterm = (A1 * κ / (R * A2 * t ^ (1. / ϵ))) ^ ϵ # used several times
        B1 = A1 / t + kterm
        B2 = (A2 / A1) * kterm
        ~, ft, fpt = fdist(t)
    
        # DE system 
        du[1] = B1 - β * exp((1. - β)ŵ) * C1 - A1 * fpt / ft                        # A₁′(θ)
        du[2] = B2 - β * exp((1. - β) * (1. - 1. / ϵ) * ŵ) * C2 - A2 * fpt / ft     # A₂′(θ)
        du[3] = 1. / A1 * kterm                                                     # Û′(θ)
    
    end
    
    function dyn_shoot(U0, C)
        u0 = [1e-4 1e-4 U0] # If A₁(θ̲) or A₂(θ̲) is actually zero, then we get undefined values in de_system()
        C1, C2 = C
        p = (C1, C2)
        tspan = (θ_min, θ_max)
        prob = ODEProblem(de_system!, u0, tspan, p)
        # gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
        gpath = solve(prob, alg_hints = [:stiff]) 
        diff = norm(gpath.u[end][1:2], Inf)
        return diff 
    end

    # Try a few values
    function test_shoot(Um, C)
        # println("U0 = $Um")
        ep = try
            dyn_shoot(Um, C)
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

    c1vals = range(0.01, 20., length = 40)
    c2vals = copy(c1vals)
    test_ums = range(-30., 30., length = 41)
    for c1 in c1vals
        for c2 in c2vals
            testvalsl = [test_shoot(test_ums[i], [c1 c2]) for i in 1:length(test_ums)]
            if !all(isnan.(testvalsl))
                println([R κ c1 c2])
            end
        end
    end
    # println("All values attempted")

end

# R = 1.1
# κ = 1. 


# full_solve([R κ])

# rvals = range(1., 3., length = 10)
# kapvals = range(0.01, 10, length = 20)
# for r in rvals, kap in kapvals
#     full_solve([r kap])
# end

function de_system!(du, u, p, t)
    
    # Unpack 
    A1, A2, U = u
    C1, C2 = p 

    # Interim variables
    ŵ = (1. / β) * (U - log(A1 / t))
    kterm = (A1 * κ / (R * A2 * t ^ (1. / ϵ))) ^ ϵ # used several times
    B1 = A1 / t + kterm
    B2 = (A2 / A1) * kterm
    ~, ft, fpt = fdist(t)

    # DE system 
    du[1] = B1 - β * exp((1. - β)ŵ) * C1 - A1 * fpt / ft                        # A₁′(θ)
    du[2] = B2 - β * exp((1. - β) * (1. - 1. / ϵ) * ŵ) * C2 - A2 * fpt / ft     # A₂′(θ)
    du[3] = 1. / A1 * kterm                                                     # Û′(θ)

end


R = 3.2
κ = 0.2

U0 = -2.1
u0 = [1e-5 1e-5 U0] # If A₁(θ̲) or A₂(θ̲) is actually zero, then we get undefined values in de_system()
p = (8.157199917813927e-5, 8.157199917813927e-5)
tspan = (θ_min, θ_max)
prob = ODEProblem(de_system!, u0, tspan, p)
gpath = solve(prob, alg_hints = [:stiff])
diff = norm(gpath.u[end][1:2], Inf)

function c1_integrand(gpath, t)
    A1, A2, U = gpath(t)
    ~, ft, ~ = fdist(t)
    kterm = (A1 * κ / (R * A2 * t ^ (1. / ϵ))) ^ ϵ
    B1 = A1 / t + kterm
    return B1 * ft
end

function c2_integrand(gpath, t)
    A1, A2, U = gpath(t)
    ~, ft, ~ = fdist(t)
    kterm = (A1 * κ / (R * A2 * t ^ (1. / ϵ))) ^ ϵ
    B2 = (A2 / A1) * kterm
    return B2 * ft
end

C1_up = gauss_leg(t -> c1_integrand(gpath, t), 20, θ_min, θ_max) 
C2_up = gauss_leg(t -> c2_integrand(gpath, t), 20, θ_min, θ_max) 