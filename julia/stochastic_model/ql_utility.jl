
# Solves the model with quasilinear utility

using LinearAlgebra, Distributions, Plots, DifferentialEquations

# Parameters
α = 0.5         # Pr(y>0)
β = 0.95        # discounting
θ_min = 1.0     # works with 3.8
θ_max = 3.0     # looks ok with 4.6     
w = 1.2

# Interest rate
R = 1.0
λ_1 = 1.0
λ_0 = R * λ_1
0
# Solve over a grid of theta values
Nt = 100
θ_grid = range(θ_min, θ_max, length = Nt)

# Distribution for θ
function par(x; bd = false)
    # F(x), f(x), f′(x) for Pareto (bounded/unbounded)
    
    # a = 1.5 # shape parameter
    
    # if bd 
    #     L = θ_min
    #     H = θ_max 
    #     den = 1.0 - (L / H) ^ a 
        
    #     cdf = (1.0 - L ^ a * x ^ (-a)) / den 
    #     pdf = a * L ^ a * x ^ (-a - 1.0) / den 
    #     fpx = a * (-a - 1.0) * L ^ a * x ^ (-a - 2.0)
    # else
    #     xm = θ_min
    #     cdf = 1.0 - (xm / x) ^ a
    #     pdf = a * xm ^ a / (x ^ (a + 1.0))
    #     fpx = (-a - 1.0) * a * xm ^ a / (x ^ (a + 2.0))
    # end
    
    # Uniform
    a = θ_min
    b = θ_max
    cdf = (x - a) / (b - a)
    pdf = 1.0 / (b - a)
    fpx = 0.0
    
    return cdf, pdf, fpx

end

function de_system!(du, u, p, t)
    U, μ = u    # current guess
    Um = p[1]   # make sure this is a number

    # Solve for unknowns
    c1y = α * β * t + μ / (λ_1 * t)
    ϕ = α * λ_1 * t + μ / t - R * λ_1
    if ϕ > 0.0
        c1u = c1y - β * ϕ / (λ_1 * (1.0 - α))
        c0 = U - β * (α * log(c1y) + (1.0 - α) * log(c1u))
        k = Um - c0 - β * log(c1u)
    else
        ϕ = 0.0
        k = 0.0
        c1u = copy(c1y)
        c0 = U - β * log(c1y)
    end

    # Get differential equation values 
    du[1] = k / t 

    ~, ft, fpt = par(t)
    if t == θ_min
        du[2] = λ_1 * c1y / β - μ * fpt / ft - 1.0 - ϕ
    else
        du[2] = λ_1 * c1y / β - μ * fpt / ft - 1.0
    end


end

# Shooting algorithm
function tax_shoot(Umin)
    u0 = [Umin 0.0]
    p = (Umin)  # pass this as parameter and bound
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan, p)
    gpath = solve(prob) 
    return gpath.u[end][2] # want to get μ(θ̲) = 0

end

tax_shoot(5.0)

# @time Umin_opt = find_zero(tax_shoot, bkt)