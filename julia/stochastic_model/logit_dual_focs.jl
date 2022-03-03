
# Testing FOCs in dual to planner's problem

using DifferentialEquations, Distributions, LinearAlgebra, Plots, 
    QuadGK, NLsolve, ForwardDiff, Optim, LaTeXStrings


# Parameters
α = 0.8         # Pr(y>0)
β = 0.95        # discounting
θ_min = 2.0     # up from 3.2
θ_max = 4.0     # up from 4.6
w = 1.2         
R = 1.8         # 

#= 

Can set R now, but if it's below 1.9 or so, these get weird

=#

# Promise-Keeping
Ustar = -2.0        # minimum total utility 
γ = 1.5             # multiplier on PKC

# LF allocations, as initial guess
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
c0_lf = 0.6 * ones(nt)
klf = 0.5 * ones(nt)
blf = 0.1 * ones(nt)
c1y_lf = tgrid .* klf
c1u_lf = R * blf
Ulf = log.(c0_lf) + β * (α * log.(c1y_lf) + (1 - α) * log.(c1u_lf))

μ_lf = 0.1 * ones(nt)
μ_lf[[1 end]] .= 0.0

function foc_kp!(focs, x, U, μ, θ, Um)
    c0, k, c1y, c1u, ϕ = x
    focs[1] = 1.0 - c1y / (β * R * c0) + ϕ / (c0 + k) - μ * k / (θ * c0 ^ 2)    # c0 
    focs[2] = c1y - c1u - β * R * ϕ / (1.0 - α)                                 # c1u
    focs[3] = 1.0 + μ / (θ * c0) + ϕ / (c0 + k) - α * θ / R                     # k 
    focs[4] = U - log(c0) - β * (α * log(c1y) + (1 - α) * log(c1u))
    focs[5] = Um - log(c0 + k) - β * log(c1u)
    return focs
    
end

function foc_k0!(focs, x, U)
    c0, c1 = x 
    focs[1] = log(c0) + β * log(c1) - U
    focs[2] = 1.0 - c1 / (β * R * c0)
    return focs 

end

function foc_newton_kp(U, μ, θ, i, Um)
    x0 = [c0_lf[i] klf[i] c1y_lf[i] c1u_lf[i] 0.5]'
    mxit = 200
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0
    
    while diff > tol && its < mxit
        focs = foc_kp!(similar(x0), x0, U, μ, θ, Um)
        diff = maximum(abs.(focs))
        dfoc = ForwardDiff.jacobian(x -> foc_kp!(similar(x), x, U, μ, θ, Um), x0)
        d = dfoc \ focs
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

function foc_newton_k0(U, i)
    x0 = [c0_lf[i] c1u_lf[i]]'
    mxit = 200
    tol = 1e-6
    diff = 10.0
    its = 0
    fail = 0
    
    while diff > tol && its < mxit
        focs = foc_k0!(similar(x0), x0, U)
        diff = maximum(abs.(focs))
        dfoc = ForwardDiff.jacobian(x -> foc_k0!(similar(x), x, U), x0)
        d = dfoc \ focs
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
    if its == mxit && diff > tol
        fail == 1
    end
    return x0, fail
end

function opt_allocs()
    c0 = zeros(nt)
    k = copy(c0)
    c1y = copy(c0)
    c1u = copy(c0)
    ϕ = copy(c0)
    for i = 1:nt
        x, f = foc_newton_kp(Ulf[i], μ_lf[i], tgrid[i], i, Ulf[1])
        if f == 1
            x0, f0 = foc_newton_k0(Ulf[i], i)
            if f0 == 1
                println(i)
                println("neither converged")
            else
                c0[i], c1u[i] = x0
                c1y[i] = c1u[i]
            end
        else
            c0[i], k[i], c1y[i], c1u[i], ϕ[i] = x 
        end
    end

    return c0, k, c1y, c1u, ϕ
end

c0, k, c1y, c1u, ϕ = opt_allocs()

pt0 = plot(tgrid, [c0 k], label = [L"c_0(\theta)" L"k(\theta)"],
    title = "t = 0", xlab = L"\theta", legend = :topleft)
pt1 = plot(tgrid, [c1u c1y ϕ],
    label = [L"c_1^0(\theta)" L"c_1^y(\theta)" L"\phi(\theta)"],
    title = "t = 1", xlab = L"\theta", legend = :topleft)
plot(pt0, pt1, layout = (2, 1))
