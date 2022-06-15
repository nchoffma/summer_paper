#= 

Solves the planner's infinite-horixon CMP, 
with θ∼N(μ,σ) and households as monopolists

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, NLsolve, 
    ForwardDiff, Optim, LaTeXStrings, Roots, 
    FastGaussQuadrature, Printf, DelimitedFiles, ApproxFun

gr()
println("******** cpp_inf_mpl_test.jl ********")

# Parameters
β = 0.9         # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

R = 1.1

solnpath = "julia/complementarities/monopolists/results/"

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

# Functions for truncated normal dist 
tmean = (θ_max + θ_min) / 2
tsig = 0.3
function tnorm_pdf(x)
    pdf.(truncated(Normal(tmean, tsig), θ_min, θ_max), x)
end

function tnorm_cdf(x) # for convenience
    cdf.(truncated(Normal(tmean, tsig), θ_min, θ_max), x)
end

function tnorm_fprime(x)
    ForwardDiff.derivative(tnorm_pdf, x)
end 

# Distribution for output 
function fdist(x)

    # # # Uniform
    # L = θ_min
    # H = θ_max 
    # cdf = (x - L) / (H - L)
    # pdf = 1. / (H - L)
    # fpx = 0.

    # Truncated normal
    cdf = tnorm_cdf(x)
    pdf = tnorm_pdf(x)
    fpx = tnorm_fprime(x)
    
    return cdf, pdf, fpx
    
end

# Gauss-legendre quadrature
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

function find_bracket(f; bkt0 = (-0.3, -0.2), step = 0.1, print_every = false)
    # Finds the bracket (x, x + step) containing the zero of 
    # the function f
    
    a, b = bkt0
    its = 0
    mxit = 50_000

    if print_every
        println([a, f(a), b, f(b)])
    end
    
    while f(a) * f(b) > 0.0 && its < mxit
        if f(a) > 0.0           # f(a), f(b) positive
            if f(a) > f(b)      # f decreasing
                a = copy(b)
                b += step
            else                # f increasing
                b = copy(a)
                a -= step
            end
        else                    # f(a), f(b) negative
            if f(a) > f(b)      # f increasing
                b = copy(a)
                a -= step
            else                # f decreasing
                a = copy(b)
                b += step
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

function full_info_soln(wL, wH, wp0, cpts0,
    print_every = 50)

    println("Solving full-info problem for w∈[$wL, $wH]")

    # Build the space and approximation
    C0 = Fun(S0, ApproxFun.transform(S0, cpts0))

    # Calculate constant terms in C 
    tterm = gauss_leg(x -> (x ^ (ϵ - 1.)) * tnorm_pdf(x), 50, θ_min, θ_max)
    cons1 = ((ϵ - 1.) / (ϵ * R)) ^ ϵ * tterm 
    cons2 = (1. / R) * ((ϵ - 1.) / (ϵ * R)) ^ (ϵ - 1.) * tterm 

    function get_wp(w, Cf; w0 = w)

        function fr(wp)
            if wp > wH
                Cp_wp = Cf'(wH)
            elseif wp < wL
                Cp_wp = Cf'(wL)
            else
                Cp_wp = Cf'(wp)
            end
    
            return Cp_wp - R * β * exp(w - β * wp)
        end
    
        bkt0 = (floor(w0 - 0.01, digits = 1), ceil(w0 + 0.01, digits = 1))
        stp = round(bkt0[2] - bkt0[1], digits = 5)
        bkt = find_bracket(fr, bkt0 = bkt0, step = stp) 
        wpstar = find_zero(fr, bkt,
            atol = 1e-6, rtol = 1e-6,
            xatol = 1e-6, xrtol = 1e-6) 
    
        return wpstar 

        
    end

    function iterate_c_full(cpts0;
        ω = 1.)
    
        diff = 10
        tol = 1e-6
        its = 0
        mxit = 5000
        cpts1 = copy(cpts0)
        wpvals = copy(wp0)
        cwp = zeros(m_cheb)  
    
        while diff > tol && its < mxit
            C0 = Fun(S0, ApproxFun.transform(S0, cpts0))
            
            for i in 1:m_cheb
    
                # Get w' values 
                get_wp_curr(w) = get_wp(w, C0, w0 = wpvals[i])
                wpvals[i] = get_wp_curr(wp0[i])
    
                # Get C(w') values, interpolating where needed 
                wp = wpvals[i]
                if wp > wH
                    cwp[i] = C0(wH) + C0'(wH) * (wp - wH)
                elseif wp < wL 
                    cwp[i] = C0(wL) + C0'(wL) * (wp - wL)
                else
                    cwp[i] = C0(wp)
                end
    
            end
    
            cpts1 = exp.(wp0 - β * wpvals) + cwp / R .+ cons1 .- cons2
    
            diff = norm(cpts0 - cpts1, Inf)
            rd = round(diff, digits = 5)
            rdwp = round.(wpvals, digits = 2)
            if mod(its, print_every) == 0
                println("iteration $its: diff = $rd, w' = $rdwp")
            end
            
            its += 1
            cpts0 = ω * cpts1 + (1. - ω) * cpts0
        end
        return cpts1, wpvals
    end
    
    cpts, wps = iterate_c_full(cpts0)
    gamcs = exp.(wp0 - β * wps)         # these are the c values, which equal γ

    # Plot and Return
    wgrid = range(wL, wH, length = 100)
    Cfull = Fun(S0, ApproxFun.transform(S0, cpts))

    pC = plot(wgrid, Cfull.(wgrid),
        title = "Full Info Solution",
        label = "C(w)",
        legend = :bottomright,
        xlabel = "w")

    pwp = plot(wp0, [wp0 wps],
        title = "Full-info w'",
        label = ["w'=w" "w'"],
        legend = :bottomright,
        xlabel = "w")

    display(plot(pC, pwp, layout = (1, 2)))

    return cpts, gamcs, wps

end

function foc_k(x, U, μ, θ)
        
    c, wp, k = x

    if wp > wH
        Cp_wp = C0'(wH)
    elseif wp < wL
        Cp_wp = C0'(wL)
    else
        Cp_wp = C0'(wp)
    end

    # if warnings
    #     if wp > wH || wp < wL
    #         println("Warning: w' out of bounds")
    #     end
    # end

    # FOC vector
    focs = zeros(eltype(x), 3)
    focs[1] = 1. - Cp_wp / (β * R * c) - μ * k / (θ * c ^ 2)
    focs[2] = 1. - (ϵ - 1.) / (R * ϵ) * θ ^ (1. - 1. / ϵ) * k ^ (-1. / ϵ) + 
        μ / (θ * c)
    focs[3] = log(c) + β * wp - U 
    
    return focs
    
end

function newton_k(U, μ, θ, i)
    x0 = [c0_lf[i] wp_lf[i] klf[i]]'
    mxit = 500
    tol = 1e-10
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
    ~, ft, fpt = fdist(t)

    # if warnings 
    #     if wp > wH || wp < wL
    #         println("Warning: w' out of bounds")
    #     end
    # end

    if wp > wH
        Cp_wp = C0'(wH)
    elseif wp < wL
        Cp_wp = C0'(wL)
    else
        Cp_wp = C0'(wp)
    end


    du[1] = k / (c0 * t)
    du[2] = γ - Cp_wp / (β * R) - fpt / ft * μ

end

function tax_shoot(U_0)
    u0 = [U_0 0.0]
    tspan = (θ_min, θ_max)
    prob = ODEProblem(de_system!, u0, tspan)
    gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 
    return gpath.u[end][2] # want to get μ(θ̲) = 0
    
end

# Promise-keeping (U* = 0)
function pkc_integrand(t, gpath)
    U, mu = gpath(t)
    Ft, ft, fpt = fdist(t)
    return U * ft
end

# Optimal allocations along the path
function opt_allocs(gpath)
    cp = zeros(nt)
    kp = similar(cp)
    wp = similar(cp)
    mup = similar(cp)
    
    for i in 1:nt
        U, mu = gpath(tgrid[i])
        cp[i], wp[i], kp[i] = alloc_single(U, mu, tgrid[i])
        mup[i] = mu

    end

    return cp, wp, kp, mup
end

# Value of C1
function C_integrand(t, gpath)
    
    U, mu = gpath(t)
    Ft, ft, fpt = fdist(t)
    ct, wpt, kt = alloc_single(U, mu, t)
    
    return ft * (ct + kt + 1 / R * (C0(wp) - (t * kt) ^ (1. - 1. / ϵ)) )

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

# State space w 
wL = -6.
wH = 1.

m_cheb = 5
S0 = Chebyshev(wL..wH)
wp0 = points(S0, m_cheb)

# Testing 
cpts0 = (wp0 .- wp0[end] .+ 0.5) .^ 1.5 

# cpts0 = vec(readdlm(solnpath * "cpts_hia.txt"))
# cpts0 = vec(readdlm(solnpath * "cpts2_hia.txt"))

cpts_full, gamcs_full, wps_full = full_info_soln(wL, wH, wp0, cpts0);

C0 = Fun(S0, ApproxFun.transform(S0, cpts_full)) # start from full-info solution

# gams0 = vec(readdlm(solnpath * "gs.txt"))
# U0s = vec(readdlm(solnpath * "U0s.txt"))

# Starting guess: full-info soln 
# ip = 1 # γ*∈[1.18, 1.19] bkt = (0.59, 0.6) # γ in FI is in this range (!!)
# ip = 2 # γ*∈[1., 1.01] bkt = (-1.0, -0.99) # ditto 
# ip = 3 # γ*∈[0.77, 0.78] bkt = (-3.77, -3.76)
# ip = 4 # γ*∈[0.59, 0.6] bkt = (-6.5, -6.49)
# ip = 5 # γ∈[0.5, 0.51] bkt = (-8.27, -8.26) # a good starting guess is bkt around wp0[i] - 0.5

ip = 5
wstar = wp0[ip]

γ = 0.62
println("\nγ = $γ")

tax_shoot(-7.)

# test_ums = range(-8., -6., step = 0.1) 
# testvals = [test_shoot(test_ums[i]) for i in 1:length(test_ums)]
# display(plot(test_ums, testvals, size = (1000, 1000)))

bkt = (-6.27, -6.26)
stp = round(bkt[2] - bkt[1], digits = 5)
bkt = find_bracket(um -> tax_shoot(um), bkt0 = bkt, step = stp)
bkt = round.(bkt, digits = 5)
println("Bracket found, bkt = $bkt")
Umin_opt = find_zero(x -> tax_shoot(x), bkt)

u0 = [Umin_opt 0.0]
tspan = (θ_min, θ_max)
prob = ODEProblem(de_system!, u0, tspan)
gpath = solve(prob, AutoTsit5(Rosenbrock23()), alg_hints = [:stiff]) 

# Promise-keeping (U* = 0)
function pkc_integrand(t, gpath)
    U, mu = gpath(t)
    Ft, ft, fpt = fdist(t)
    return U * ft
end

pkc = gauss_leg(t -> pkc_integrand(t, gpath), 50, θ_min, θ_max)
println("∫UdF = $pkc")
println("w* = $wstar")

ct, wt, kt, mut = opt_allocs(gpath) 
display(plot(tgrid, [ct wt kt],
    label = ["c" "w'" "k"]))
