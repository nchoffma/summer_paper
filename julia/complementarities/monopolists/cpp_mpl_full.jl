#= 

Solves the planner's problem in the full-info case

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, NLsolve, 
    ForwardDiff, Optim, LaTeXStrings, Roots, 
    FastGaussQuadrature, Printf, DelimitedFiles, ApproxFun

gr()
println("******** cpp_inf_mpl.jl ********")

# Parameters
β = 0.9         # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

R = 1.1

m_cheb = 5

solnpath = "julia/complementarities/monopolists/results/"

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
    mxit = 100_000

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

# Building function to complete the above, given state space and starting function

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

#= State spaces that work: 

w ∈ [-6., 1.]
w ∈ [-7., 1.]
w ∈ [-7., 2.]
w ∈ [-8., 1.]
w ∈ [-9., 1.]
w ∈ [-9., 2.] # w' ≈ w 
w ∈ [-10., 4.]

=#

# State space w 
wL = -6.
wH = 1.

S0 = Chebyshev(wL..wH)
wp0 = points(S0, m_cheb)
cpts0 = (wp0 .- wp0[end] .+ 0.5) .^ 1.5

cpts_full, gamcs_full, wps_full = full_info_soln(wL, wH, wp0, cpts0);
