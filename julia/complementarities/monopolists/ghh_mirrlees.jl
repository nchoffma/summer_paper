#= 

Solves the standard Mirrlees problem, with GHH constraints and κ

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, NLsolve, 
    ForwardDiff, Optim, LaTeXStrings, Roots, 
    FastGaussQuadrature, Printf, DelimitedFiles, ApproxFun

gr()
println("******** ghh_mirrlees.jl ********")

# Parameters
β = 0.9         # discounting
θ_min = 1.
θ_max = 2.
w = 1.
ϵ = 4.

R = 1.1

κ = 1.

thl = 1.
thh = 2. 

solnpath = "julia/complementarities/monopolists/results/"
figpath = solnpath * "figures/"

# Step one: solve unconstrained problem 

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

    println("\nSolving full-info problem for w∈[$wL, $wH], κ = $κ\n")

    # Build the space and approximation
    C0 = Fun(S0, ApproxFun.transform(S0, cpts0))

    # Calculate constant term in C 
    cons = (1. / 4.) * (κ / R) ^ 2 * (thl ^ 2 + thh ^ 2)

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
    
            cpts1 = exp.(wp0 - β * wpvals) + cwp / R .+ cons .- 1. / (R ^ 2)
    
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
    gamcs = exp.(wp0 - β * wps)

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

# State space w 
wL = -5.
wH = 5.

m_cheb = 5
S0 = Chebyshev(wL..wH)
wp0 = points(S0, m_cheb)

cpts0 = (wp0 .- wp0[end] .+ 0.5) .^ 1.5 

cpts_full, gamcs_full, wps_full = full_info_soln(wL, wH, wp0, cpts0)

# Full-info c and l, as starting guesses 
tgrid = [thl, thh]
cfull = [exp(wp0[i] - β * wps_full[i]) + 0.5(tgrid[j] * κ / R) ^ 2 for i in 1:m_cheb, j in 1:2]
lfull = repeat(κ / R * tgrid', m_cheb, 1)
wpfull = repeat(wps_full, 1, 2)

# Step two: solve the private-info problem 
iw = 1 
w = wp0[iw]
C0 = Fun(S0, ApproxFun.transform(S0, cpts_full))

# Objective 
function c_obj(x)
    # Get the value of C(w), given guess 
    # x = [cl, ch, ll, lh, wpl, wph]
    c = x[1:2]
    l = x[3:4]
    wp = x[5:6]

    return 0.5 * sum(c + (1. / R) * (C0.(wp) - κ .* tgrid .* l))
end

# Gradient and Hessian 
function c_obj_grad!(g, x)
    g[1] = 0.5
    g[2] = 0.5
    g[3] = -0.5 / R * κ * tgrid[1]
    g[4] = -0.5 / R * κ * tgrid[2]
    g[5] = 0.5 / R * C0'(x[5])
    g[6] = 0.5 / R * C0'(x[6])
    g
end

function c_obj_hess!(h, x)
    # h[:, :] .= 0.
    h[5, 5] = 0.5 / R * C0''(x[5])
    h[6, 6] = 0.5 / R * C0''(x[6])
    h
end

# Constraint function
function c_cons!(cx, x)
    c = x[1:2]
    l = x[3:4]
    wp = x[5:6]

    cx[1] = 0.5 * sum(log.(c - 0.5 * l.^2) + β * wp) - w 
    cx[2] = log(c[2] - 0.5 * l[2] ^ 2) + β * wp[2] - 
        log(max(c[1] - 0.5 * (tgrid[1] * l[1] / tgrid[2]) ^ 2, 1e-15)) - β * wp[1]
    cx[3] = log(c[1] - 0.5 * l[1] ^ 2) + β * wp[1] - 
        log(max(c[2] - 0.5 * (tgrid[2] * l[2] / tgrid[1]) ^ 2, 1e-15)) - β * wp[2]
    cx
end

# Test it out 
x = [cfull[iw,:]; lfull[iw,:]; wpfull[iw,:]]
c = x[1:2]
l = x[3:4]
wp = x[5:6]

# c_obj(x)
c_cons!(zeros(3), x)

# ograd_a = c_obj_grad!(zeros(6), x)
# ohess_a = c_obj_hess!(zeros(6, 6), x) # objective is good 

# Compute the derivative matrices using ForwardDiff

c_con_j! = (j, x) -> ForwardDiff.jacobian!(j, y -> c_cons!(zeros(eltype(y), 3), y), x)
# cjac_fd = c_con_j!(zeros(3, 6), x) # this works 

function c_con_h!(h, x, λ)
    for i in 1:3
        h += λ[i] * ForwardDiff.hessian(y -> c_cons!(zeros(eltype(y), 3), y)[i], x)
    end
    h
end

# λ = ones(3)
# chess_fd = c_con_h!(zeros(6, 6), x, λ) # works too

# Using full-info solution as initial guess, find initial point which satisfies the constraints 
function cons_viol(x)
    return min.(c_cons!(zeros(3), x), 0.)
end
cons_viol(x)

res = nlsolve(cons_viol, x, iterations = 10_000) 
x0 = res.zero
cons_viol(x0)

lx = [0., 0., 0., 0., -Inf, -Inf]
ux = fill(Inf, 6) 

lc = fill(0, 3)
uc = fill(Inf, 3)

df = TwiceDifferentiable(c_obj, c_obj_grad!, c_obj_hess!, x0)
dfc = TwiceDifferentiableConstraints(c_cons!, c_con_j!, c_con_h!, lx, ux, lc, uc)

res = optimize(df, dfc, x0, IPNewton(), 
    Optim.Options(iterations = 100,
                    show_trace = true,
                    show_every = 20)
)