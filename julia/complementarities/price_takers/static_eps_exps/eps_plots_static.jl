#= 

Plotting in the static model: ϵ experiments

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, NLsolve, 
    ForwardDiff, Optim, LaTeXStrings, Roots, 
    FastGaussQuadrature, Printf, DelimitedFiles, ApproxFun, Colors

# Spaces 
θ_min = 1.
θ_max = 2.
nt = 100
tgrid = range(θ_min, θ_max, length = nt)

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

function fdist(x)
    
    # Truncated normal
    cdf = tnorm_cdf(x)
    pdf = tnorm_pdf(x)
    fpx = tnorm_fprime(x)
    
    return cdf, pdf, fpx
    
end

tcdf(t) = fdist(t)[1]

solnpath = "julia/complementarities/static_eps_exps/"

# Read in ϵ = 4 case
epspath = "e4/"
k_4 = readdlm(solnpath * epspath * "ksol.txt")
ror_4 = readdlm(solnpath * epspath * "rors.txt")
wedge4 = readdlm(solnpath * epspath * "wedges.txt")
cdfR_4 = readdlm(solnpath * epspath * "ror_cdf.txt")
tk_4 = wedge4[:, 1]
tb_4 = wedge4[:, 2]

# Read in ϵ = 6 case 
epspath = "e6/"
k_6 = readdlm(solnpath * epspath * "ksol.txt")
ror_6 = readdlm(solnpath * epspath * "rors.txt")
wedge6 = readdlm(solnpath * epspath * "wedges.txt")
cdfR_6 = readdlm(solnpath * epspath * "ror_cdf.txt")
tk_6 = wedge6[:, 1]
tb_6 = wedge6[:, 2]

# Plotting 

# Investment 
pk = plot(tgrid, [k_4 k_6], 
    title = L"k^\prime(\theta,\bar{p})",
    xlab = L"\theta",
    label = [L"\epsilon = 4" L"\epsilon = 6"],
    legend = :topleft,
    linestyle = [:solid :dash], 
    linecolor = :darkred);
# display(pk)
# savefig(solnpath * "k_comp.png")


# Rates of return 
pR = plot(tgrid, [ror_4 ror_6],
    title = L"\bar{p}\hat{p}(\theta)\theta",
    xlab = L"\theta",
    linestyle = [:solid :dash], 
    linecolor = :darkred,
    legend = false)
# display(pR)
# savefig(solnpath * "RoR_comp.png")

p_τb = plot(tgrid, [tb_4 tb_6],
    title = L"\tau_b(\theta, \bar{p})",
    xlabel = L"\theta",
    linestyle = [:solid :dash], 
    linecolor = :darkred,
    legend = false);

p_τk = plot(tgrid, [tk_4 tk_6],
    title = L"\tau_k(\theta, \bar{p})",
    xlabel = L"\theta",
    linestyle = [:solid :dash], 
    linecolor = :darkred,
    legend = false);

display(plot(pk, pR, p_τb, p_τk, layout = (2, 2) ) )
savefig(solnpath * "static_comps.png")

# RoR CDFs 
plot(tgrid, [tcdf.(tgrid) cdfR_4 cdfR_6],
    title = "Rate of Return: CDF",
    label = [L"F(\theta)" L"\epsilon = 4" L"\epsilon = 6"],
    linecolor = [:darkblue :darkred :darkred], 
    linestyle = [:solid :solid :dash],
    legend = :topleft)
savefig(solnpath * "static_cdf_comps.png")