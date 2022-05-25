#= 

Plots the results of the ϵ experiments

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, NLsolve, 
    ForwardDiff, Optim, LaTeXStrings, Roots, 
    FastGaussQuadrature, Printf, DelimitedFiles, ApproxFun, Colors

# Spaces 
θ_min = 1.
θ_max = 2.
nt = 100
tgrid = range(θ_min, θ_max, length = nt)
pL = 0.
pH = 0.8  
m_cheb = 5
S0 = Chebyshev(pL..pH)
p0 = points(S0, m_cheb)

solnpath = "julia/complementarities/results/normal/eps_experiments/figures/"

# Read in data for ϵ = 4.0
c_4 = readdlm(solnpath * "eps4/csol_c.txt")
k_4 = readdlm(solnpath * "eps4/ksol_c.txt")
wp_4 = readdlm(solnpath * "eps4/wsol_c.txt")
ror_4 = readdlm(solnpath * "eps4/RoRs.txt")
tk_4 = readdlm(solnpath * "eps4/tk.txt")
tb_4 = readdlm(solnpath * "eps4/tb.txt")

# Read in data for ϵ = 5.8
c_58 = readdlm(solnpath * "eps58/csol_c.txt")
k_58 = readdlm(solnpath * "eps58/ksol_c.txt")
wp_58 = readdlm(solnpath * "eps58/wsol_c.txt")
ror_58 = readdlm(solnpath * "eps58/RoRs.txt")
tk_58 = readdlm(solnpath * "eps58/tk.txt")
tb_58 = readdlm(solnpath * "eps58/tb.txt")

# Plots # 

# Consumption 
plot(tgrid, c_4, 
    title = L"c(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false, 
    linecolor = [:red :darkblue :darkgreen :purple :brown])
pc = plot!(tgrid, c_58, 
    title = L"c(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false,
    linecolor = [:red :darkblue :darkgreen :purple :brown],
    linestyle = :dash);
display(pc)
savefig(solnpath * "c_comp.png")

# Investment 
plot(tgrid, k_4, 
    title = L"k^\prime(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false, 
    linecolor = [:red :darkblue :darkgreen :purple :brown])
pk = plot!(tgrid, k_58, 
    title = L"k^\prime(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false,
    linecolor = [:red :darkblue :darkgreen :purple :brown],
    linestyle = :dash);
display(pk)
savefig(solnpath * "k_comp.png")

# Promise utility 
plot(tgrid, wp_4, 
    title = L"w^\prime(\theta,\bar{p})",
    xlab = L"\theta",
    linecolor = [:red :darkblue :darkgreen :purple :brown],
    legend = false);
pw = plot!(tgrid, wp_58, 
    title = L"w^\prime(\theta,\bar{p})",
    xlab = L"\theta",
    legend = false,
    linecolor = [:red :darkblue :darkgreen :purple :brown],
    linestyle = :dash);
display(pw)
savefig(solnpath * "wp_comp.png")

# Rates of return 
plot(tgrid, ror_4,
    title = L"\bar{p}\hat{p}(\theta)\theta",
    xlab = L"\theta",
    linecolor = [:red :darkblue :darkgreen :purple :brown],
    legend = false)
pR = plot!(tgrid, ror_58,
    title = L"\bar{p}\hat{p}(\theta)\theta",
    xlab = L"\theta",
    legend = false,
    linecolor = [:red :darkblue :darkgreen :purple :brown],
    linestyle = :dash)
display(pR)
savefig(solnpath * "RoR_comp.png")

# Wedges 
plot(tgrid, tb_4,
    title = L"\tau_b(\theta, \bar{p})",
    xlabel = L"\theta",
    linecolor = [:red :darkblue :darkgreen :purple :brown],
    legend = false);
p_τb = plot!(tgrid, tb_58,
    title = L"\tau_b(\theta, \bar{p})",
    xlabel = L"\theta",
    linecolor = [:red :darkblue :darkgreen :purple :brown],
    legend = false,
    linestyle = :dash);

plot(tgrid, tk_4,
    title = L"\tau_b(\theta, \bar{p})",
    xlabel = L"\theta",
    linecolor = [:red :darkblue :darkgreen :purple :brown],
    legend = false);
p_τk = plot!(tgrid, tk_58,
    title = L"\tau_b(\theta, \bar{p})",
    xlabel = L"\theta",
    linecolor = [:red :darkblue :darkgreen :purple :brown],
    legend = false,
    linestyle = :dash);
display(plot(p_τb, p_τk, layout = (1, 2) ) )
savefig(solnpath * "wedges_comp.png")

