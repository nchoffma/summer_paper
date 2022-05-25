#= 

Studies the full-persistence case, to see what patterns emerge

This does not attempt to solve for the period-0 unknowns 
γ, μ, U, κ

Instead, it just assumes values for them and 
determines how c_t, k_t, and p_t evolve over time 

=#

using DifferentialEquations, Distributions, LinearAlgebra, Plots, NLsolve, 
    ForwardDiff, Optim, LaTeXStrings, Roots, 
    FastGaussQuadrature, Printf, DelimitedFiles, ApproxFun

gr()
println("******** full_persistence.jl ********")

# Fixed parameters 
β = 0.9
θ_min = 1.
θ_max = 2.
ϵ = 4.
R = 1.1

# Moveable parameters 
μ = 0.1
γ = 0.5
κ = 0.2

