
# Bounded Pareto distribution
using Plots

function bdpar_cdf(x, α, L, H)
    cdf = (1 - L ^ α * x ^ (-α)) / 
        (1 - (L / H) ^ α)
end

function bdpar_pdf(x, α, L, H)
    pdf = (α * L ^ α * x ^ (-α - 1)) / 
        (1 - (L / H) ^ α)
end

function bdpar_fp(x, α, L, H)
    fpx = ((-α - 1) * α * L ^ α * x ^ (-α - 2)) / 
        (1 - (L / H) ^ α)
end

# Plotting 
L = 0.3
H = 4.5

x = range(L, H, length = 1000)

plot(x, bdpar_pdf.(x, 3.0, L, H))
plot!(x, bdpar_pdf.(x, 2.0, L, H))
plot!(x, bdpar_pdf.(x, 1.5, L, H))

1.0 - bdpar_cdf(4.0, 3.0, L, H)