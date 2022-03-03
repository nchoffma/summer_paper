
using LinearAlgebra, Interpolations, ApproxFun, Plots
# gr(fmt = :png)
gr()

# Problem 1: Polynomial approximation

# function to be approximated
f(x) = x ^ (1.0 / 3.0)

# A) Taylor series
function fhat_tay(x)
    # Evaluate via Horner's method
    a = [1, 1/3, -2/18, 10/(27 * 6), -80/(81 * 24)] # coeffs
    n = length(a) - 1
    sum = a[end]
    for i in n:-1:1
        sum = a[i] + sum * (x - 1.0)
    end
    return sum
    
end

# B) Chebyshev

# i) Naiive regression 
# Nodes
n_cheb = 4
m_cheb = 5
z(j) = -cos(((2j - 1) / 2m_cheb) * pi)
zk = [z(j) for j in 1:m_cheb]
xk = zk .+ 1.0 # Map into fxn support
yk = f.(xk) # Evaluate function

# Chebyshev Polynomials
Tz = zeros(n_cheb + 1, m_cheb)
Tz[1, :] .= 1.0
Tz[2, :] = zk
for i in 3:(n_cheb + 1)
    for j in 1:m_cheb
        Tz[i, j] = 2.0 * zk[j] * Tz[i - 1, j] - Tz[i - 2, j]
    end
end

# a coeffs
ai = Tz' \ yk # b/c m = n + 1, can solve directly 

# Approximate function
function fhat_i(x; n = 4)
    T = zeros(n + 1)
    T[1] = 1.0
    T[2] = x - 1.0
    for i in 3:(n + 1)
        T[i] = 2.0 * (x - 1.0) * T[i - 1] - T[i - 2]
    end
    return ai' * T
    
end

# ii) Fast Fourier
S = Chebyshev(0..2)
p = points(S, m_cheb)
v = f.(p)
fhat_ii = Fun(S, ApproxFun.transform(S,v))
# TODO: Figure out how to get coefficients from this

# iii) Barycentric approximation
λ = zeros(length(xk))
for i in 1:length(λ)
    λ[i] = 1.0 / cumprod(xk[i] .- xk[1:end .!= i])[end]
end
λ

function fhat_bary(x)
    num = 0.0
    den = 0.0
    for i in 1:length(xk)
        num += (λ[i] * yk[i]) / (x - xk[i])
        den += λ[i] / (x - xk[i])
    end
    return num / den
    
end

# C) Cubic spline on equally spaced points 
xkc = range(0, 2, length = 5)
ykc = f.(xkc)
fhat_spline = CubicSplineInterpolation(xkc, ykc)

# Plotting
xint = range(0, 2, length = 101)
y_true = f.(xint)
y_tay = fhat_tay.(xint)
y_cheb_reg = fhat_i.(xint)
y_cheb_fft = fhat_ii.(xint)
y_bary = fhat_bary.(xint)
y_spline = fhat_spline.(xint)

gr()
p_tay = plot(xint, [y_true y_tay],
    label = ["True Function" "Approx"],
    title = "Taylor Series",
    legend = :bottomright);
p_cheb = plot(xint, [y_true, y_cheb_reg, y_cheb_fft], 
    label = ["True Function" "Approx (Naive)" "Approx (FFT)"],
    title = "Chebyshev",
    legend = :bottomright); # FFT and regression are exactly the same
p_bary = plot(xint, [y_true, y_bary], 
    label = ["True Function" "Approx"],
    title = "Barycentric",
    legend = :bottomright);
p_spline = plot(xint, [y_true, y_spline], 
    label = ["True Function" "Approx"],
    title = "Cubic Spline",
    legend = :bottomright);
plot(p_tay, p_cheb, p_bary, p_spline, layout = (2, 2))
# savefig("assignments/writeups/assign3/q1_plot.png")

