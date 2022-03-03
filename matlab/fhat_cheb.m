function fx = fhat_cheb(x, ai)
    
% Calculates the Chebyshev approximation of the function f(x)
% at the point x, given the coefficients ai (calcualated by cheb_approx)

z = -1.0 + 2.0 * (x - a) / (b - a)
T = zeros(n_cheb + 1)
T(1) = 1.0
T(2) = z
for i in 3:(n_cheb + 1)
    T(i) = 2.0 * z * T(i - 1) - T(i - 2)
end
fx = ai' * T

end