function [zk, xk] = cheb_nodes(m, n, a, b)
%cheb_nodes - Calculates Chebyshev nodes
%
% Syntax: zk = cheb_nodes(m, n, a, b)
%
% Calculates the m cheb nodes for the Chebyshev polynomial
% of order m, on the interval (a,b)

zk = -cos(((2 * 1:m - 1) / 2m_cheb) * pi)
xk = (b - a) / 2.0 * (zk + 1.0) + a

end