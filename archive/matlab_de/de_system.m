function du = de_system(u, t, pars, Um)
%de_system - System of ODEs
%
% Syntax: du = de_system()
%
% Calculates dU and dmu, as input to Matlab DE solver
% Note: de_pars is distinct from pars, as it includes U(theta_min)

% Get the allocations
U = u(1);
mu = u(2);
[c0, k, c1y, c1u, phi] = alloc_single(U, mu, t, Um, pars);

% Differential equations
du = zeros(2, 1);
du(1) = k / (c0 * t);
beta = pars(2); 
tmin = pars(end - 1);
lam1 = pars(6)
if t == tmin 
    du(2) = lam1 /  beta * c1y - 1 - phi;
else
    du(2) = lam1 /  beta * c1y - 1;
end

end