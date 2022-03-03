
% Using Matlab's differential equations solver to solve the system 

%% Setup

% Parameters
alpha = 0.5;
beta = 0.95;
w = 1.2;
tmin = 1.0;
tmax = 4.0;

% Interest rate
R = 1.1;
lam1 = 1.9;
lam0 = R * lam1;
pars = [alpha beta w R lam0 lam1 tmin tmax];

x0 = 1.0;                                                   % can be an interval
tspan = [tmin tmax];
shoot = @(Um) tax_shoot(Um, tspan);
Umin_opt = fzero(shoot, x0);                           % Shooting solution

u0 = [Umin_opt 0.0];                                        % starting point
[t, u] = ode23s(@(t, u) de_system(t, u, Umin), tspan, u0);  % solve the problem

% Optimal allocations
c0 = zeros(length(t), 1);
k = c0;
c1y = c0;
c1u = c0;
phi = c0;

for i = 1:length(t)
    [c0(i), k(i), c1y(i), c1u(i), phi(i)] = alloc_single(u[i, 1], u[i, 2], t[i], Um, pars);
end

% Plot