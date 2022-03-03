function u_end = tax_shoot(Um, tspan)
%tax_shoot - Shooting function for tax problem
%
% Syntax: u_end = tax_shoot(Um, tspan)
%
% Takes in U(theta_min), and solves the differential equation to return mu(theta_max), which 
% should be zero.

u0 = [Um 0.0];
ode_fun = @(t, u) de_system(t, u, Um);
[t, u] = ode23s(ode_fun, tspan, u0);
u_end = u(end); 

end