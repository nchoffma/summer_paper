function [c0, k, c1y, c1u, phi] = alloc_single(U, mu, t, Um, pars)
%alloc_single - Find allocations given a single point (U, mu)
%
% Syntax: [c0, k, c1y, c1u, phi] = alloc_single(U, mu, t, UM)
%
% Finds allocations at a single point
    
    [xp, fp] = foc_newton_kp(U, mu, t, Um, pars);
    if fp == 1
        x0 = foc_k0_cf(U);
        c0 = x0(1);
        c1u = x0(2);
        c1y = c1u;
        k = 0.0;
        phi = 0.0;
    else
        c0 = xp(1);
        k = xp(2);
        c1y = xp(3);
        c1u = xp(4);
        phi = xp(5);
    end
    
end

% Sub-functions supporting this one

function [focs, dfocs] = foc_kp(x, U, mu, t, Um, pars)
    
    % Parameters
    alpha = pars(1);
    beta = pars(2);
    R = pars(4);
    lam1 = pars(6);

    % Unpack guess
    c0 = x(1);
    k = x(2);
    c1y = x(3);
    c1u = x(4);
    phi = x(5);

    % First-order conditions (all = 0)
    focs = zeros(5, 1);
    focs(1) = c1y / (beta * c0) - mu * k / (lam1 * t * c0 ^ 2) - phi / (lam1 * (c0 + k)) - R;
    focs(2) = c1y / c1u - (beta * phi) / (lam1 * (1.0 - alpha) * c1u) - 1;
    focs(3) = alpha * t + mu / (lam1 * c0 * t) - phi / (lam1 * (c0 + k)) - R;
    focs(4) = U - log(c0) - beta * (alpha * log(c1y) + (1 - alpha) * log(c1u));
    focs(5) = Um - log(c0 + k) - beta * log(c1u);

    % Jacobian
    dfocs = zeros(5, 5);
    dfocs(1, 1) = -c1y / (beta * c0 ^ 2) + 2 * mu * k / (lam1 * t * c0 ^ 3) + ...
        phi / (lam1 * (c0 + k) ^ 2);
    dfocs(1, 2) = -mu / (lam1 * t * c0 ^ 2) + phi / (lam1 * (c0 + k) ^ 2);
    dfocs(1, 3) = 1 / (beta * c0);
    dfocs(1, 5) = -1 / (lam1 * (c0 + k));

    dfocs(2, 3) = 1;
    dfocs(2, 4) = -1;
    dfocs(2, 5) = -beta / (lam1 * (1 - alpha));

    dfocs(3, 1) = mu / (lam1 * t * c0 ^ 2) + phi / (lam1 * (c0 + k) ^ 2);
    dfocs(3, 2) = phi / (lam1 * (c0 + k) ^ 2);
    dfocs(3, 5) = -1 / (lam1 * (c0 + k));

    dfocs(4, 1) = 1 / c0;
    dfocs(4, 3) = alpha * beta / c1y;
    dfocs(4, 4) = (1 - alpha) * beta / c1u;

    dfocs(5, 1) = 1 / (c0 + k);
    dfocs(5, 2) = 1 / (c0 + k);
    dfocs(5, 4) = beta / c1u;

end

function [x0, fail] = foc_newton_kp(U, mu, t, Um, pars)
    %foc_newton_kp - Allocations for investors
    
    x0 = [0.6 0.6 0.6 * t 0.1 0.5]';
    mxit = 1000;
    tol = 1e-6;
    diff = 10.0;
    its = 0;
    fail = 0;
    
    while diff > tol & its < mxit
        [focs, dfocs] = foc_kp(x0, U, mu, t, Um, pars);
        diff = max(abs(focs));
        d = dfoc \ focs;
        while min(x0 - d) < 0.0
            d = d / 2.0;
            if maximum(d) < tol
                fail = 1;
                break
            end
        end
        if fail == 1
            break
        end
        x0 = x0 - d;
        its = its + 1;
        
    end
    
end

function x0 = foc_k0_cf(U)
% Finds allocations for lenders in closed form
    c0 = (exp(U) / (R * beta) ^ beta) ^ (1.0 / (1.0 + beta))
    c1 = R * beta * c0 
    x0 =  [c0 c1]'

end

