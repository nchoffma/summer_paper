function x = parloginv(p,alpha,nu,tau)

%------------------------------------------------------------------------
%   Finds inverse of a pareto-lognormal for the vector p
%   p must be a column vector
%------------------------------------------------------------------------
%   Roozbeh Hosseini 8/14/2015
%------------------------------------------------------------------------

if min(p)<0 || max(p)>1
    disp('p must be between 0 and 1')
    return
end

x = alpha/(alpha-1)*exp(nu+tau^2/2)*ones(size(p));

while (1)
    [P,D] =  plogncdf(x,alpha,nu,tau);
    f  = P-p;
    df = D;
    if max(abs(f))<1.e-8
        break
    end
    d = f./df;
    while min(x-d)<0
        d=d/2;
    end
    x = x - d;
end

end

    