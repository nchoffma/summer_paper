function [foc,dfoc] = StaticFOCnew(cl,U,mu,theta,lam,par)
    
%------------------------------------------------------------------------
%   This evaluates FOC and its Jaccobian to solve for Static Mirreels
%   Problem
%   Inputs:
%   U - promised utility
%   mu - multuplier on IC onstraint
%   lam - multiplier on feasibility
%   cl = [c;l] - a guess for consumptiona and hours worked
%   The code evaluates promise keeping V(c,l) = U and FOC_wrt_L
%   This is used to solve c and l as function of (U,mu,lam) by newton
%   method.
%------------------------------------------------------------------------
%   Roozbeh Hosseini
%   5/22/2017
%------------------------------------------------------------------------
sig = par(1);
gam = par(2);
psi = par(3);

foc  = zeros(2,1);
dfoc = zeros(2,2);

c  = cl(1);
l  = cl(2);

foc(1) = c^(1-sig)/(1-sig)-psi*l^gam/gam-U;
foc(2) = theta-psi*l^(gam-1)*(c^sig-mu*gam/lam/theta);
    
dfoc(1,1) = c^(-sig);
dfoc(1,2) = -psi*l^(gam-1);
dfoc(2,1) = -sig*c^(sig-1)*psi*l^(gam-1);
dfoc(2,2) = -(gam-1)*psi*l^(gam-2)*(c^sig-mu*gam/lam/theta);

end