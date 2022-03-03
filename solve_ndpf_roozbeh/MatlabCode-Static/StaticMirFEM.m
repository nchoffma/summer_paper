% StaticMirFEM.m
%------------------------------------------------------------------------
%   This code solves a static mirrlees optimal tax problem
%   Using Finite Element Method
%------------------------------------------------------------------------
%   Roozbeh Hosseini
%   5/22/2017
%------------------------------------------------------------------------
clear

%------------------------------------------------------------------------
%   Preference Parameters
%------------------------------------------------------------------------
sig = 1.5;
gam = 3;
psi = .2726;
%------------------------------------------------------------------------

%------------------------------------------------------------------------
%   Type Distribution is ParetoLogNormal
%------------------------------------------------------------------------
apar  = 3;
muth  = -1/apar;
sigth = 0.5;

N     = 100;
lo    = parloginv(1e-2,apar,muth,sigth);
hi    = parloginv(1-1e-6,apar,muth,sigth);
theta = logspace(log10(lo),log10(hi),N);
F     = 1-1e-8-1e-2;
F0    = (1e-2)/F;
f     = plognpdf(theta,apar,muth,sigth)/F;

par = [sig gam psi apar muth sigth];
%------------------------------------------------------------------------

%------------------------------------------------------------------------
%   Laissez-faire
%   This is just C = \theta*L
%   I use laissez-faire allocation as inital guess to solve
%   Full Info allocation   
%------------------------------------------------------------------------
lLF = (theta.^(1-sig)/psi).^(1/(gam+sig-1));
yLF = theta.*lLF;
cLF = yLF;
wLF = cLF.^(1-sig)/(1-sig)-psi*lLF.^gam/gam;
%------------------------------------------------------------------------

%------------------------------------------------------------------------
%   Full Info
%   This is just the planning problem without the implementability
%   constraint.
%------------------------------------------------------------------------
Ath = 0;    
for i = 1:N-1
    th1 = theta(i);
    th2 = theta(i+1);
    [wq,thq] = qgausl(th1,th2,5);
    fq = plognpdf(thq,apar,muth,sigth)/F;
    Ath = Ath+(thq.^(gam/(gam-1)).*wq)*fq'/psi^(1/(gam-1));
end

lamFB = 1;
while(1)
    f0  = lamFB^(-1/sig)-Ath*lamFB^(1/(gam-1));
    df0 = (-1/sig)*lamFB^(-1/sig-1)-Ath*(1/(gam-1))*lamFB^(1/(gam-1)-1);
    if abs(f0)<1.0e-8
        break
    end
    d = f0/df0;
    if lamFB-d<0
        d = d/2;
    end
    lamFB = lamFB-d;
end
cFB = lamFB^(-1/sig)*ones(1,N);
lFB = (theta*lamFB/psi).^(1/(gam-1));
yFB = theta.*lFB;
wFB = cFB.^(1-sig)/(1-sig)-psi*lFB.^gam/gam;
%------------------------------------------------------------------------

%------------------------------------------------------------------------
%   FEM paramters
%------------------------------------------------------------------------
ne = N-1;
nq = 5;

%------------------------------------------------------------------------
%   Initial guess
%   Note: Since the distribution has pareto tail, we know that maginal tax
%   at the top converges to a non-zero value.
%   Therefore, I drop the boundary conditions. Imposing them is
%   straighforward, if needed.
%------------------------------------------------------------------------
lam = 1;
a0  = wLF';
b0  = -.1*ones(N,1);

b0(1) = 0;
b0(N) = 0;


Temp = eye(2*N,2*N);
BV = Temp;

while(1)

INTR  = zeros(2*N,1);
dINTR = zeros(2*N,2*N);

for n = 1:ne
%-----we are in element n----- 
    x1 = theta(n);
    x2 = theta(n+1);
%-----calculate quarature point (for Galerkin integration)-----     
    [wq,thq] = qgausl(x1,x2,nq);        
    for i = 1:nq
%-----calculate allocations and derivatives w.r.t uknowns-----         
        th  = thq(i);
        wth = wq(i);
        Le  = x2-x1;
        e   = 2*(th-x1)/(x2-x1)-1;
        bs1 = 0.5*(1-e);
        bs2 = 0.5*(1+e);
%-----find U,mu, U-dot and mu-dot-----         
        U  = a0(n)*bs1+a0(n+1)*bs2;
        mu = b0(n)*bs1+b0(n+1)*bs2;        
        U_dot  = (a0(n+1)-a0(n))/Le;
        mu_dot = (b0(n+1)-b0(n))/Le;
%-----Now we need to find c and l-------
        fth  = plognpdf(th,apar,muth,sigth)/F;
        dfth = Dplognpdf(th,apar,muth,sigth)/F; 
%-----Solve for c and l on the current node th (given U, mu and lam)----
        cl = [cLF(n);lLF(n)];   % initial guess
        while(1)
            [foc,dfoc] = StaticFOCnew(cl,U,mu,th,lam,par); 
            if max(abs(foc))<1.0e-8
                break
            end
            d = dfoc\foc;
            while min(cl-d)<0
                d = d/2;
            end
            cl = cl-d;
        end
        c = cl(1);
        l = cl(2);
%-----Find derivatives of c and l w.r.t to U and mu----        
        dU  = dfoc\[1;0];
        dmu = dfoc\[0;-psi*gam*l^(gam-1)/th/lam];
        dcdU = dU(1);
        dldU = dU(2);
        dcdmu = dmu(1);
        dldmu = dmu(2);
%-----Now we can evluate equations----  
        FU  = U_dot - psi*l^gam/th;
        Fmu = mu_dot - lam*c^sig + mu*dfth/fth + 1; 
    
        INTR(n)   = INTR(n)+bs1*wth*FU;
        INTR(n+1) = INTR(n+1)+bs2*wth*FU;
        INTR(N+n)   = INTR(N+n)+bs1*wth*Fmu;
        INTR(N+n+1) = INTR(N+n+1)+bs2*wth*Fmu;
%-----Now we need to do the derivatives----     
        dFU_da_n  = -1/Le-gam*psi*l^(gam-1)/th*dldU*bs1;
        dFU_db_n  = -gam*psi*l^(gam-1)/th*dldmu*bs1;
        dFmu_da_n = -sig*lam*c^(sig-1)*dcdU*bs1;
        dFmu_db_n = -1/Le-sig*lam*c^(sig-1)*dcdmu*bs1+bs1*dfth/fth;
        
        dFU_da_n1  = 1/Le-gam*psi*l^(gam-1)/th*dldU*bs2;
        dFU_db_n1  = -gam*psi*l^(gam-1)/th*dldmu*bs2;
        dFmu_da_n1 = -sig*lam*c^(sig-1)*dcdU*bs2;
        dFmu_db_n1 = 1/Le-sig*lam*c^(sig-1)*dcdmu*bs2+bs2*dfth/fth;
        
        dFU  = zeros(1,2*N);
        dFmu = zeros(1,2*N); 
        
        dFU(n)     = dFU_da_n;
        dFU(n+1)   = dFU_da_n1;
        dFU(N+n)   = dFU_db_n;
        dFU(N+n+1) = dFU_db_n1;

        dFmu(n)     = dFmu_da_n;
        dFmu(n+1)   = dFmu_da_n1;
        dFmu(N+n)   = dFmu_db_n;
        dFmu(N+n+1) = dFmu_db_n1;

        for ii = 1:2*N
            dINTR(n,ii)   = dINTR(n,ii)+wth*bs1*dFU(ii);
            dINTR(n+1,ii) = dINTR(n+1,ii)+wth*bs2*dFU(ii);
            dINTR(N+n,ii)   = dINTR(N+n,ii)+wth*bs1*dFmu(ii);
            dINTR(N+n+1,ii) = dINTR(N+n+1,ii)+wth*bs2*dFmu(ii);
        end  
    end    
end

INT_bv  = BV*INTR;
dINT_bv = BV*dINTR*BV'; 

norm = max(abs(INT_bv));
if norm<1.e-8
    break
end

d = dINT_bv\INT_bv;

if norm>.1  % use this to control the speed of updating when error is large
    A = 1;
else
    A = 1;
end

a0 = a0-d(1:N)/A;
b0 = b0-d(N+1:2*N)/A;
b0 = -abs(b0);  % makes sure mu is always negative


%-----Plot Progress of the Code----
subplot(2,2,1:2)
plot(INT_bv)
title('Errors')
subplot(2,2,3)
plot(theta,a0)
title('U(\theta)')
subplot(2,2,4)
plot(theta,b0)
title('\mu(\theta)')

disp(norm)
pause(.01) % This allows figures to plot and show up on screen
end

%------------------------------------------------------------------------
%   Solve for Efficient Allocations, given efficient U and mu
%------------------------------------------------------------------------
C = zeros(1,N);
L = zeros(1,N);

for i = 1:N
    U   = a0(i);
    mu  = b0(i);
    th  = theta(i);
    fth = f(i);
    
    cl = [cLF(n);lLF(n)];
    while(1)
        [foc,dfoc] = StaticFOCnew(cl,U,mu,th,lam,par); 
        if max(abs(foc))<1.0e-8
            break
        end
        d = dfoc\foc;
        while min(cl-d)<0
            d = d/2;
        end
        cl = cl-d;
    end
    C(i) = cl(1);
    L(i) = cl(2);
end
Y = theta.*L;

%------------------------------------------------------------------------
%   Solve for Optimal Marginal Taxes
%------------------------------------------------------------------------
tau_L = 1-psi*L.^(gam-1)./theta./C.^(-sig);

%------------------------------------------------------------------------
%   Plot Efficient Allocations and Optimal Tax function
%------------------------------------------------------------------------
figure
subplot(2,2,1)
plot(theta,C)
title('Consumpion')
xlabel('\theta')

subplot(2,2,2)
plot(theta,L)
title('Hours')
xlabel('\theta')

subplot(2,2,3:4)
plot(Y,tau_L)
title('Marginal Tax')
xlabel('Income (y=\theta l(\theta))')

set(gcf,'papersize',[5.0,4.0])
set(gcf,'paperposition',[0,0,5.0,4.0])
print('-dpdf','static.pdf')



