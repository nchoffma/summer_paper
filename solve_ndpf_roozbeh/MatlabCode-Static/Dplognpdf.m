function df = Dplognpdf(X,alpha,nu,tau)


f = plognpdf(X,alpha,nu,tau);

df = -(alpha+1)*f./X+...
    exp(alpha*nu+alpha^2*tau^2/2)*alpha*X^(-alpha-1).*...
    normpdf((log(X)-nu-alpha*tau^2)/tau)/tau./X;