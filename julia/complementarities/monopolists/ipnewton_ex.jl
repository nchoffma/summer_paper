using Optim

fun(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function fun_grad!(g, x)
    g[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    g[2] = 200.0 * (x[2] - x[1]^2)
end

function fun_hess!(h, x)
    h[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    h[1, 2] = -400.0 * x[1]
    h[2, 1] = -400.0 * x[1]
    h[2, 2] = 200.0
end;

function con2_c!(c, x)
    c[1] = x[1]^2 + x[2]^2     ## First constraint
    c[2] = x[2]*sin(x[1])-x[1] ## Second constraint
    c
end
function con2_jacobian!(J, x)
    # First constraint
    J[1,1] = 2*x[1]
    J[1,2] = 2*x[2]
    # Second constraint
    J[2,1] = x[2]*cos(x[1])-1.0
    J[2,2] = sin(x[1])
    J
end
function con2_h!(h, x, λ)
    # First constraint
    h[1,1] += λ[1]*2
    h[2,2] += λ[1]*2
    # Second constraint
    h[1,1] += λ[2]*x[2]*-sin(x[1])
    h[1,2] += λ[2]*cos(x[1])
    # Symmetrize h
    h[2,1]  = h[1,2]
    h
end;

lx = Float64[]; ux = Float64[]
x0 = [0.25, 0.25]
lc = [-Inf, 0.0]; uc = [0.5^2, 0.0]
dfc = TwiceDifferentiableConstraints(con2_c!, con2_jacobian!, con2_h!,
                                     lx, ux, lc, uc)

df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)

res = optimize(df, dfc, x0, IPNewton())
