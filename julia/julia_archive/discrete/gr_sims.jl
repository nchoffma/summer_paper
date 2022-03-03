using LinearAlgebra, Statistics, Compat, Roots

# Parameters
del = 0.8
sig = 0.8
pi = 1.0

big_a = 4
rho = 1.6
bet = 0.55
theta = 1.0

## Functions

# Payoffs
util(c) = c ^ (1 - sig)
w0(z) = del * z ^ (1 - sig)
w1(e) = del * pi * e ^ (1 - sig)

prod(k) = theta * (big_a / bet) * (k ^ bet)
kfunc(z) = (bet * z / (theta * big_a)) ^ (1 / bet) # inverse of prod()

function total_payoff(y, z, a)
    util(y - kfunc(z)) + w0(z) + w1(max[z - a, 0.0])
end

# Derivatives
util_p(c) = (1 - sig) * c ^ -sig
w0_p(z) = del * (1 - sig) * z ^ -sig
w1_p(e) = del * pi * (1 - sig) * e ^ -sig
f_p(k) = theta * big_a * k ^ (bet - 1)

# Optimal z
function zopt0(z, y) # z < a 
    w0_p(z) - (util_p(y - kfunc(z)) / f_p(kfunc(z)))
end

function zopt1(z, y, a) # z > a 
    w0_p(z) + w1_p(z - a) - (util_p(y - kfunc(z)) / f_p(kfunc(z)))
end

function find_interv(y, a; both = true)
    zvals0 = range(0, prod(y), length = 600)
    z_cands0 = zopt0.(zvals0, y)
    z_cands0 = zvals0[z_cands0 < Inf & z_cands0 > -Inf & !isnan(z_cands0)]
    interv0 = [min(z_cands0), max(z_cands0)]

    if min(interv0) == max(interv0)
        interv0 = interv0 * [0.999, 1.001] # if one point, extend around there
    end

    if both
        zvals1 = range(a, prod(y), length = 600)
        z_cands1 = zopt1.(zvals0, y, a)
        z_cands1 = zvals1[z_cands1 < Inf & z_cands1 > -Inf & !isnan(z_cands1)]
        interv1 = [min(z_cands1), max(z_cands1)]

        if min(interv1) == max(interv1)
            interv1 = interv1 * [0.999, 1.001] # if one point, extend around there
        end

        return interv0, interv1
        
    else
        return interv0
    end 
end

function find_z(y, a)
    if a >= prod(y) # aspirations frustrated regardless
        interv0 = find_interv(y, a, both = false)
        z0 = fzero(zopt0, interv0)
        return z0
    else # have to check both
        intervs = find_interv(y, a)
        
    end
end

# function test_range(u, l; both = true)
#     interv0 = [u l]
#     if both 
#         interv1 = [u - 2, l + 3]
#         return interv0, interv1
#     else
#         return interv0
#     end
# end

# r1 = test_range(2, 3) # can access first range with r1[1]
# r2 = test_range(2, 3, both = false)

# f(x) = sin(4 * (x - 1/4)) + x + x^20 - 1
# iv = [0, 1]
# find_zero(f, iv) # this works