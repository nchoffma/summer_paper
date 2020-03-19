
# Summary and Setup -------------------------------------------------------

# Replicates model from Genicot and Ray (2017)

library(purrr)

# Setup and Params --------------------------------------------------------

# Payoff params
del <- 0.8
sig <- 0.8
pi <- 1.0

# Various production functions
big_a <- 4
rho <- 2.1
bet <- 0.55

# Functions ---------------------------------------------------------------

# Payoff functions
util <- function(c) {c ^ (1 - sig)}
w0 <- function(z) {del * z ^ (1 - sig)}
w1 <- function(e) {del * pi * e ^ (1 - sig)}

# Production functions
prod <- function(z) rho * z
kfunc <- function(z) z / rho # inverse of prod()

# Total Payoff
total_payoff <- function(y, z, a) {
  util(y - kfunc(z)) + w0(z) + w1(max(c(z - a, 0)))
}

# Derivatives
util_p <- function(c) {(1 - sig) * c ^ -sig}
w0_p <- function(z) {del * (1 - sig) * z ^ -sig}
w1_p <- function(e) {del * pi * (1 - sig) * e ^ -sig}

# Optimal z
zopt0 <- function(z, y) { # z < a
  w0_p(z) - (util_p(y - kfunc(z)) / rho)
}

zopt1 <- function(z, y, a) { # z > a
  w0_p(z) + w1_p(z - a) - (util_p(y - kfunc(z)) / rho)
}

# Find the interval to search for optimal z (fairly naiive)
find_interv <- function(y, a) {
  zvals = seq(0.0, rho * y, by = 0.01)
  if (missing(a)) {
    z_cands = zopt0(zvals, y)
    z_cands = zvals[z_cands < Inf & z_cands > -Inf & !is.na(z_cands)]
    interv = range(z_cands)
  } else {
    z_cands_a = zopt1(zvals, y, a)
    z_cands_a = zvals[z_cands_a < Inf & z_cands_a > -Inf & !is.na(z_cands_a)]
    interv = range(z_cands_a)
  }
  if (max(interv) == Inf | min(interv) == -Inf) {
    cat(c(y, a, "\n"))
    stop("No candidate z values found")
  } else {
    return(interv) 
  }
}

# Find optimal z
find_z <- function(y, a) {
  # Find the roots
  interv0 = find_interv(y)
  interv1 = find_interv(y, a)
  z0 = uniroot(zopt0, interval = interv0, y = y, extendInt = "yes")$root
  z1 = uniroot(zopt1, interval = interv1, y = y, a = a, extendInt = "yes")$root

  # Which is better?
  u0 = util(y - kfunc(z0)) + w0(z0)
  u1 = util(y - kfunc(z1)) + w0(z0) + w1(max(c(z0 - a, 0)))

  if (u0 > u1) {
    return(z0)
  } else {
    return(z1)
  }
}

# Find optimal g
choose_g <- function(y, a) {
  z_opt = find_z(y, a)
  g_opt = z_opt / y
}


# Testing -----------------------------------------------------------------

yvals <- seq(0.8, 4.0, by = 0.01) # uniform dist
avg_inc <- mean(yvals)
a_y <- 0.5 * (yvals + avg_inc)

opt_g <- map2_dbl(yvals, a_y, choose_g)
plot(yvals, opt_g, "l")

# WORKS as long as support not too wide...
# Can play with inequality of initial dist by adding/subtracting pts


# Get next dist -----------------------------------------------------------

yvals <- seq(0.8, 4.0, by = 0.01) # uniform dist

# Find steady state
dif <- 10.0
tol <- 1e-3
its <- 0

while (dif > tol) {
  avg_inc <- mean(yvals)
  a_y <- 0.5 * (yvals + avg_inc)

  yvals_next <- map2_dbl(yvals, a_y, find_z)
  dif <- max(abs(yvals - yvals_next))
  its <- its + 1
  yvals <- yvals_next
}

# TODO: fix this
y <- yvals[1]
a <- a_y[1]
zvals = seq(0.01, rho * y * 10, length.out = 300)
z_cands_a = zopt1(zvals, y, a)
z_cands_a = zvals[z_cands_a < Inf & z_cands_a > -Inf & !is.na(z_cands_a)]
interv = range(z_cands_a)
