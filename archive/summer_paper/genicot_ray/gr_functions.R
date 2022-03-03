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

find_interv <- function(y, a, both = T) {
  # Smarter way
  # both: do we evaluate optimal z on both sides of a (a < rho * y),
  # or not (a > rho * y)?
  
  # z0 (<a) have to find this in either case
  zvals0 = seq(0, a, length.out = 300)
  z_cands0 = zopt0(zvals0, y)
  z_cands0 = zvals0[z_cands0 < Inf & z_cands0 > -Inf & !is.na(z_cands0)]
  interv0 = range(z_cands0)
  if (min(interv0) == -Inf | max(interv0) == Inf) {
    cat(y, "\n")
    stop("No candidate z values found for zopt0")
  }
  
  if (both) {
    # z1 (>a)
    zvals1 = seq(a, rho * y, length.out = 300)
    z_cands1 = zopt1(zvals1, y, a)
    z_cands1 = zvals1[z_cands1 < Inf & z_cands1 > -Inf & !is.na(z_cands1)]
    interv1 = range(z_cands1)
    if (min(interv1) == -Inf | max(interv1) == Inf) {
      cat("y = ", y, "a = ", a, "\n")
      stop("No candidate z values found for zopt1")
    }
    
    return(list(interv0 = interv0, interv1 = interv1)) 
    
  } else {
    return(list(interv0 = interv0))
  }
}

find_z <- function(y, a) {
  # Check which z's are feasible
  if (a >= rho * y) { # aspirations will be frustrated regardless
    interv0 = find_interv(y, a, both = F)$interv0
    z0 = uniroot(zopt0, interval = interv0, y = y, extendInt = "yes")$root
    return(z0)
  } else { # have to check both options
    intervs = find_interv(y, a)
    z0 = uniroot(zopt0, interval = intervs$interv0, 
                 y = y, extendInt = "yes")$root
    z1 = uniroot(zopt1, interval = intervs$interv1, 
                 y = y, a = a, extendInt = "yes")$root
    
    # Which is better?
    u0 = util(y - kfunc(z0)) + w0(z0)
    u1 = util(y - kfunc(z1)) + w0(z1) + w1(max(c(z1 - a, 0)))
    
    if (u0 > u1) {
      return(z0)
    } else {
      return(z1)
    }
  }
}
