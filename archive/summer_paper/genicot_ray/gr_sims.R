
# Summary and Setup -------------------------------------------------------

# Replicates model from Genicot and Ray (2017)

library(tidyverse)
library(purrr)
library(ggplot2)
library(ggthemes)
theme_set(theme_tufte() + theme(
  axis.line = element_line(color = "black")
))
fig_path <- "figures/"

# Setup and Params --------------------------------------------------------

# Payoff params
del <- 0.8
sig <- 0.8
pi <- 1.0

# Various production functions
big_a <- 4
rho <- 1.6
bet <- 0.55
theta <- 1.0
cegm <- T # use CEGM?

# Functions ---------------------------------------------------------------

# Payoff functions
util <- function(c) {c ^ (1 - sig)}
w0 <- function(z) {del * z ^ (1 - sig)}
w1 <- function(e) {del * pi * e ^ (1 - sig)}

# Production functions
if (cegm) {
  prod <- function(k) {rho * k}
  kfunc <- function(z) { z / rho}
} else {
  prod <- function(k) theta * (big_a / bet) * (k ^ bet)
  kfunc <- function(z) (bet * z / (theta * big_a)) ^ (1 / bet) # inverse of prod()
}

# Total Payoff
total_payoff <- function(y, z, a) {
  util(y - kfunc(z)) + w0(z) + w1(max(c(z - a, 0)))
}

# Derivatives
util_p <- function(c) {(1 - sig) * c ^ -sig}
w0_p <- function(z) {del * (1 - sig) * z ^ -sig}
w1_p <- function(e) {del * pi * (1 - sig) * e ^ -sig}

if (cegm) {
  f_p <- function(k) {rho}
} else {
  f_p <- function(k) {theta * big_a * k ^ (bet - 1)}
}

# Optimal z
zopt0 <- function(z, y) { # z < a
  w0_p(z) - (util_p(y - kfunc(z)) / f_p(kfunc(z)))
}

zopt1 <- function(z, y, a) { # z > a
  w0_p(z) + w1_p(z - a) - (util_p(y - kfunc(z)) / f_p(kfunc(z)))
}

find_interv <- function(y, a, both = T) {
  # Smarter way
  # both: do we evaluate optimal z on both sides of a (a < f(y)),
  # or not (a > f(y))?
  
  # z0 (<a) have to find this in either case
  zvals0 = seq(0, prod(y), length.out = 600)
  z_cands0 = zopt0(zvals0, y)
  z_cands0 = zvals0[z_cands0 < Inf & z_cands0 > -Inf & !is.na(z_cands0)]
  interv0 = range(z_cands0)
  
  if (min(interv0) == -Inf | max(interv0) == Inf) {
    cat(y, "\n")
    stop("No candidate z values found for zopt0")
  }
  
  if (min(interv0) == max(interv0)) {
    interv0 = interv0 * c(0.999, 1.001) # if one point, extend around there
  }
  
  if (both) {
    # z1 (>a)
    zvals1 = seq(a, prod(y), length.out = 600)
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
  if (a >= prod(y)) { # aspirations will be frustrated regardless
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

# Payoff and Cost Plots ---------------------------------------------------

# Figure 1: Aspirations and Payoffs
nzpts <- 500
a <- 20
y <- 25
fig_data <- tibble(
  z = seq(0, 50, length.out = nzpts),
  payoff = w0(z) + w1(pmax(c(z - a), 0)),
  cost = util(y) - util(y - kfunc(z))
)

ggplot(fig_data, aes(z, payoff)) + 
  geom_line() + 
  geom_vline(xintercept = a, linetype = "dashed") + 
  coord_cartesian(xlim = c(0, 40)) + 
  labs(
    title = "Figure 1: Payoffs with Aspirations",
    y = "Payoff", 
    caption = paste0("y = ", y, "; a = ", a)
  ) + 
  annotate(
    "text", x = 10, y = 1.5, 
    label = expression(w[0](z))
  ) + 
  annotate(
    "text", x = 30, y = 2.5, 
    label = expression(w[0](z) + w[1](z - a))
  )

ggsave(paste0(fig_path, "fig1.jpg"))

ggplot(fig_data, aes(z, payoff)) + 
  geom_line() + 
  geom_vline(xintercept = a, linetype = "dashed") + 
  geom_line(aes(y = cost)) + 
  coord_cartesian(xlim = c(0, 40)) + 
  labs(
    title = "Figure 2: Payoffs and Costs",
    subtitle = "Problem is Concave on either side of a",
    y = element_blank(), 
    caption = paste0("y = ", y, "; a = ", a)
  ) + 
  annotate(
    "text", x = 10, y = 1.5, 
    label = expression(w[0](z))
  ) + 
  annotate(
    "text", x = 30, y = 2.5, 
    label = expression(w[0](z) + w[1](z - a))
  ) + 
  annotate(
    "text", x = 30, y = 0.55, 
    label = expression(u(y) - u(y - k(z)))
  )

ggsave(paste0(fig_path, "fig2.jpg"))


# Threshold y* ------------------------------------------------------------

y_init <- seq(5.0, 100.0, length.out = 100) # uniform dist
a_y <- 0.5 * (y_init + mean(y_init))
opt_z <- map2_dbl(y_init, a_y, find_z)
opt_g <- opt_z / y_init

fig4_data <- tibble(
  y = y_init,
  asp = a_y,
  opt_z = opt_z,
  opt_g = opt_g
)

fig4_data <- fig4_data %>% 
  mutate(
    g_f = if_else(opt_z < a, opt_g, NaN), # frustrated
    g_s = if_else(opt_z >= a, opt_g, NaN)
  )

ggplot(fig4_data, aes(x = y)) + 
  geom_line(aes(y = g_f)) + 
  geom_line(aes(y = g_s)) + 
  geom_vline(xintercept = 24.5, linetype = "dashed") + 
  labs(
    title = "Figure 4: Growth and Income threshold y*",
    y = "g"
  ) + 
  annotate(
    "text", x = 15, y = 0.78, 
    label = expression(underline(g))
  ) + 
  annotate(
    "text", x = 60, y = 1.4, 
    label = "g(r)"
  )
ggsave(paste0(fig_path, "fig4.jpg"))

# Example 1 ---------------------------------------------------------------

# Deterministic production
y_init <- seq(10.0, 100.0, length.out = 100) # uniform dist
yvals <- y_init

# Find steady state
dif <- 10.0
tol <- 1e-3
its <- 0
maxit <- 100

while (dif > tol & its < maxit) {
  avg_inc <- mean(yvals)
  a_y <- 0.5 * (yvals + avg_inc)

  yvals_next <- map2_dbl(yvals, a_y, find_z)
  
  dif <- max(abs(yvals - yvals_next))
  its <- its + 1
  yvals <- yvals_next
}

plot(y_init, yvals)
hist(yvals, breaks = length(y_init))
p_new <- mean(round(yvals, 2) == min(round(yvals, 2)))

# Figure 5: Varying p -----------------------------------------------------

top_vals <- seq(100, 120, by = 1)

find_ss <- function(top) {
  y_init = seq(10, top, length.out = 100) # uniform dist
  yvals = y_init
  
  # Find steady state
  dif = 10.0
  tol = 1e-3
  its = 0
  maxit = 100
  
  while (dif > tol & its < maxit) {
    avg_inc = mean(yvals)
    a_y = 0.5 * (yvals + avg_inc)
    
    yvals_next = map2_dbl(yvals, a_y, find_z)
    
    dif = max(abs(yvals - yvals_next))
    its = its + 1
    yvals = yvals_next
  }
  
  p_new = mean(round(yvals, 2) == min(round(yvals, 2)))
  return(c(
    y_h = max(yvals),
    y_l = min(yvals),
    p = p_new,
    its = its
  ))
}

fig5_data <- map(top_vals, find_ss)
fig5_data <- bind_rows(lapply(fig5_data, as.data.frame.list))

fig5_data <- fig5_data %>% 
  mutate(
    mean_inc = p * y_l + (1 - p) * y_h,
    asp_l = 0.5 * (y_l + mean_inc),
    asp_h = 0.5 * (y_h + mean_inc)
  )

ggplot(fig5_data, aes(x = p)) + 
  geom_line(aes(y = y_l, color = "y_l")) + 
  geom_line(aes(y = asp_l, color = "a_l"), linetype = "dashed") +  
  geom_line(aes(y = y_h, color = "y_h")) + 
  geom_line(aes(y = asp_h, color = "a_h"), linetype = "dashed") + 
  labs(
    title = "Figure 5: Bimodal Distributions",
    x = "p", y = "Income", color = element_blank()
  )

ggsave(paste0(fig_path, "fig5.jpg"))

# Example 2: Heterogeneity (TODO) -----------------------------------------

# Stochastic production
y_init <- seq(10.0, 1000.0, length.out = 100) # uniform dist
yvals <- y_init

# Find steady state
dif <- 10.0
tol <- 1e-3
its <- 0
maxit <- 100
p_old <- 1

while (dif > tol & its < maxit) {
  avg_inc <- mean(yvals)
  a_y <- 0.5 * (yvals + avg_inc) # aspirations
  theta <- rlnorm(1, 1, 0.04) # random theta (aggregate)
  
  yvals_next <- map2_dbl(yvals, a_y, find_z)
  
  its <- its + 1
  yvals <- yvals_next
}

# Q: How are they getting the distribution

