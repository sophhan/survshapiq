# Load necessary packages
library(simsurv)
library(survival)
#library(dplyr)

################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-INDEPENDENCE 
# Step 1: Create a covariate data frame
set.seed(123)
n <- 5000
covs <- data.frame(
  id = 1:n,
  x1 = rnorm(n, mean = 0, sd = 1),
  x2 = rnorm(n, mean = 0, sd = 1),
  x3 = rnorm(n, mean = 0, sd = 1)
)
betas <- data.frame(lambda = rep(0.03, n),
                    x1 = rep(0.8, n), 
                    x2 = rep(0.5, n), 
                    x3 = rep(0.9, n),
                    x1_x3 = rep(-0.6, n)
)
# Step 2: Simulate data
hazard_fun <- function(t, x, betas = beta, ...) {
  # linear predictor with interaction
  lin_pred <- betas["x1"] * x["x1"] +
    betas["x2"] * x["x2"] +
    betas["x3"] * x["x3"] +
    betas["x1_x3"] * x["x1"] * x["x3"]
  # hazard function
  return(betas["lambda"] * exp(lin_pred))
}
# Step 3: Simulate survival data
simdata_ti <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))
# Step 4: Merge with covariates
simdata_ti_full <- merge(simdata_ti, covs, by = "id")
# Step 5: Check it out
head(simdata_ti_full)
round(simdata_ti_full$eventtime,1)
table(round(simdata_ti_full$eventtime,0))
# Step 6: Export to csv
write.csv(simdata_ti_full, "/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_ti.csv", row.names = F)


################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-DEPENDENCE IN MAIN EFFECTS
set.seed(123)
# Step 1: create covariate dataframe 
n <- 5000
covs <- data.frame(
  id = 1:n,
  x1 = rnorm(n, 0, 1),
  x2 = rnorm(n, 0, 1),
  x3 = rnorm(n, 0, 1)
)

# Start with milder coefficients to reduce extreme heterogeneity
betas <- c(lambda = 0.03,
           x1 = 0.8,
           x1_t = -1.2,
           x2 = 0.5,
           x3 = 0.9,
           x1_x3 = -0.6)

# Hazard function: time-dependent PH via log(t+1)
hazard_fun <- function(t, x, betas, ...) {
  lp <- betas["x1"] * x["x1"] +
        betas["x1_t"] * x["x1"] * log(t + 1) +
        betas["x2"] * x["x2"] +
        betas["x3"] * x["x3"] +
        betas["x1_x3"] * x["x1"] * x["x3"]
  betas["lambda"] * exp(lp)
}

maxt <- 70

sim_td <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = maxt, interval = c(0, 1000))

# Check distribution of *events* (exclude admin-censored at 70)
event_times <- sim_td$eventtime[sim_td$event == 1]
summary(event_times)
# Step 4: Merge with covariates
simdata_td_full <- merge(sim_td, covs, by = "id")
# Step 5: Check it out
head(simdata_td_full)
table(round(simdata_td_full$eventtime,0))
# Step 6: Export to csv
write.csv(simdata_td_full, "/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_td_main.csv", row.names = F)


################ LINEAR MAIN EFFECTS AND LINEAR INTERACTIONS
###### TIME-DEPENDENCE IN INTERACTIONS
set.seed(123)
# Step 1: create covariate dataframe 
n <- 5000
covs <- data.frame(
  id = 1:n,
  x1 = rnorm(n, 0, 1),
  x2 = rnorm(n, 0, 1),
  x3 = rnorm(n, 0, 1)
)

# Start with milder coefficients to reduce extreme heterogeneity
betas <- c(lambda = 0.03,    
           x1 = 0.8,
           x2 = 0.5,
           x3 = 0.9,
           x1_x2 = -0.6,
           x1_x2_t = -0.4)  

# Hazard function: time-dependent PH via log(t+1)
hazard_fun <- function(t, x, betas, ...) {
  lp <- betas["x1"] * x["x1"] +
        betas["x2"] * x["x2"] +
        betas["x3"] * x["x3"] +
        betas["x1_x2"] * x["x1"] * x["x2"] +
        betas["x1_x2_t"] * x["x1"] * x["x2"] * log(t + 1)
  betas["lambda"] * exp(lp)
}

maxt <- 70

sim_td <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = maxt, interval = c(0, 1000))

# Check distribution of *events* (exclude admin-censored at 70)
event_times <- sim_td$eventtime[sim_td$event == 1]
summary(event_times)
# Step 4: Merge with covariates
simdata_td_full <- merge(sim_td, covs, by = "id")
# Step 5: Check it out
head(simdata_td_full)
table(round(simdata_td_full$eventtime,0))
# Step 6: Export to csv
write.csv(simdata_td_full, "/home/slangbei/survshapiq/survshapiq/simulation/simdata_linear_td_inter.csv", row.names = F)


################ ADDITIVE MAIN EFFECT MODEL 
########## TIME-INDEPENDENCE 
set.seed(123)

# 1) Covariates
n <- 5000
covs <- data.frame(
  id = 1:n,
  x1 = rnorm(n, 0, 1),
  x2 = rnorm(n, 0, 1),
  x3 = rnorm(n, 0, 1)
)

# 2) Coefficients
betas <- c(
  lambda = 0.015,   # higher baseline so more events before censoring
  x1     = -1.5,   # mild quadratic
  x2     = 1,     # mild S-shape
  x3     = 0.6      # moderate linear
)

hazard_fun <- function(t, x, betas, ...) {
  x1_effect <- betas["x1"] * ((x["x1"]^2) - 1)
  x2_effect <- betas["x2"] * ((2/pi) * atan(0.5 * x["x2"]))
  lp <- x1_effect + x2_effect + betas["x3"] * x["x3"]
  betas["lambda"] * exp(lp)
}

# 4) Simulate
sim_td <- simsurv(hazard = hazard_fun, x = covs, betas = betas,
                  maxt = 70, interval = c(0, 100))

# 5) Merge & quick checks
simdata_add_full <- merge(sim_td, covs, by = "id")
head(simdata_add_full)
table(round(simdata_add_full$eventtime, 0))

# 6) Export
write.csv(simdata_add_full,
          "/home/slangbei/survshapiq/survshapiq/simulation/simdata_add_ti.csv",
          row.names = FALSE)



################ GENERAL ADDITIVE MODEL 
########## TIME-INDEPENDENCE 
set.seed(123)

# Step 1: create covariate dataframe 
n <- 5000
covs <- data.frame(
  id = 1:n,
  x1 = rnorm(n, 0, 1),
  x2 = rnorm(n, 0, 1),
  x3 = rnorm(n, 0, 1)
)

# New coefficients (include nonlinear terms for age & bmi)
betas <- c(
  lambda   = 0.01,   # baseline hazard
  # main effects
  x1_lin   = 0.2,    
  x1_quad  = -0.3,   # nonlinear curvature
  x2_s     = 0.5,    # nonlinear S-shape
  x3_lin   = -0.4,   # treatment effect
  # interactions
  x1x2_lin = 0.2,    
  x1x3_int = 0.3   
  #x2x3_nl  = -0.4
)

hazard_fun <- function(t, x, betas, ...) {
  # nonlinear transforms
  x1_quad <- (x["x1"]^2 - 1)            # centered quadratic
  x2_s    <- (2/pi) * atan(0.7 * x["x2"])  # bounded S-shape
  
  # linear terms
  x1_lin  <- x["x1"]
  x2_lin  <- x["x2"]
  x3_lin  <- x["x3"]
  
  # interactions
  x1x2_lin <- x1_lin * x2_lin
  x1x3_int <- x1_quad * x3_lin          # nonlinear × binary
  #x2x3_nl  <- x2_s * x3_lin             # nonlinear × binary
  
  lp <- betas["x1_lin"]  * x1_lin   +
        betas["x1_quad"] * x1_quad  +
        betas["x2_s"]    * x2_s     +
        betas["x3_lin"]  * x3_lin   +
        betas["x1x2_lin"]* x1x2_lin +
        betas["x1x3_int"]* x1x3_int 
        #betas["x2x3_nl"] * x2x3_nl
  
  betas["lambda"] * exp(lp)
}

sim_td <- simsurv(hazard = hazard_fun, x = covs, betas = betas, 
                  maxt = 70, interval = c(0, 100))

# Step 4: Merge with covariates
simdata_genadd_full <- merge(sim_td, covs, by = "id")
# Step 5: Quick check
head(simdata_genadd_full)
table(round(simdata_genadd_full$eventtime,0))

# Step 6: Export to csv
write.csv(simdata_genadd_full, 
          "/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_ti.csv", 
          row.names = FALSE)


################ GENERAL ADDITIVE MODEL 
########## TIME-DEPENDENCE IN MAIN EFFECTS
set.seed(123)

# Step 1: create covariate dataframe 
n <- 5000
covs <- data.frame(
  id = 1:n,
  x1 = rnorm(n, 0, 1),
  x2 = rnorm(n, 0, 1),
  x3 = rnorm(n, 0, 1)
)

# New coefficients (include nonlinear terms for age & bmi)
betas <- c(
  lambda   = 0.01,   # baseline hazard
  # main effects
  x1_lin   = 0.2,
  x1_quad  = -0.3,   # nonlinear curvature
  td_x1    = -0.4,   # time-dependent slope for x1
  x2_s     = 0.5,    # nonlinear S-shape
  x3_lin   = -0.4,   # treatment effect
  # interactions
  x1x2_lin = 0.2,
  x1x3_int = 0.3
)

hazard_fun <- function(t, x, betas, ...) {
  # nonlinear transforms
  x1_quad <- (x["x1"]^2 - 1)                # centered quadratic
  x2_s    <- (2/pi) * atan(0.7 * x["x2"])  # bounded S-shape

  # linear terms
  x1_lin  <- x["x1"]
  x2_lin  <- x["x2"]
  x3_lin  <- x["x3"]

  # interactions
  x1x2_lin <- x1_lin * x2_lin
  x1x3_int <- x1_quad * x3_lin

  # time-dependent effect of x1
  x1_td <- x1_lin * (betas["x1_lin"] + betas["td_x1"] * log(t + 1))

  lp <- x1_td +
        betas["x1_quad"] * x1_quad  +
        betas["x2_s"]    * x2_s     +
        betas["x3_lin"]  * x3_lin   +
        betas["x1x2_lin"]* x1x2_lin +
        betas["x1x3_int"]* x1x3_int 

  betas["lambda"] * exp(lp)
}

# Step 3: Simulate
sim_td <- simsurv(hazard = hazard_fun, x = covs, betas = betas, 
                  maxt = 70, interval = c(0, 100))

# Step 4: Merge with covariates
simdata_genadd_full <- merge(sim_td, covs, by = "id")

# Step 5: Quick check
head(simdata_genadd_full)
table(round(simdata_genadd_full$eventtime, 0))

# Step 6: Export to csv
write.csv(simdata_genadd_full, "/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_td_main.csv", row.names = FALSE)



################ GENERAL ADDITIVE MODEL 
########## TIME-DEPENDENCE IN INTERACTIONS
set.seed(123)

# Step 1: create covariate dataframe 
n <- 5000
covs <- data.frame(
  id = 1:n,
  x1 = rnorm(n, 0, 1),
  x2 = rnorm(n, 0, 1),
  x3 = rnorm(n, 0, 1)
)

# New coefficients (include nonlinear terms for age & bmi)
betas <- c(
  lambda     = 0.01,   # baseline hazard
  # main effects
  x1_lin     = 0.2,
  x1_quad    = -0.3,   # nonlinear curvature
  x2_s       = 0.5,    # nonlinear S-shape
  x3_lin     = -0.4,   # treatment effect
  # interactions
  x1x2_lin   = 0.2,
  td_x1x2    = -0.4,   # time-dependent slope for x1*x2
  x1x3_int   = 0.3
)

hazard_fun <- function(t, x, betas, ...) {
  # nonlinear transforms
  x1_quad <- (x["x1"]^2 - 1)                 # centered quadratic
  x2_s    <- (2/pi) * atan(0.7 * x["x2"])   # bounded S-shape

  # linear terms
  x1_lin  <- x["x1"]
  x2_lin  <- x["x2"]
  x3_lin  <- x["x3"]

  # interactions
  x1x2_lin <- x1_lin * x2_lin
  x1x3_int <- x1_quad * x3_lin

  # time-dependent effect of x1*x2
  x1x2_td <- x1x2_lin * (betas["x1x2_lin"] + betas["td_x1x2"] * log(t + 1))

  lp <- betas["x1_lin"]  * x1_lin   +
        betas["x1_quad"] * x1_quad  +
        betas["x2_s"]    * x2_s     +
        betas["x3_lin"]  * x3_lin   +
        x1x2_td                        + # << time-dependent interaction
        betas["x1x3_int"]* x1x3_int 

  betas["lambda"] * exp(lp)
}

# Step 3: Simulate
sim_td <- simsurv(hazard = hazard_fun, x = covs, betas = betas, 
                  maxt = 70, interval = c(0, 100))

# Step 4: Merge with covariates
simdata_genadd_full <- merge(sim_td, covs, by = "id")

# Step 5: Quick check
head(simdata_genadd_full)
table(round(simdata_genadd_full$eventtime, 0))

# Step 6: Export to csv
write.csv(simdata_genadd_full, 
          "/home/slangbei/survshapiq/survshapiq/simulation/simdata_genadd_td_interaction.csv", 
          row.names = FALSE)
