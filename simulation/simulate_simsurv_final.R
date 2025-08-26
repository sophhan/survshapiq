# Load necessary packages
library(simsurv)
library(survival)
library(dplyr)

### Simulate continuous/binary interaction with time-dependence with additional variables 
# Step 1: Create a covariate data frame
set.seed(123)
n <- 5000
covs <- data.frame(
  id = 1:n,
  age = rnorm(n, mean = 0, sd = 1),
  treatment = rbinom(n, 1, 0.5),
  bmi = rnorm(n, mean = 0, sd = 1)
)
betas <- data.frame(lambda = rep(0.01, n),
                    age = rep(0.3, n), 
                    bmi = rep(0.9, n), 
                    treatment = rep(-0.7, n),
                    age_treatment = rep(-5, n)
)
# Step 2: Simulate data
hazard_fun <- function(t, x, betas = beta, ...) {
  # linear predictor with interaction
  lin_pred <- betas["age"] * x["age"] +
    betas["bmi"] * x["bmi"] +
    betas["treatment"] * x["treatment"] +
    betas["age_treatment"] * x["age"] * x["treatment"] 
  # hazard function
  return(betas["lambda"] * exp(lin_pred))
}
# Step 3: Simulate survival data
simdata_ti <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 200, interval = c(0, 1000))
# Step 4: Merge with covariates
simdata_ti_full <- merge(simdata_ti, covs, by = "id")
# Step 5: Check it out
head(simdata_ti_full)
# Step 6: Export to csv
write.csv(simdata_ti_full, "/home/slangbei/survshapiq/survshapiq/simulation/simdata_ti.csv", row.names = F)


### Simulate continuous/binary interaction with time-dependence with additional variables 
# Step 1: Create a covariate data frame
set.seed(123)
n <- 5000
covs <- data.frame(
  id = 1:n,
  age = rnorm(n, mean = 0, sd = 1),
  treatment = rbinom(n, 1, 0.5),
  bmi = rnorm(n, mean = 0, sd = 1)
)
betas <- data.frame(lambda = rep(0.01, n),
                    age = rep(0.3, n), 
                    bmi = rep(0.9, n), 
                    treatment = rep(-0.7, n),
                    age_treatment = rep(-5, n),
                    age_treatment_t = rep(9, n)
)
# Step 2: Simulate data
hazard_fun <- function(t, x, betas = beta, ...) {
  # linear predictor with interaction
  lin_pred <- betas["age"] * x["age"] +
    betas["bmi"] * x["bmi"] +
    betas["treatment"] * x["treatment"] +
    betas["age_treatment"] * x["age"] * x["treatment"] +
    betas["age_treatment_t"] * x["age"] * x["treatment"] * log(t + 1)
  # hazard function
  return(betas["lambda"] * exp(lin_pred))
}
# Step 3: Simulate survival data
simdata_td <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 200, interval = c(0, 1000))
# Step 4: Merge with covariates
simdata_td_full <- merge(simdata_td, covs, by = "id")
# Step 5: Check it out
head(simdata_td_full)
# Step 6: Export to csv
write.csv(simdata_td_full, "/home/slangbei/survshapiq/survshapiq/simulation/simdata_td.csv", row.names = F)


### Simulate continuous interaction without time-dependence with additional variables 
# Step 1: Create a covariate data frame
set.seed(123)
n <- 5000
covs <- data.frame(
  id = 1:n,
  age = rnorm(n, mean = 0, sd = 1),
  treatment = rnorm(n, mean = 0, sd = 1),
  bmi = rnorm(n, mean = 0, sd = 1)
)
betas <- data.frame(lambda = rep(0.03, n),
                    age = rep(0.8, n), 
                    bmi = rep(0.5, n), 
                    treatment = rep(0.9, n),
                    age_treatment = rep(-0.6, n)
)
# Step 2: Simulate data
hazard_fun <- function(t, x, betas = beta, ...) {
  # linear predictor with interaction
  lin_pred <- betas["age"] * x["age"] +
    betas["bmi"] * x["bmi"] +
    betas["treatment"] * x["treatment"] +
    betas["age_treatment"] * x["age"] * x["treatment"] 
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
write.csv(simdata_ti_full, "/home/slangbei/survshapiq/survshapiq/simulation/simdata_ti_haz.csv", row.names = F)


### Simulate continuous interaction with time-dependence with additional variables 
set.seed(123)
# Step 1: create covariate dataframe 
n <- 5000
covs <- data.frame(
  id = 1:n,
  age = rnorm(n, 0, 1),
  treatment = rnorm(n, 0, 1),
  bmi = rnorm(n, 0, 1)
)

# Start with milder coefficients to reduce extreme heterogeneity
betas <- c(lambda = 0.03,    
           age = 0.8,
           bmi = 0.5,
           treatment = 0.9,
           age_treatment = -0.6,
           age_treatment_t = -0.4)  

# Hazard function: time-dependent PH via log(t+1)
hazard_fun <- function(t, x, betas, ...) {
  lp <- betas["age"] * x["age"] +
        betas["bmi"] * x["bmi"] +
        betas["treatment"] * x["treatment"] +
        betas["age_treatment"] * x["age"] * x["treatment"] +
        betas["age_treatment_t"] * x["age"] * x["treatment"] * log(t + 1)
  betas["lambda"] * exp(lp)
}

# Calibrate lambda to hit ~target_event_rate before maxt
calibrate_lambda <- function(target_event_rate = 0.6, maxt = 70, covs, betas, tries = 3) {
  # helper: simulate once, return event rate before maxt
  sim_rate <- function(lambda) {
    b <- betas; b["lambda"] <- lambda
    sim <- simsurv(hazard = hazard_fun, x = covs, betas = b, maxt = maxt, interval = c(0, 1000))
    mean(sim$event == 1)
  }
  # bracket lambda on log scale for stability
  f <- function(log_lambda) sim_rate(exp(log_lambda)) - target_event_rate
  uniroot(f, lower = log(1e-4), upper = log(1), tol = 1e-3)$root |> exp()
}

maxt <- 70
target_event_rate <- 0.6
betas["lambda"] <- calibrate_lambda(target_event_rate, maxt, covs, betas)
betas

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
write.csv(simdata_td_full, "/home/slangbei/survshapiq/survshapiq/simulation/simdata_td_haz.csv", row.names = F)
