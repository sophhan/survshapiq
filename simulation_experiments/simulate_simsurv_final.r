# load necessary packages
library(simsurv)
library(survival)

# define file path to save simulated datasets
path <- "/simulation/data/"

#---------------------------
# 1) Linear G(t|x), TI (no interactions)
#---------------------------

# create a covariate data frame
set.seed(123)
n <- 1000
covs <- data.frame(
  id = 1:n,
  x1 = rnorm(n, mean = 0, sd = 1),
  x2 = rnorm(n, mean = 0, sd = 1),
  x3 = rnorm(n, mean = 0, sd = 1)
)

# simulate data
betas <- c(lambda = 0.03,
           x1 = 0.4,
           x2 = -0.8,
           x3 = -0.6)

hazard_fun <- function(t, x, betas = beta, ...) {
  lp <- betas["x1"] * x["x1"] +
    betas["x2"] * x["x2"] +
    betas["x3"] * x["x3"]

  return(betas["lambda"] * exp(lp))
}

sim <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))

# merge with covariates
simdata <- merge(sim, covs, by = "id")

# check it out
summary(simdata$eventtime)

# export to csv
write.csv(simdata, 
          paste0(path, "1_simdata_linear_ti.csv"), 
          row.names = F)

#---------------------------
# 2) Linear G(t|x), TD Main (no interactions)
#---------------------------

# simulate data
betas <- c(lambda = 0.03,
           x1 = 0.4,
           x2 = -0.8,
           x3 = -0.6)

hazard_fun <- function(t, x, betas = beta, ...) {
  lp <- betas["x1"] * x["x1"] * log(t + 1) +
    betas["x2"] * x["x2"] +
    betas["x3"] * x["x3"]

  return(betas["lambda"] * exp(lp))
}

sim <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))

# merge with covariates
simdata <- merge(sim, covs, by = "id")

# check it out
summary(simdata$eventtime)

# export to csv
write.csv(simdata, 
          paste0(path, "2_simdata_linear_tdmain.csv"), 
          row.names = F)


#---------------------------
# 3) Linear G(t|x), TI (interactions)
#---------------------------

# simulate data
betas <- c(lambda = 0.03,
           x1 = 0.4,
           x2 = -0.8,
           x3 = -0.6,
           x1x3 = -0.9)

hazard_fun <- function(t, x, betas = beta, ...) {
  lp <- betas["x1"] * x["x1"] +
    betas["x2"] * x["x2"] +
    betas["x3"] * x["x3"] +
    betas["x1x3"] * x["x1"] * x["x3"]

  return(betas["lambda"] * exp(lp))
}

sim <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))

# merge with covariates
simdata <- merge(sim, covs, by = "id")

# check it out
summary(simdata$eventtime)

# export to csv
write.csv(simdata, 
          paste0(path, "3_simdata_linear_ti_inter.csv"), 
          row.names = F)


#---------------------------
# 4) Linear G(t|x), TD MAIN (interactions)
#---------------------------

# simulate data
betas <- c(lambda = 0.03,
           x1 = 0.4,
           x2 = -0.8,
           x3 = -0.6,
           x1x3 = -0.9)

hazard_fun <- function(t, x, betas, ...) {
   lp <- betas["x1"] * x["x1"] * log(t + 1) +
        betas["x2"] * x["x2"] +
        betas["x3"] * x["x3"] +
        betas["x1x3"] * x["x1"] * x["x3"] 
        
  betas["lambda"] * exp(lp)
}

sim <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))

# merge with covariates
simdata <- merge(sim, covs, by = "id")

# check it out
summary(simdata$eventtime)
table(round(simdata$eventtime, 0))

# export to csv
write.csv(simdata, 
          paste0(path, "4_simdata_linear_tdmain_inter.csv"), 
          row.names = F)


#---------------------------
# 5) Linear G(t|x), TD INTER (interactions)
#---------------------------

# simulate data
betas <- c(lambda = 0.03,    
           x1 = 0.4,
           x2 = -0.8,
           x3 = -0.6,
           x1x3 = -0.9)  

hazard_fun <- function(t, x, betas, ...) {
  lp <- betas["x1"] * x["x1"] +
        betas["x2"] * x["x2"] +
        betas["x3"] * x["x3"] +
        betas["x1x3"] * x["x1"] * x["x3"] * log(t + 1) 

  betas["lambda"] * exp(lp)
}

sim <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))

# merge with covariates
simdata <- merge(sim, covs, by = "id")

# check it out
summary(simdata$eventtime)

# export to csv
write.csv(simdata, 
          paste0(path, "5_simdata_linear_tdinter.csv"), 
          row.names = F)


#---------------------------
# 6) Generalized Additive G(t|x), TI (no interactions)
#---------------------------

# simulate data
betas <- c(lambda = 0.03,  
  x1     = 0.4,   
  x2     = -0.8,     
  x3     = -0.6)

hazard_fun <- function(t, x, betas, ...) {
  lp <- betas["x1"] * (x["x1"]^2) + 
        betas["x2"] * ((2/pi) * atan(0.7 * x["x2"])) + 
        betas["x3"] * x["x3"]

  betas["lambda"] * exp(lp)
}

sim <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))

# merge with covariates
simdata <- merge(sim, covs, by = "id")

# check it out
summary(simdata$eventtime)

# export to csv
write.csv(simdata,
          paste0(path, "6_simdata_genadd_ti.csv"),
          row.names = FALSE)


#---------------------------
# 7) Generalized Additive G(t|x), TD MAIN (no interactions)
#---------------------------

# simulate data
betas <- c(lambda = 0.03,   
  x1     = 0.4,   
  x2     = -0.8,    
  x3     = -0.6
)

hazard_fun <- function(t, x, betas, ...) {
  lp <- betas["x1"] * (x["x1"]^2 * log(t+1)) + 
        betas["x2"] * ((2/pi) * atan(0.7 * x["x2"])) + 
        betas["x3"] * x["x3"] 

  betas["lambda"] * exp(lp)
}

sim <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))

# merge with covariates
simdata <- merge(sim, covs, by = "id")

# check it out
summary(simdata$eventtime)

# export to csv
write.csv(simdata,
          paste0(path, "7_simdata_genadd_tdmain.csv"),
          row.names = FALSE)



#---------------------------
# 8) Generalized Additive G(t|x), TI (interactions)
#---------------------------

# simulate data
betas <- c(lambda = 0.03,
  x1     = 0.4,   
  x2     = -0.8,    
  x3     = -0.6,
  x1x2 = -0.5,
  x1x3 = 0.2
)

hazard_fun <- function(t, x, betas, ...) {
  lp <- betas["x1"] * (x["x1"]^2) +
        betas["x2"] * (2 / pi) * atan(0.7 * x["x2"]) +
        betas["x3"]  * x["x3"] +
        betas["x1x2"] * (x["x1"] * x["x2"]) +
        betas["x1x3"] * (x["x1"] * x["x3"]^2)

  betas["lambda"] * exp(lp)
}

sim <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))

# merge with covariates
simdata <- merge(sim, covs, by = "id")

# check it out
summary(simdata$eventtime)

# export to csv
write.csv(simdata, 
          paste0(path, "8_simdata_genadd_ti_inter.csv"), 
          row.names = FALSE)


#---------------------------
# 9) Generalized Additive G(t|x), TD MAIN (interactions)
#---------------------------

# simulate data
betas <- c(lambda = 0.03,
  x1     = 0.4,   
  x2     = -0.8,    
  x3     = -0.6,
  x1x2 = -0.5,
  x1x3 = 0.2
)

hazard_fun <- function(t, x, betas, ...) {
  lp <- betas["x1"] * (x["x1"]^2) * log(t + 1) +
        betas["x2"] * (2/pi) * atan(0.7 * x["x2"]) +
        betas["x3"] * x["x3"] +
        betas["x1x2"] * (x["x1"] *  x["x2"]) +
        betas["x1x3"] * (x["x1"] * x["x3"]^2)

  betas["lambda"] * exp(lp)
}

sim <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))

# merge with covariates
simdata <- merge(sim, covs, by = "id")

# check it out
summary(simdata$eventtime)

# export to csv
write.csv(simdata, 
          paste0(path, "9_simdata_genadd_tdmain_inter.csv"), 
          row.names = FALSE)



#---------------------------
# 10) Generalized Additive G(t|x), TD Iinter (interactions)
#---------------------------

# simulate data
betas <- c(lambda = 0.03,
  x1     = 0.4,   
  x2     = -0.8,    
  x3     = -0.6,
  x1x2 = -0.5,
  x1x3 = 0.2
)

hazard_fun <- function(t, x, betas, ...) {
  lp <- betas["x1"] * (x["x1"]^2)   +
        betas["x2"] * (2 / pi) * atan(0.7 * x["x2"])     +
        betas["x3"] * x["x3"]   +
        betas["x1x2"] * (x["x1"] * x["x2"]) +
        betas["x1x3"] * (x["x1"] * x["x3"]^2 * log(t+1))

  betas["lambda"] * exp(lp)
}

sim <- simsurv(hazard = hazard_fun, x = covs, betas = betas, maxt = 70, interval = c(0, 1000))

# merge with covariates
simdata <- merge(sim, covs, by = "id")

# check it out
summary(simdata$eventtime)
table(round(simdata$eventtime,0))
head(simdata)
# export to csv
write.csv(simdata, 
          paste0(path, "10_simdata_genadd_tdinter.csv"), 
          row.names = FALSE)




