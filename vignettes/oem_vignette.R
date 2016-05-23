## ---- echo=FALSE, message = FALSE, cache=FALSE---------------------------
# install.packages("oem", repos = "http://cran.us.r-project.org")

## ---- message = FALSE, cache=FALSE, eval = FALSE-------------------------
#  install.packages("devtools", repos = "http://cran.us.r-project.org")

## ---- message = FALSE, cache=FALSE, eval=FALSE---------------------------
#  library(devtools)
#  install_github("jaredhuling/oem")

## ---- message = FALSE, cache=FALSE---------------------------------------
library(oem)

## ---- message = FALSE, cache=FALSE---------------------------------------
nobs  <- 1e4
nvars <- 100
x <- matrix(rnorm(nobs * nvars), ncol = nvars)
y <- drop(x %*% c(0.5, 0.5, -0.5, -0.5, 1, rep(0, nvars - 5))) + rnorm(nobs, sd = 4)

## ---- message = FALSE, cache=FALSE---------------------------------------
fit1 <- oem(x = x, y = y, penalty = "lasso")

## ---- fig.show='hold', fig.width = 7.15, fig.height = 5------------------
plot(fit1)

## ---- message = FALSE, cache=FALSE---------------------------------------
fit2 <- oem(x = x, y = y, penalty = c("lasso", "mcp", "grp.lasso"),
            groups = rep(1:20, each = 5))

## ---- fig.show='hold', fig.width = 7.15, fig.height = 5------------------
layout(matrix(1:3, ncol = 3))
plot(fit2, which.model = 1)
plot(fit2, which.model = 2)
plot(fit2, which.model = 3)

## ---- message = FALSE, cache=FALSE---------------------------------------
nobs  <- 1e5
nvars <- 100
x2 <- matrix(rnorm(nobs * nvars), ncol = nvars)
y2 <- drop(x2 %*% c(0.5, 0.5, -0.5, -0.5, 1, rep(0, nvars - 5))) + rnorm(nobs, sd = 4)

system.time(fit2a <- oem(x = x2, y = y2, penalty = c("grp.lasso"),
                         groups = rep(1:20, each = 5), nlambda = 100L))
system.time(fit2b <- oem(x = x2, y = y2, penalty = c("grp.lasso", "lasso", "mcp", "scad", "elastic.net"),
                         groups = rep(1:20, each = 5), nlambda = 100L))
system.time(fit2c <- oem(x = x2, y = y2, penalty = c("grp.lasso", "lasso", "mcp", "scad", "elastic.net"),
                         groups = rep(1:20, each = 5), nlambda = 500L))

## ---- message = FALSE, cache=FALSE---------------------------------------
nobs  <- 5e4
nvars <- 100
x2 <- matrix(rnorm(nobs * nvars), ncol = nvars)

y2 <- rbinom(nobs, 1, prob = 1 / (1 + exp(-drop(x2 %*% c(0.15, 0.15, -0.15, -0.15, 0.25, rep(0, nvars - 5))))))


system.time(fit2a <- oem(x = x2, y = y2, penalty = c("grp.lasso"),
                         family = "binomial",
                         groups = rep(1:20, each = 5), nlambda = 100L))
system.time(fit2b <- oem(x = x2, y = y2, penalty = c("grp.lasso", "lasso", "mcp", "scad", "elastic.net"),
                         family = "binomial",
                         groups = rep(1:20, each = 5), nlambda = 100L))


## ---- message = FALSE, cache=FALSE---------------------------------------
cvfit1 <- cv.oem(x = x, y = y, penalty = c("lasso", "mcp", "grp.lasso"), 
                 groups = rep(1:20, each = 5), 
                 nfolds = 10)

## ---- fig.show='hold', fig.width = 7.15, fig.height = 3.75---------------
layout(matrix(1:3, ncol = 3))
plot(cvfit1, which.model = 1)
plot(cvfit1, which.model = 2)
plot(cvfit1, which.model = 3)

## ---- message = FALSE, cache=FALSE---------------------------------------
nobs  <- 2e3
nvars <- 20
x <- matrix(runif(nobs * nvars, max = 2), ncol = nvars)

y <- rbinom(nobs, 1, prob = 1 / (1 + exp(-drop(x %*% c(0.25, -1, -1, -0.5, -0.5, -0.25, rep(0, nvars - 6))))))

## ---- message = FALSE, cache=FALSE---------------------------------------
cvfit2 <- cv.oem(x = x, y = y, penalty = c("lasso", "mcp", "grp.lasso"), 
                 family = "binomial",
                 type.measure = "class",
                 groups = rep(1:10, each = 2), 
                 nfolds = 10)

## ---- echo = FALSE, fig.show='hold', fig.width = 7.15, fig.height = 3.75----
layout(matrix(1:3, ncol = 3))
plot(cvfit2, which.model = 1)
plot(cvfit2, which.model = 2)
plot(cvfit2, which.model = 3)

## ------------------------------------------------------------------------
mean(y)

## ---- message = FALSE, cache=FALSE---------------------------------------
cvfit2 <- cv.oem(x = x, y = y, penalty = c("lasso", "mcp", "grp.lasso"), 
                 family = "binomial",
                 type.measure = "auc",
                 groups = rep(1:10, each = 2), 
                 nfolds = 10)

## ---- echo = FALSE, fig.show='hold', fig.width = 7.15, fig.height = 3.75----
layout(matrix(1:3, ncol = 3))
plot(cvfit2, which.model = 1)
plot(cvfit2, which.model = 2)
plot(cvfit2, which.model = 3)

## ---- message = FALSE, cache=FALSE---------------------------------------

nobs  <- 1e4
nvars <- 102
x <- matrix(rnorm(nobs * nvars), ncol = nvars)
y <- drop(x %*% c(0.5, 0.5, -0.5, -0.5, 1, 0.5, rep(0, nvars - 6))) + rnorm(nobs, sd = 4)

lams <- exp(seq(log(2.5), log(5e-5), length.out = 100L))

ols.estimates <- coef(lm.fit(y = y, x = cbind(1, x)))[-1]

fit.adaptive <- oem(x = x, y = y, penalty = c("lasso"),
                    penalty.factor = 1 / abs(ols.estimates),
                    lambda = lams)

group.indicators <- rep(1:34, each = 3)

## norms of OLS estimates for each group
group.norms      <- sapply(1:34, function(idx) sqrt(sum((ols.estimates[group.indicators == idx]) ^ 2)))
fit.adaptive.grp <- oem(x = x, y = y, penalty = c("grp.lasso"),
                        group.weights = 1 / group.norms,
                        groups = group.indicators, 
                        lambda = lams)


## ---- echo = FALSE, fig.show='hold', fig.width = 7.15, fig.height = 4.25----
layout(matrix(1:2, ncol = 2))
plot(fit.adaptive)
plot(fit.adaptive.grp)

