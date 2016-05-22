## ---- echo=FALSE, message = FALSE, cache=FALSE---------------------------
# install.packages("oem", repos = "http://cran.us.r-project.org")

## ---- message = FALSE, cache=FALSE, eval = FALSE-------------------------
#  install.packages("devtools", repos = "http://cran.us.r-project.org")

## ---- message = FALSE, cache=FALSE, eval=FALSE---------------------------
#  library(devtools)
#  install_github("jaredhuling/oem")

## ---- echo=FALSE, message = FALSE, cache=FALSE---------------------------
library(oem)

## ---- echo=FALSE, message = FALSE, cache=FALSE---------------------------
nobs  <- 1e4
nvars <- 100
x <- matrix(rnorm(nobs * nvars), ncol = nvars)
y <- drop(x %*% c(0.5, 0.5, -0.5, -0.5, 1, rep(0, nvars - 5))) + rnorm(nobs, sd = 2)

## ---- echo=FALSE, message = FALSE, cache=FALSE---------------------------
fit1 <- oem(x = x, y = y, penalty = "lasso")

## ---- fig.show='hold', fig.width = 7.15, fig.height = 5------------------
plot(fit1)

## ---- echo=FALSE, message = FALSE, cache=FALSE---------------------------
fit2 <- oem(x = x, y = y, penalty = c("lasso", "mcp", "grp.lasso"),
            groups = rep(1:20, each = 5))

## ---- fig.show='hold', fig.width = 7.15, fig.height = 5------------------
layout(matrix(1:3, ncol = 3))
plot(fit2, which.model = 1)
plot(fit2, which.model = 2)
plot(fit2, which.model = 3)

## ---- echo=FALSE, message = FALSE, cache=FALSE---------------------------
cvfit1 <- cv.oem(x = x, y = y, penalty = c("lasso", "mcp", "grp.lasso"), 
                 groups = rep(1:20, each = 5), 
                 nfolds = 10)

## ---- fig.show='hold', fig.width = 7.15, fig.height = 4------------------
layout(matrix(1:3, ncol = 3))
plot(cvfit1, which.model = 1)
plot(cvfit1, which.model = 2)
plot(cvfit1, which.model = 3)

