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
system.time(cvfit1 <- cv.oem(x = x, y = y, penalty = c("lasso", "mcp", "grp.lasso"), 
                             groups = rep(1:20, each = 5), 
                             nfolds = 10))

## ---- fig.show='hold', fig.width = 7.15, fig.height = 3.75---------------
layout(matrix(1:3, ncol = 3))
plot(cvfit1, which.model = 1)
plot(cvfit1, which.model = 2)
plot(cvfit1, which.model = 3)

## ---- message = FALSE, cache=FALSE---------------------------------------

nobsc  <- 1e5
nvarsc <- 100
xc <- matrix(rnorm(nobsc * nvarsc), ncol = nvarsc)
yc <- drop(xc %*% c(0.5, 0.5, -0.5, -0.5, 1, rep(0, nvarsc - 5))) + rnorm(nobsc, sd = 4)

system.time(cvalfit1 <- cv.oem(x = xc, y = yc, penalty = "lasso", 
                               groups = rep(1:20, each = 5), 
                               nfolds = 10))

system.time(xvalfit1 <- xval.oem(x = xc, y = yc, penalty = "lasso",
                                 groups = rep(1:20, each = 5), 
                                 nfolds = 10))

system.time(xvalfit2 <- xval.oem(x = xc, y = yc, penalty = "lasso",
                                 groups = rep(1:20, each = 5), 
                                 nfolds = 10, ncores = 2))

system.time(ofit1 <- oem(x = xc, y = yc, penalty = "lasso",
                         groups = rep(1:20, each = 5)))

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
                 nfolds = 10, standardize = FALSE)

## ---- echo = FALSE, fig.show='hold', fig.width = 7.15, fig.height = 3.75----
yrng <- range(c(unlist(cvfit2$cvup), unlist(cvfit2$cvlo)))
layout(matrix(1:3, ncol = 3))
plot(cvfit2, which.model = 1, ylim = yrng)
plot(cvfit2, which.model = 2, ylim = yrng)
plot(cvfit2, which.model = 3, ylim = yrng)

## ------------------------------------------------------------------------
mean(y)

## ---- message = FALSE, cache=FALSE---------------------------------------
cvfit2 <- cv.oem(x = x, y = y, penalty = c("lasso", "mcp", "grp.lasso"), 
                 family = "binomial",
                 type.measure = "auc",
                 groups = rep(1:10, each = 2), 
                 nfolds = 10, standardize = FALSE)

## ---- echo = FALSE, fig.show='hold', fig.width = 7.15, fig.height = 3.75----
yrng <- range(c(unlist(cvfit2$cvup), unlist(cvfit2$cvlo)))
layout(matrix(1:3, ncol = 3))
plot(cvfit2, which.model = 1, ylim = yrng)
plot(cvfit2, which.model = 2, ylim = yrng)
plot(cvfit2, which.model = 3, ylim = yrng)

## ---- message = FALSE, cache=FALSE---------------------------------------
xtx <- crossprod(xc) / nrow(xc)
xty <- crossprod(xc, yc) / nrow(xc)


system.time(fit <- oem(x = xc, y = yc, 
                       penalty = c("lasso", "grp.lasso"), 
                       standardize = FALSE, intercept = FALSE,
                       groups = rep(1:20, each = 5)))

system.time(fit.xtx <- oem.xtx(xtx = xtx, xty = xty, 
                               penalty = c("lasso", "grp.lasso"), 
                               groups = rep(1:20, each = 5))  )  
                   
max(abs(fit$beta[[1]][-1,] - fit.xtx$beta[[1]]))
max(abs(fit$beta[[2]][-1,] - fit.xtx$beta[[2]])) 

col.std <- apply(xc, 2, sd)
fit.xtx.s <- oem.xtx(xtx = xtx, xty = xty, 
                     scale.factor = col.std,
                     penalty = c("lasso", "grp.lasso"), 
                     groups = rep(1:20, each = 5))  


## ---- message = FALSE, cache=FALSE---------------------------------------
set.seed(123)
nrows <- 50000
ncols <- 100
bkFile <- "bigmat.bk"
descFile <- "bigmatk.desc"
bigmat <- filebacked.big.matrix(nrow=nrows, ncol=ncols, type="double",
                                backingfile=bkFile, backingpath=".",
                                descriptorfile=descFile,
                                dimnames=c(NULL,NULL))

# Each column value with be the column number multiplied by
# samples from a standard normal distribution.
set.seed(123)
for (i in 1:ncols) bigmat[,i] = rnorm(nrows)*i

yb <- rnorm(nrows) + bigmat[,1] - bigmat[,2]

## out-of-memory computation
fit <- big.oem(x = bigmat, y = yb, 
               penalty = c("lasso", "grp.lasso"), 
               groups = rep(1:20, each = 5))

## fitting with in-memory computation
fit2 <- oem(x = bigmat[,], y = yb, 
            penalty = c("lasso", "grp.lasso"), 
            groups = rep(1:20, each = 5))   
           
max(abs(fit$beta[[1]] - fit2$beta[[1]]))            


## ---- message = FALSE, cache=FALSE---------------------------------------

nobsc  <- 1e5
nvarsc <- 500
xc <- matrix(rnorm(nobsc * nvarsc), ncol = nvarsc)
yc <- drop(xc %*% c(0.5, 0.5, -0.5, -0.5, 1, rep(0, nvarsc - 5))) + rnorm(nobsc, sd = 4)


system.time(fit <- oem(x = xc, y = yc, 
                       penalty = c("lasso", "grp.lasso"), 
                       standardize = FALSE, intercept = FALSE,
                       groups = rep(1:20, each = 25)))

system.time(fitp <- oem(x = xc, y = yc, 
                        penalty = c("lasso", "grp.lasso"), 
                        standardize = FALSE, intercept = FALSE,
                        groups = rep(1:20, each = 25), ncores = 2))


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

