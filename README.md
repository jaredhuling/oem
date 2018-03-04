
[![version](http://www.r-pkg.org/badges/version/oem)](https://cran.r-project.org/package=oem)

### Build Status

| OS                  | Build                                                                                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux x86\_64 / OSX | [![Build Status](https://travis-ci.org/jaredhuling/oem.svg?branch=master)](https://travis-ci.org/jaredhuling/oem)                                                      |
| Windows x86\_64     | [![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/github/jaredhuling/oem?branch=master&svg=true)](https://ci.appveyor.com/project/jaredhuling/oem) |

## Introduction

The oem package provides estimaton for various penalized linear models
using the [Orthogonalizing EM
algorithm](http://amstat.tandfonline.com/doi/abs/10.1080/00401706.2015.1054436).
Documentation for the package can be found here: [oem
site](http://jaredhuling.org/oem).

Install using the **devtools** package (RcppEigen must be installed
first as well):

``` r
devtools::install_github("jaredhuling/oem")
```

or by cloning and building using `R CMD INSTALL`

## Models

### Lasso

``` r
library(microbenchmark)
library(glmnet)
library(oem)
# compute the full solution path, n > p
set.seed(123)
n <- 1000000
p <- 100
m <- 25
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)
y <- drop(x %*% b) + rnorm(n)

lambdas = oem(x, y, intercept = TRUE, standardize = FALSE, penalty = "elastic.net")$lambda[[1]]

microbenchmark(
    "glmnet[lasso]" = {res1 <- glmnet(x, y, thresh = 1e-10, 
                                      standardize = FALSE,
                                      intercept = TRUE,
                                      lambda = lambdas)}, 
    "oem[lasso]"    = {res2 <- oem(x, y,
                                   penalty = "elastic.net",
                                   intercept = TRUE, 
                                   standardize = FALSE,
                                   lambda = lambdas,
                                   tol = 1e-10)},
    times = 5
)
```

    ## Unit: seconds
    ##           expr      min       lq     mean   median       uq      max neval
    ##  glmnet[lasso] 7.440743 7.447395 7.747722 7.551486 7.613698 8.685287     5
    ##     oem[lasso] 1.958371 1.958397 2.030689 1.961159 1.975121 2.300396     5

``` r
# difference of results
max(abs(coef(res1) - res2$beta[[1]]))
```

    ## [1] 1.048072e-07

``` r
res1 <- glmnet(x, y, thresh = 1e-12, 
               standardize = FALSE,
               intercept = TRUE,
               lambda = lambdas)

# answers are now more close once we require more precise glmnet solutions
max(abs(coef(res1) - res2$beta[[1]]))
```

    ## [1] 3.763397e-09

### Nonconvex Penalties

``` r
library(sparsenet)
library(ncvreg)
library(plus)
# compute the full solution path, n > p
set.seed(123)
n <- 5000
p <- 200
m <- 25
b <- matrix(c(runif(m, -0.5, 0.5), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)
y <- drop(x %*% b) + rnorm(n)

mcp.lam <- oem(x, y, penalty = "mcp",
               gamma = 2, intercept = TRUE, 
               standardize = TRUE,
               nlambda = 200, tol = 1e-10)$lambda[[1]]

scad.lam <- oem(x, y, penalty = "scad",
               gamma = 4, intercept = TRUE, 
               standardize = TRUE,
               nlambda = 200, tol = 1e-10)$lambda[[1]]

microbenchmark(
    "sparsenet[mcp]" = {res1 <- sparsenet(x, y, thresh = 1e-10, 
                                          gamma = c(2,3), #sparsenet throws an error 
                                                          #if you only fit 1 value of gamma
                                          nlambda = 200)},
    "oem[mcp]"    = {res2 <- oem(x, y,  
                                 penalty = "mcp",
                                 gamma = 2,
                                 intercept = TRUE, 
                                 standardize = TRUE,
                                 nlambda = 200,
                                 tol = 1e-10)},
    "ncvreg[mcp]"    = {res3 <- ncvreg(x, y,  
                                   penalty = "MCP",
                                   gamma = 2,
                                   lambda = mcp.lam,
                                   eps = 1e-7)}, 
    "plus[mcp]"    = {res4 <- plus(x, y,  
                                   method = "mc+",
                                   gamma = 2,
                                   lam = mcp.lam)},
    "oem[scad]"    = {res5 <- oem(x, y,  
                                 penalty = "scad",
                                 gamma = 4,
                                 intercept = TRUE, 
                                 standardize = TRUE,
                                 nlambda = 200,
                                 tol = 1e-10)},
    "ncvreg[scad]"    = {res6 <- ncvreg(x, y,  
                                   penalty = "SCAD",
                                   gamma = 4,
                                   lambda = scad.lam,
                                   eps = 1e-7)}, 
    "plus[scad]"    = {res7 <- plus(x, y,  
                                   method = "scad",
                                   gamma = 4,
                                   lam = scad.lam)}, 
    times = 5
)
```

    ## Unit: milliseconds
    ##            expr       min        lq      mean    median        uq
    ##  sparsenet[mcp] 1781.3474 1795.5607 1972.0889 1898.6168 2091.9188
    ##        oem[mcp]  161.5994  162.6565  165.3022  165.5735  166.0121
    ##     ncvreg[mcp] 7170.5933 8386.3088 8605.6697 8511.9816 8557.9524
    ##       plus[mcp] 1661.7881 1672.9392 1740.7073 1678.9892 1817.3182
    ##       oem[scad]  136.5101  137.3331  139.6136  137.3635  137.5628
    ##    ncvreg[scad] 8052.8620 8162.1654 8247.8753 8222.6007 8317.8704
    ##      plus[scad] 1782.0859 1806.7236 2162.2228 2041.9774 2329.9171
    ##         max neval
    ##   2293.0009     5
    ##    170.6695     5
    ##  10401.5125     5
    ##   1872.5020     5
    ##    149.2985     5
    ##   8483.8781     5
    ##   2850.4098     5

``` r
diffs <- array(NA, dim = c(4, 1))
colnames(diffs) <- "abs diff"
rownames(diffs) <- c("MCP:  oem and ncvreg", "SCAD: oem and ncvreg",
                     "MCP:  oem and plus", "SCAD: oem and plus")
diffs[,1] <- c(max(abs(res2$beta[[1]] - res3$beta)), max(abs(res5$beta[[1]] - res6$beta)),
               max(abs(res2$beta[[1]][-1,1:nrow(res4$beta)] - t(res4$beta))),
               max(abs(res5$beta[[1]][-1,1:nrow(res7$beta)] - t(res7$beta))))
diffs
```

    ##                          abs diff
    ## MCP:  oem and ncvreg 1.725859e-07
    ## SCAD: oem and ncvreg 5.094648e-08
    ## MCP:  oem and plus   2.684136e-11
    ## SCAD: oem and plus   1.732409e-11

### Group Lasso

``` r
library(gglasso)
library(grpreg)
library(grplasso)
# compute the full solution path, n > p
set.seed(123)
n <- 10000
p <- 200
m <- 25
b <- matrix(c(runif(m, -0.5, 0.5), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)
y <- drop(x %*% b) + rnorm(n)
groups <- rep(1:floor(p/10), each = 10)

grp.lam <- oem(x, y, penalty = "grp.lasso",
               groups = groups,
               nlambda = 100, tol = 1e-10)$lambda[[1]]


microbenchmark(
    "gglasso[grp.lasso]" = {res1 <- gglasso(x, y, group = groups, 
                                            lambda = grp.lam, 
                                            intercept = FALSE,
                                            eps = 1e-8)},
    "oem[grp.lasso]"    = {res2 <- oem(x, y,  
                                       groups = groups,
                                       intercept = FALSE,
                                       standardize = FALSE,
                                       penalty = "grp.lasso",
                                       lambda = grp.lam,
                                       tol = 1e-10)},
    "grplasso[grp.lasso]"    = {res3 <- grplasso(x=x, y=y, 
                                                 index = groups,
                                                 standardize = FALSE, 
                                                 center = FALSE, model = LinReg(), 
                                                 lambda = grp.lam * n * 2, 
                                                 control = grpl.control(trace = 0, tol = 1e-10))}, 
    "grpreg[grp.lasso]"    = {res4 <- grpreg(x, y, group = groups, 
                                             eps = 1e-10, lambda = grp.lam)},
    times = 5
)
```

    ## Unit: milliseconds
    ##                 expr       min        lq      mean    median        uq
    ##   gglasso[grp.lasso] 3584.7379 3631.2507 4129.3979 3942.3034 4511.4096
    ##       oem[grp.lasso]  110.0432  113.5345  122.9799  115.4253  132.8513
    ##  grplasso[grp.lasso] 7528.9735 7678.8419 8652.0709 7840.2965 8539.7128
    ##    grpreg[grp.lasso] 1915.9051 2013.3947 2313.3978 2208.1039 2452.0606
    ##         max neval
    ##   4977.2881     5
    ##    143.0452     5
    ##  11672.5298     5
    ##   2977.5246     5

``` r
diffs <- array(NA, dim = c(2, 1))
colnames(diffs) <- "abs diff"
rownames(diffs) <- c("oem and gglasso", "oem and grplasso")
diffs[,1] <- c(  max(abs(res2$beta[[1]][-1,] - res1$beta)), max(abs(res2$beta[[1]][-1,] - res3$coefficients))  )
diffs
```

    ##                      abs diff
    ## oem and gglasso  1.303906e-06
    ## oem and grplasso 1.645871e-08

#### Bigger Group Lasso Example

``` r
set.seed(123)
n <- 500000
p <- 200
m <- 25
b <- matrix(c(runif(m, -0.5, 0.5), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)
y <- drop(x %*% b) + rnorm(n)
groups <- rep(1:floor(p/10), each = 10)

grp.penalties <- c("grp.lasso", "grp.mcp", "grp.scad", 
                   "grp.mcp.net", "grp.scad.net",
                   "sparse.group.lasso")
system.time(res <- oem(x, y, 
                       penalty = grp.penalties,
                       groups  = groups,
                       alpha   = 0.25, # mixing param for l2 penalties
                       tau     = 0.5)) # mixing param for sparse grp lasso 
```

    ##    user  system elapsed 
    ##    3.26    0.22    3.54

### Fitting Multiple Penalties

The oem algorithm is quite efficient at fitting multiple penalties
simultaneously when p is not too big.

``` r
set.seed(123)
n <- 100000
p <- 100
m <- 15
b <- matrix(c(runif(m, -0.25, 0.25), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)
y <- drop(x %*% b) + rnorm(n)

microbenchmark(
    "oem[lasso]"    = {res1 <- oem(x, y,
                                   penalty = "elastic.net",
                                   intercept = TRUE, 
                                   standardize = TRUE,
                                   tol = 1e-10)},
    "oem[lasso/mcp/scad/ols]"    = {res2 <- oem(x, y,
                                   penalty = c("elastic.net", "mcp", 
                                               "scad", "grp.lasso", 
                                               "grp.mcp", "sparse.grp.lasso",
                                               "grp.mcp.net", "mcp.net"),
                                   gamma = 4,
                                   tau = 0.5,
                                   alpha = 0.25,
                                   groups = rep(1:10, each = 10),
                                   intercept = TRUE, 
                                   standardize = TRUE,
                                   tol = 1e-10)},
    times = 5
)
```

    ## Unit: milliseconds
    ##                     expr      min       lq     mean   median       uq
    ##               oem[lasso] 209.7527 209.8486 225.8205 215.7640 216.5487
    ##  oem[lasso/mcp/scad/ols] 259.0799 263.1141 274.3198 265.2685 286.3861
    ##       max neval
    ##  277.1885     5
    ##  297.7507     5

``` r
#png("../mcp_path.png", width = 3000, height = 3000, res = 400);par(mar=c(5.1,5.1,4.1,2.1));plot(res2, which.model = 2, main = "mcp",lwd = 3,cex.axis=2.0, cex.lab=2.0, cex.main=2.0, cex.sub=2.0);dev.off()
#

layout(matrix(1:4, ncol=2, byrow = TRUE))
plot(res2, which.model = 1, lwd = 2,
     xvar = "lambda")
plot(res2, which.model = 2, lwd = 2,
     xvar = "lambda")
plot(res2, which.model = 4, lwd = 2,
     xvar = "lambda")
plot(res2, which.model = 7, lwd = 2,
     xvar = "lambda")
```

<img src="vignettes/mult-1.png" style="display: block; margin: auto;" />
