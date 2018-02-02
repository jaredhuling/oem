---
output:
  html_document:
    keep_md: yes
    self_contained: true
---

[![version](http://www.r-pkg.org/badges/version/oem)](https://cran.r-project.org/package=oem)

### Build Status
|  OS             | Build           |
|-----------------|-----------------|
| Linux x86_64 / OSX   | [![Build Status](https://travis-ci.org/jaredhuling/oem.svg?branch=master)](https://travis-ci.org/jaredhuling/oem)      | 
| Windows x86_64     | [![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/github/jaredhuling/oem?branch=master&svg=true)](https://ci.appveyor.com/project/jaredhuling/oem)     |










## Introduction

The oem package provides estimaton for various penalized linear models using the [Orthogonalizing EM algorithm](http://amstat.tandfonline.com/doi/abs/10.1080/00401706.2015.1054436). Documentation for the package can be found here: [oem site](http://casualinference.org/oem) (still under construction).

Install using the **devtools** package (RcppEigen must be installed first as well):


```r
devtools::install_github("jaredhuling/oem")
```

or by cloning and building using `R CMD INSTALL`

## Models

### Lasso


```r
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

```
## Unit: seconds
##           expr      min       lq     mean   median       uq      max neval
##  glmnet[lasso] 8.454950 8.925667 9.183843 9.255825 9.351194 9.931580     5
##     oem[lasso] 2.456566 2.640007 2.754852 2.881469 2.884303 2.911914     5
```

```r
# difference of results
max(abs(coef(res1) - res2$beta[[1]]))
```

```
## [1] 1.048072e-07
```

```r
res1 <- glmnet(x, y, thresh = 1e-12, 
               standardize = FALSE,
               intercept = TRUE,
               lambda = lambdas)

# answers are now more close once we require more precise glmnet solutions
max(abs(coef(res1) - res2$beta[[1]]))
```

```
## [1] 3.763397e-09
```

### Nonconvex Penalties


```r
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

```
## Warning message: some lam not reached by the plus path and dropped
## Warning message: some lam not reached by the plus path and dropped
## Warning message: some lam not reached by the plus path and dropped
## Warning message: some lam not reached by the plus path and dropped
## Warning message: some lam not reached by the plus path and dropped
## Warning message: some lam not reached by the plus path and dropped
## Warning message: some lam not reached by the plus path and dropped
## Warning message: some lam not reached by the plus path and dropped
## Warning message: some lam not reached by the plus path and dropped
## Warning message: some lam not reached by the plus path and dropped
```

```
## Unit: milliseconds
##            expr       min        lq      mean    median         uq
##  sparsenet[mcp] 2320.8100 2352.9517 2458.7722 2392.4679  2517.0486
##        oem[mcp]  189.3625  212.6236  233.1697  226.6125   267.3957
##     ncvreg[mcp] 8129.6092 8555.1828 9115.0166 8886.1990  9958.1984
##       plus[mcp] 2099.2994 2500.4747 2554.6303 2589.2470  2756.9943
##       oem[scad]  137.4814  139.5355  184.2794  187.5250   220.8466
##    ncvreg[scad] 8808.6070 9585.5057 9916.8835 9804.8729 10375.9077
##      plus[scad] 2304.6635 2496.2100 2617.9396 2539.4899  2831.3368
##         max neval
##   2710.5826     5
##    269.8545     5
##  10045.8938     5
##   2827.1361     5
##    236.0083     5
##  11009.5241     5
##   2917.9977     5
```

```r
diffs <- array(NA, dim = c(4, 1))
colnames(diffs) <- "abs diff"
rownames(diffs) <- c("MCP:  oem and ncvreg", "SCAD: oem and ncvreg",
                     "MCP:  oem and plus", "SCAD: oem and plus")
diffs[,1] <- c(max(abs(res2$beta[[1]] - res3$beta)), max(abs(res5$beta[[1]] - res6$beta)),
               max(abs(res2$beta[[1]][-1,1:nrow(res4$beta)] - t(res4$beta))),
               max(abs(res5$beta[[1]][-1,1:nrow(res7$beta)] - t(res7$beta))))
diffs
```

```
##                          abs diff
## MCP:  oem and ncvreg 1.725859e-07
## SCAD: oem and ncvreg 5.094648e-08
## MCP:  oem and plus   2.684136e-11
## SCAD: oem and plus   1.732409e-11
```



### Group Lasso


```r
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

```
## Unit: milliseconds
##                 expr       min        lq     mean    median        uq
##   gglasso[grp.lasso] 4004.4774 4086.1746 4108.586 4103.6553 4126.2035
##       oem[grp.lasso]  112.8335  114.3252  125.083  118.6199  120.8217
##  grplasso[grp.lasso] 7816.4330 8224.7857 9044.954 8851.7595 9250.1811
##    grpreg[grp.lasso] 2253.6705 2319.5918 2431.038 2379.3103 2565.8449
##         max neval
##   4222.4212     5
##    158.8146     5
##  11081.6107     5
##   2636.7702     5
```

```r
diffs <- array(NA, dim = c(2, 1))
colnames(diffs) <- "abs diff"
rownames(diffs) <- c("oem and gglasso", "oem and grplasso")
diffs[,1] <- c(  max(abs(res2$beta[[1]][-1,] - res1$beta)), max(abs(res2$beta[[1]][-1,] - res3$coefficients))  )
diffs
```

```
##                      abs diff
## oem and gglasso  1.303906e-06
## oem and grplasso 1.645871e-08
```

#### Bigger Group Lasso Example


```r
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

```
##    user  system elapsed 
##    3.62    0.17    4.03
```

### Fitting Multiple Penalties

The oem algorithm is quite efficient at fitting multiple penalties simultaneously when p is not too big.


```r
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

```
## Unit: milliseconds
##                     expr      min       lq     mean   median       uq
##               oem[lasso] 213.7904 219.6538 226.4532 219.8930 237.8327
##  oem[lasso/mcp/scad/ols] 259.2121 260.7772 267.5143 268.8317 271.7456
##       max neval
##  241.0959     5
##  277.0047     5
```

```r
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
