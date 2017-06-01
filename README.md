
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

lambdas = oem(x, y, intercept = TRUE, standardize = FALSE, penalty = "elastic.net")$lambda

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
##  glmnet[lasso] 7.575130 7.647406 8.185526 8.134984 8.448356 9.121753     5
##     oem[lasso] 2.048697 2.237141 2.296096 2.245134 2.273289 2.676219     5
```

```r
# difference of results
max(abs(coef(res1) - res2$beta[[1]]))
```

```
## [1] 1.048072e-07
```

```r
res1 <- glmnet(x, y, thresh = 1e-12, # thresh must be very low for glmnet to be accurate
                                      standardize = FALSE,
                                      intercept = TRUE,
                                      lambda = lambdas)

max(abs(coef(res1) - res2$beta[[1]]))
```

```
## [1] 3.763398e-09
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
               nlambda = 200, tol = 1e-10)$lambda

scad.lam <- oem(x, y, penalty = "scad",
               gamma = 4, intercept = TRUE, 
               standardize = TRUE,
               nlambda = 200, tol = 1e-10)$lambda

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
##            expr       min        lq      mean    median        uq
##  sparsenet[mcp] 1772.2340 1783.1975 1857.9591 1835.0776 1904.5578
##        oem[mcp]  156.1389  156.6702  172.6061  157.2951  159.4313
##     ncvreg[mcp] 7053.6730 7127.6700 7184.7262 7156.3276 7206.7456
##       plus[mcp] 1600.8701 1708.5586 1759.7027 1778.9716 1837.7239
##       oem[scad]  131.9789  133.0584  141.9454  137.8785  140.9257
##    ncvreg[scad] 7583.0249 7632.4411 7727.6092 7641.8928 7802.3927
##      plus[scad] 1750.1372 1757.9950 1999.7647 2043.7093 2101.0122
##        max neval
##  1994.7284     5
##   233.4951     5
##  7379.2148     5
##  1872.3896     5
##   165.8855     5
##  7978.2944     5
##  2345.9697     5
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
## MCP:  oem and plus   2.684108e-11
## SCAD: oem and plus   1.732414e-11
```



### Group Lasso


```r
library(gglasso)
library(grpreg)
library(grplasso)
# compute the full solution path, n > p
set.seed(123)
n <- 5000
p <- 200
m <- 25
b <- matrix(c(runif(m, -0.5, 0.5), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)
y <- drop(x %*% b) + rnorm(n)
groups <- rep(1:floor(p/10), each = 10)

grp.lam <- oem(x, y, penalty = "grp.lasso",
               groups = groups,
               nlambda = 100, tol = 1e-10)$lambda


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
##                 expr        min         lq       mean     median
##   gglasso[grp.lasso] 1871.23145 1947.55818 2011.37105 1981.32627
##       oem[grp.lasso]   84.79955   85.49979   92.73577   91.10412
##  grplasso[grp.lasso] 4319.38172 4683.59374 4885.40948 5011.62115
##    grpreg[grp.lasso] 1161.21075 1181.88023 1283.63716 1307.33189
##          uq       max neval
##  2055.28069 2201.4587     5
##    98.26249  104.0129     5
##  5162.37125 5250.0795     5
##  1332.52493 1435.2380     5
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
## oem and gglasso  1.729379e-06
## oem and grplasso 4.828369e-08
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

system.time(res <- oem(x, y, penalty = "grp.lasso",
                       groups = groups,
                       standardize = TRUE,
                       intercept = TRUE,
                       nlambda = 100, tol = 1e-10))
```

```
##    user  system elapsed 
##    3.29    0.16    3.48
```

```r
# memory usage is out of control here.
# oem uses approximately 1/3 of the memory
system.time(res2 <- grpreg(x, y, group = groups, 
                           eps = 1e-10, lambda = res$lambda))
```

```
##    user  system elapsed 
##   71.49    2.03   80.37
```

```r
# I think the standardization is done
# differently for grpreg
max(abs(res$beta[[1]] - res2$beta))
```

```
## [1] 0.0007842304
```

```r
mean(abs(res$beta[[1]] - res2$beta))
```

```
## [1] 8.363514e-06
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
                                   penalty = c("elastic.net", "mcp", "scad", "grp.lasso"),
                                   gamma = 4,
                                   groups = rep(1:10, each = 10),
                                   intercept = TRUE, 
                                   standardize = TRUE,
                                   tol = 1e-10)},
    times = 5
)
```

```
## Unit: milliseconds
##                     expr      min       lq     mean  median       uq
##               oem[lasso] 258.9629 272.3903 317.1007 300.378 363.5564
##  oem[lasso/mcp/scad/ols] 260.3768 267.3488 279.7942 281.632 287.8948
##       max neval
##  390.2162     5
##  301.7185     5
```

```r
#png("../mcp_path.png", width = 3000, height = 3000, res = 400);par(mar=c(5.1,5.1,4.1,2.1));plot(res2, which.model = 2, main = "mcp",lwd = 3,cex.axis=2.0, cex.lab=2.0, cex.main=2.0, cex.sub=2.0);dev.off()
#

layout(matrix(1:4, ncol=2, byrow = TRUE))
plot(res2, which.model = 1, lwd = 2,
     xvar = "lambda")
plot(res2, which.model = 2, lwd = 2,
     xvar = "lambda")
plot(res2, which.model = 3, lwd = 2,
     xvar = "lambda")
plot(res2, which.model = 4, lwd = 2,
     xvar = "lambda")
```

<img src="vignettes/mult-1.png" style="display: block; margin: auto;" />
