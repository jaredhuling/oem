
### Build Status
|                 | Build           |
|-----------------|-----------------|
| Linux x86_64    | [![Build Status](https://travis-ci.org/jaredhuling/oem.svg?branch=master)](https://travis-ci.org/jaredhuling/oem)      | 
| OSX             | [![Build Status](https://travis-ci.org/jaredhuling/oem.svg?branch=master)](https://travis-ci.org/jaredhuling/oem)          |
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
##  glmnet[lasso] 7.292307 7.383917 7.658757 7.580808 7.671030 8.365722     5
##     oem[lasso] 2.040648 2.047586 2.113338 2.063423 2.078822 2.336209     5
##  cld
##    b
##   a
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
##  sparsenet[mcp] 1740.2711 1740.4014 1745.6558 1746.0448 1749.9996
##        oem[mcp]  156.1490  156.3148  157.4937  156.6029  158.7574
##     ncvreg[mcp] 8332.3995 8376.8815 8405.0775 8387.1697 8451.3359
##       plus[mcp] 1704.8241 1723.2366 1756.4895 1727.7677 1796.3541
##       oem[scad]  132.5754  132.5845  133.0697  132.6441  133.2717
##    ncvreg[scad] 8527.7561 8595.5140 8637.7294 8670.6568 8692.7131
##      plus[scad] 1886.8247 1901.8046 1982.6517 1961.2432 2041.5625
##        max neval   cld
##  1751.5619     5  b   
##   159.6443     5 a    
##  8477.6007     5    d 
##  1830.2648     5  b   
##   134.2728     5 a    
##  8702.0070     5     e
##  2121.8234     5   c
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
## MCP:  oem and ncvreg 5.134558e-10
## SCAD: oem and ncvreg 2.087087e-10
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
##   gglasso[grp.lasso] 1758.24982 1760.96713 1764.63203 1767.35226
##       oem[grp.lasso]   79.10119   79.33289   79.85101   79.80578
##  grplasso[grp.lasso] 2575.35601 2602.75647 2609.91463 2613.97859
##    grpreg[grp.lasso] 1036.92605 1041.19859 1042.27122 1041.82265
##          uq        max neval  cld
##  1767.53028 1769.06064     5   c 
##    80.06472   80.95049     5 a   
##  2623.16826 2634.31381     5    d
##  1044.23010 1047.17872     5  b
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
## oem and gglasso  8.341970e-05
## oem and grplasso 8.341973e-05
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
##    3.17    0.17    3.34
```

```r
# memory usage is out of control here.
# oem uses approximately 1/3 of the memory
system.time(res2 <- grpreg(x, y, group = groups, 
                           eps = 1e-10, lambda = res$lambda))
```

```
##    user  system elapsed 
##   73.89    1.44   75.83
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
##                     expr      min       lq     mean   median       uq
##               oem[lasso] 236.5429 236.9945 244.4077 238.2651 249.9254
##  oem[lasso/mcp/scad/ols] 249.6645 252.7753 260.2248 252.9913 265.3522
##       max neval cld
##  260.3107     5   a
##  280.3408     5   a
```

```r
#png("../mcp_path.png", width = 3000, height = 3000, res = 400);par(mar=c(5.1,5.1,4.1,2.1));plot(res2, which.model = 2, main = "mcp",lwd = 3,cex.axis=2.0, cex.lab=2.0, cex.main=2.0, cex.sub=2.0);dev.off()
#

layout(matrix(1:4, ncol=2, byrow = TRUE))
plot(res2, which.model = 1, lwd = 2)
plot(res2, which.model = 2, lwd = 2)
plot(res2, which.model = 3, lwd = 2)
plot(res2, which.model = 4, lwd = 2)
```

<img src="README_files/figure-html/mult-1.png" title="" alt="" style="display: block; margin: auto;" />
