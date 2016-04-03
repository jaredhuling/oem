





## Introduction

The oem package provides estimaton for various penalized linear models using the [Orthogonalizing EM algorithm](http://amstat.tandfonline.com/doi/abs/10.1080/00401706.2015.1054436). 

Install using the **devtools** package:

```r
#devtools::install_github("jaredhuling/oem")
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

lambdas = oem(x, y, intercept = FALSE, standardize = FALSE)$lambda

microbenchmark(
    "glmnet[lasso]" = {res1 <- glmnet(x, y, thresh = 1e-10, # thresh must be very low for glmnet to be accurate
                                      standardize = FALSE,
                                      intercept = FALSE,
                                      lambda = lambdas)}, 
    "oem[lasso]"    = {res2 <- oem(x, y,
                                   penalty = "elastic.net",
                                   intercept = FALSE, 
                                   standardize = FALSE,
                                   lambda = lambdas,
                                   tol = 1e-10)},
    times = 5
)
```

```
## Unit: seconds
##           expr      min       lq     mean   median       uq      max neval
##  glmnet[lasso] 6.381488 6.481014 7.171990 6.993996 7.982421 8.021032     5
##     oem[lasso] 2.023880 2.083363 2.365171 2.377320 2.624369 2.716922     5
##  cld
##    b
##   a
```

```r
# difference of results
max(abs(coef(res1) - res2$beta[[1]]))
```

```
## [1] 1.037584e-07
```

### MCP


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


microbenchmark(
    "sparsenet[mcp]" = {res1 <- sparsenet(x, y, thresh = 1e-10, # thresh must be very low for glmnet to be accurate
                                          gamma = c(2,3), #sparsenet throws an error if you only fit 1 value of gamma
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
                                   nlambda = 200,
                                   eps = 1e-8)}, 
    "plus[mcp]"    = {res4 <- plus(x, y,  
                                   method = "mc+",
                                   gamma = 2,
                                   lam = res2$lambda,
                                   eps = 1e-10)},
    times = 5
)
```


```
## Unit: milliseconds
##            expr      min        lq      mean    median       uq       max
##  sparsenet[mcp] 1535.880 1551.5570 1605.5377 1564.9102 1637.944 1737.3974
##        oem[mcp]  133.719  136.9696  140.4288  137.7298  145.569  148.1564
##     ncvreg[mcp] 6178.016 6258.3558 6304.6958 6274.2707 6374.116 6438.7197
##       plus[mcp] 1479.134 1556.9671 1601.5719 1627.2505 1651.201 1693.3071
##  neval cld
##      5  b 
##      5 a  
##      5   c
##      5  b
```



