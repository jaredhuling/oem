
#' Orthogonalizing EM
#'
#' @param x input matrix or SparseMatrix object. 
#' Each row is an observation, each column corresponds to a covariate
#' @param y numeric response vector of length nobs.
#' @param family "gaussian" for least squares problems, "binomial" for binary response. 
#' @param penalty Specification of penalty type in lowercase letters. Choices include "lasso", 
#' "ols" (Ordinary least squares, no penaly), "elastic.net", "scad", "mcp", "grp.lasso"
#' @param weights observation weights. Not implemented yet. Defaults to 1 for each observation (setting weight vector to 
#' length 0 will default all weights to 1)
#' @param lambda A user supplied lambda sequence. By default, the program computes
#' its own lambda sequence based on nlambda and lambda.min.ratio. Supplying
#' a value of lambda overrides this.
#' @param nlambda The number of lambda values - default is 100.
#' @param lambda.min.ratio Smallest value for lambda, as a fraction of lambda.max, the (data derived) entry
#' value (i.e. the smallest value for which all coefficients are zero). The default
#' depends on the sample size nobs relative to the number of variables nvars. If
#' nobs > nvars, the default is 0.0001, close to zero. If nobs < nvars, the default
#' is 0.01. A very small value of lambda.min.ratio will lead to a saturated fit
#' when nobs < nvars.
#' @param alpha mixing value for elastic.net. penalty applied is (1 - alpha) * (ridge penalty) + alpha * (lasso penalty)
#' @param gamma tuning parameter for SCAD and MCP penalties. must be >= 1
#' @param groups A vector of describing the grouping of the coefficients. See the example below. All unpenalized variables
#' should be put in group 0
#' @param penalty.factor Separate penalty factors can be applied to each coefficient. 
#' This is a number that multiplies lambda to allow differential shrinkage. Can be 0 for some variables, 
#' which implies no shrinkage, and that variable is always included in the model. Default is 1 for all 
#' variables. 
#' @param group.weights penalty factors applied to each group for the group lasso. Similar to penalty.factor, 
#' this is a number that multiplies lambda to allow differential shrinkage. Can be 0 for some groups, 
#' which implies no shrinkage, and that group is always included in the model. Default is sqrt(group size) for all
#' groups. 
#' @param standardize Logical flag for x variable standardization, prior to fitting the models. 
#' The coefficients are always returned on the original scale. Default is standardize=FALSE. If 
#' variables are in the same units already, you might not wish to standardize. Keep in mind that 
#' standardization is done differently for sparse matrices, so results (when standardized) may be
#' slightly different for a sparse matrix object and a dense matrix object
#' @param intercept Should intercept(s) be fitted (default=TRUE) or set to zero (FALSE)
#' @param maxit integer. Maximum number of OEM iterations
#' @param tol convergence tolerance for OEM iterations
#' @param irls.maxit integer. Maximum number of IRLS iterations
#' @param irls.tol convergence tolerance for IRLS iterations. Only used if family != "gaussian"
#' @param ncores Integer scalar that specifies the number of threads to be used
#' @param compute.loss should the loss be computed for each estimated tuning parameter? Defaults to FALSE. Setting
#' to TRUE will dramatically increase computational time
#' @param hessian.type only for logistic regression. if hessian.type = "full", then the full hessian is used. If
#' hessian.type = "upper.bound", then an upper bound of the hessian is used. The upper bound can be dramatically
#' faster in certain situations, ie when n >> p
#' @return An object with S3 class "oemfit" 
#' @useDynLib oem
#' @import Rcpp
#' @import Matrix
#' @import foreach
#' @export
#' @examples
#' set.seed(123)
#' n.obs <- 1e4
#' n.vars <- 100
#' 
#' true.beta <- c(runif(15, -0.25, 0.25), rep(0, n.vars - 15))
#' 
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + x %*% true.beta
#' 
#' fit <- oem(x = x, y = y, 
#'            penalty = c("lasso", "grp.lasso"), 
#'            groups = rep(1:20, each = 5))
#' 
#' layout(matrix(1:2, ncol = 2))
#' plot(fit)
#' plot(fit, which.model = 2)
#' 
#' # the oem package has support for
#' # sparse design matrices
#' 
#' library(Matrix)
#' 
#' xs <- rsparsematrix(n.obs * 25, n.vars * 2, density = 0.01)
#' ys <- rnorm(n.obs * 25, sd = 3) + as.vector(xs %*% c(true.beta, rep(0, n.vars)) )
#' x.dense <- as.matrix(xs)
#' 
#' system.time(fit <- oem(x = x.dense, y = ys, 
#'                        penalty = c("lasso", "grp.lasso"), 
#'                        groups = rep(1:40, each = 5), intercept = TRUE))
#' 
#' system.time(fits <- oem(x = xs, y = ys, 
#'                         penalty = c("lasso", "grp.lasso"), 
#'                         groups = rep(1:40, each = 5), intercept = TRUE, lambda = fit$lambda))
#'                         
#' max(abs(fit$beta[[1]] - fits$beta[[1]]))
#' max(abs(fit$beta[[2]] - fits$beta[[2]]))
#' 
#' # logistic
#' y <- rbinom(n.obs, 1, prob = 1 / (1 + exp(-x %*% true.beta)))
#' 
#' system.time(res <- oem(x, y, intercept = FALSE, 
#'                        penalty = "lasso", 
#'                        family = "binomial", 
#'                        irls.tol = 1e-3, tol = 1e-8))
#' 
#' system.time(res.gr <- oem(x, y, intercept = FALSE, 
#'                           penalty = "grp.lasso", 
#'                           family = "binomial", 
#'                           groups = rep(1:10, each = 10), 
#'                           irls.tol = 1e-3, tol = 1e-8))
#' 
#' layout(matrix(1:2, ncol = 2))
#' plot(res)
#' plot(res.gr)
#' 
#' 
#' # sparse design matrix
#' xs <- rsparsematrix(n.obs * 5, n.vars, density = 0.01)
#' x.dense <- as.matrix(xs)
#' ys <- rbinom(n.obs * 5, 1, prob = 1 / (1 + exp(-x %*% true.beta)))
#' 
#' system.time(res.gr <- oem(x.dense, ys, intercept = FALSE, 
#'                           penalty = "grp.lasso", 
#'                           family = "binomial", 
#'                           groups = rep(1:10, each = 10), 
#'                           irls.tol = 1e-3, tol = 1e-8))
#'                           
#' system.time(res.gr.s <- oem(xs, ys, intercept = FALSE, 
#'                             penalty = "grp.lasso", 
#'                             family = "binomial", 
#'                             groups = rep(1:10, each = 10), 
#'                             irls.tol = 1e-3, tol = 1e-8))
#'                             
#' max(abs(res.gr$beta[[1]] - res.gr.s$beta[[1]]))
#' 
oem <- function(x, 
                y, 
                family = c("gaussian", "binomial"),
                penalty = c("elastic.net", "lasso", "ols", "mcp", "scad", "grp.lasso"),
                weights = numeric(0),
                lambda = numeric(0),
                nlambda = 100L,
                lambda.min.ratio = NULL,
                alpha = 1,
                gamma = 3,
                groups = numeric(0),
                penalty.factor = NULL,
                group.weights = NULL,
                standardize = TRUE,
                intercept = TRUE,
                maxit = 500L, 
                tol = 1e-7,
                irls.maxit = 100L,
                irls.tol = 1e-3,
                ncores = -1,
                compute.loss = FALSE,
                hessian.type = c("full", "upper.bound")) 
{
    family       <- match.arg(family)
    penalty      <- match.arg(penalty, several.ok = TRUE)
    hessian.type <- match.arg(hessian.type)
    
    dims <- dim(x)
    n <- dims[1]
    p <- dims[2]
    y <- drop(y)
    y.vals <- unique(y)
    is.sparse <- FALSE
    if(inherits(x, "sparseMatrix")){
        ##Sparse case
        is.sparse <- TRUE
        x <- as(x,"CsparseMatrix")
        x <- as(x,"dgCMatrix")
    }
    
    if (length(weights) > 0) stop("weights not implemented yet.")
    
    if (length(y) != n) {
        stop("x and y lengths do not match")
    }
    
    if (family == "binomial" & length(y.vals) > 2) {
        stop("y must be a binary outcome")
    }
    
    if (is.null(penalty.factor)) {
        penalty.factor <- rep(1, p)
    }
    
    varnames <- colnames(x)
    if(is.null(varnames)) varnames = paste("V", seq(p), sep="")
    
    if (length(weights))
    {
        if (length(weights) != n)
        {
            stop("length of weights not same as number of observations in x")
        }
        
    }
    
    if (n/ncores < 5)
    {
        ncores <- max(1, floor(n/ncores))
    }
    
    if (length(penalty.factor) != p) {
        stop("penalty.factor must have same length as number of columns in x")
    }
    penalty.factor <- drop(penalty.factor)
    
    if (any(penalty == "grp.lasso")) {
        if (length(groups) != p) {
            stop("groups must have same length as number of columns in x")
        }
        
        unique.groups <- sort(unique(groups))
        zero.idx <- unique.groups[which(unique.groups == 0)]
        groups <- drop(groups)
        if (!is.null(group.weights))
        {
            
            if (length(zero.idx) > 0)
            {
                # force group weight for 0 group to be zero
                group.weights[zero.idx] <- 0
            } else 
            {
                if ((intercept & family != "gaussian") |
                    (intercept & is.sparse))
                {
                    ## add group for zero term if it's not here
                    ## and add penalty weight of zero
                    unique.groups <- c(0, unique.groups)
                    group.weights <- c(0, group.weights)
                }
            }
            group.weights <- drop(group.weights)
            if (length(group.weights) != length(unique.groups)) {
                stop("group.weights must have same length as the number of groups")
            }
            group.weights <- as.numeric(group.weights)
        } else {
            # default to sqrt(group size) for each group weight
            group.weights <- numeric(0)
            
            if (length(zero.idx) == 0)
            {
                if ((intercept & family != "gaussian") |
                    (intercept & is.sparse))
                {
                    ## add group for zero term if it's not here
                    unique.groups <- sort(c(0, unique.groups))
                }
            }
        }
        
        
        
        if ((intercept & family != "gaussian") |
            (intercept & is.sparse))
        {
            ## add intercept to group with no penalty
            groups <- c(0, groups)
        }
        
        
    } else {
        unique.groups <- numeric(0)
        group.weights <- numeric(0)
    }
    
    
    if (is.null(lambda.min.ratio)) {
        lambda.min.ratio <- ifelse(nrow(x) < ncol(x), 0.01, 0.0001)
    } else {
        lambda.min.ratio <- as.numeric(lambda.min.ratio)
    }
    
    if(lambda.min.ratio >= 1 | lambda.min.ratio <= 0) 
    {
        stop("lambda.min.ratio must be between 0 and 1")
    }
    
    if(nlambda[1] <= 0) 
    {
        stop("nlambda must be a positive integer")
    }
    
    lambda <- sort(as.numeric(lambda), decreasing = TRUE)
    

        
    ##    ensure types are correct
    ##    before sending to c++
    
    if (length(lambda) > 0)
    {
        lambda    <- as.double(lambda)
    }
    groups        <- as.integer(groups)
    unique.groups <- as.integer(unique.groups)
    nlambda       <- as.integer(nlambda)
    alpha         <- as.double(alpha)
    gamma         <- as.double(gamma)
    tol           <- as.double(tol)
    irls.tol      <- as.double(irls.tol)
    irls.maxit    <- as.integer(irls.maxit)
    maxit         <- as.integer(maxit)
    standardize   <- as.logical(standardize)
    intercept     <- as.logical(intercept)
    compute.loss  <- as.logical(compute.loss)
    ncores        <- as.integer(ncores[1])
    
    if(maxit <= 0 | irls.maxit <= 0)
    {
        stop("maxit and irls.maxit should be positive")
    }
    if(tol < 0 | irls.tol < 0)
    {
        stop("tol and irls.tol should be nonnegative")
    }
    
    
    options <- list(maxit        = maxit,
                    tol          = tol,
                    irls_maxit   = irls.maxit,
                    irls_tol     = irls.tol,
                    ncores       = ncores,
                    hessian.type = hessian.type)
    
    res <- switch(family,
                  "gaussian" = oemfit.gaussian(is.sparse,
                                               x, y, 
                                               family, 
                                               penalty, 
                                               weights,
                                               groups,
                                               unique.groups,
                                               group.weights,
                                               lambda, 
                                               nlambda,
                                               lambda.min.ratio,
                                               alpha,
                                               gamma,
                                               penalty.factor,
                                               standardize,
                                               intercept,
                                               compute.loss,
                                               options),
                  "binomial" = oemfit.binomial(is.sparse, 
                                               x, y, 
                                               family, 
                                               penalty, 
                                               weights,
                                               groups,
                                               unique.groups,
                                               group.weights,
                                               lambda, 
                                               nlambda,
                                               lambda.min.ratio,
                                               alpha,
                                               gamma,
                                               penalty.factor,
                                               standardize,
                                               intercept,
                                               compute.loss,
                                               options)
                  )
    
    for (i in 1:length(penalty))
    {
        if (penalty[i] == "ols") res$beta[[i]] <- matrix(res$beta[[i]], ncol = 1)
        rownames(res$beta[[i]]) <- c("(Intercept)", varnames)
    }
    
    nz <- lapply(1:length(res$beta), function(m) 
        sapply(predict.oemfit(res, type = "nonzero", which.model = m), length) - 1
    )
    
    res$nobs     <- n
    res$nvars    <- p
    res$penalty  <- penalty
    res$family   <- family
    res$varnames <- varnames
    res$nzero    <- nz
    
    class(res)   <- c(class(res), "oemfit")
    res
}


oemfit.gaussian <- function(is.sparse, 
                            x, 
                            y, 
                            family, 
                            penalty, 
                            weights,
                            groups,
                            unique.groups,
                            group.weights,
                            lambda, 
                            nlambda,
                            lambda.min.ratio,
                            alpha,
                            gamma,
                            penalty.factor,
                            standardize,
                            intercept,
                            compute.loss,
                            options)
{
    if (is.sparse)
    {
        ret <- .Call("oem_fit_sparse", 
                     x, y, 
                     family, 
                     penalty, 
                     weights,
                     groups,
                     unique.groups,
                     group.weights,
                     lambda, 
                     nlambda,
                     lambda.min.ratio,
                     alpha,
                     gamma,
                     penalty.factor,
                     standardize,
                     intercept,
                     compute.loss,
                     options,
                     PACKAGE = "oem")
    } else 
    {
        ret <- .Call("oem_fit_dense", 
                     x, y, 
                     family, 
                     penalty, 
                     weights,
                     groups,
                     unique.groups,
                     group.weights,
                     lambda, 
                     nlambda,
                     lambda.min.ratio,
                     alpha,
                     gamma,
                     penalty.factor,
                     standardize,
                     intercept,
                     compute.loss,
                     options,
                     PACKAGE = "oem")
    }
    class(ret) <- "oemfit_gaussian"
    ret
}


oemfit.binomial <- function(is.sparse, 
                            x, 
                            y, 
                            family, 
                            penalty, 
                            weights,
                            groups,
                            unique.groups,
                            group.weights,
                            lambda, 
                            nlambda,
                            lambda.min.ratio,
                            alpha,
                            gamma,
                            penalty.factor,
                            standardize,
                            intercept,
                            compute.loss,
                            options)
{
    if (is.sparse)
    {
        ret <- .Call("oem_fit_logistic_sparse", 
                     x, y, 
                     family, 
                     penalty,
                     weights,
                     groups,
                     unique.groups,
                     group.weights,
                     lambda, 
                     nlambda,
                     lambda.min.ratio,
                     alpha,
                     gamma,
                     penalty.factor,
                     standardize,
                     intercept,
                     compute.loss,
                     options,
                     PACKAGE = "oem")
    } else 
    {
        ret <- .Call("oem_fit_logistic_dense", 
                     x, y, 
                     family, 
                     penalty, 
                     weights,
                     groups,
                     unique.groups,
                     group.weights,
                     lambda, 
                     nlambda,
                     lambda.min.ratio,
                     alpha,
                     gamma,
                     penalty.factor,
                     standardize,
                     intercept,
                     compute.loss,
                     options,
                     PACKAGE = "oem")
    }
    class(ret) <- "oemfit_binomial"
    ret
}



