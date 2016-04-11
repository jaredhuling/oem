
#' Orthogonalizing EM
#'
#' @param x input matrix or SparseMatrix (sparse not yet implemented. 
#' Each row is an observation, each column corresponds to a covariate
#' @param y numeric response vector of length nobs.
#' @param family "gaussian" for least squares problems, "binomial" for binary response. 
#' @param penalty Specification of penalty type in lowercase letters. Choices include "lasso", 
#' "ols" (Ordinary least squares, no penaly), "elastic.net", "scad", "mcp"
#' @param lambda A user supplied lambda sequence. By default, the program computes
#' its own lambda sequence based on nlambda and lambda.min.ratio. Supplying
#' a value of lambda overrides this.
#' @param nlambda The number of lambda values - default is 100.
#' @param lambda.min.ratio Smallest value for lambda, as a fraction of lambda.max, the (data derived) entry
#' value (i.e. the smallest value for which all coefﬁcients are zero). The default
#' depends on the sample size nobs relative to the number of variables nvars. If
#' nobs > nvars, the default is 0.0001, close to zero. If nobs < nvars, the default
#' is 0.01. A very small value of lambda.min.ratio will lead to a saturated fit
#' when nobs < nvars.
#' @param alpha mixing value for elastic.net. penalty applied is (1 - alpha) * (ridge penalty) + alpha * (lasso penalty)
#' @param gamma tuning parameter for SCAD and MCP penalties
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
#' @standardize Logical flag for x variable standardization, prior to fitting the model sequence. 
#' The coefficients are always returned on the original scale. Default is standardize=FALSE. If 
#' variables are in the same units already, you might not wish to standardize. 
#' @intercept Should intercept(s) be fitted (default=TRUE) or set to zero (FALSE)
#' @param maxit integer. Maximum number of OEM iterations
#' @param tol convergence tolerance for OEM iterations
#' @param irls.maxit integer. Maximum number of IRLS iterations
#' @param irls.tol convergence tolerance for IRLS iterations. Only used if family != "gaussian"
#' @return An object with S3 class "oemfit" 
#' @useDynLib oem
#' @import Rcpp
#' @exportPattern "^[[:alpha:]]+"
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
#' fit <- oem(x = x, y = y, penalty = c("lasso", "grp.lasso"), groups = rep(1:20, each = 5))
#' 
#' layout(matrix(1:2, ncol = 2))
#' plot(fit)
#' plot(fit, which.model = 2)
#' 
#' # logistic
#' y <- rbinom(n.obs, 1, prob = 1 / (1 + exp(-x %*% true.beta)))
#' 
#' system.time(res <- oem(x, y, intercept = FALSE, penalty = "lasso", family = "binomial", irls.tol = 1e-3, tol = 1e-8))
#' 
#' library(glmnet)
#' 
#' system.time(glmn <- glmnet(x, y, lambda = res$lambda, standardize =FALSE, intercept = FALSE, family = "binomial", thresh = 1e-12))
#' 
#' max(abs(coef(glmn) - res$beta[[1]]))
#' 
#' system.time(glmn <- glmnet(x, y, lambda = res$lambda, standardize =FALSE, intercept = FALSE, family = "binomial", thresh = 1e-15))
#' 
#' max(abs(coef(glmn) - res$beta[[1]]))
#' 
#' system.time(glmn <- glmnet(x, y, lambda = res$lambda, standardize =FALSE, intercept = FALSE, family = "binomial", thresh = 1e-18))
#'
#' ## group lasso (logistic model)
#' 
#' library(gglasso)
#' 
#' system.time(res.gr <- oem(x, y, intercept = FALSE, penalty = "grp.lasso", family = "binomial", irls.tol = 1e-3, tol = 1e-8, groups = rep(1:10, each = 10)))
#' 
#' layout(matrix(1:2, ncol = 2))
#' plot(res)
#' plot(res.gr)
#' 
#' system.time(ggl <- gglasso(x, 2 * y - 1, group = rep(1:10, each = 10), loss = "logit", lambda = res.gr$lambda, intercept = FALSE, eps = 1e-8))
#' max(abs(ggl$beta - res.gr$beta[[1]][-1,]))
#' 
#' system.time(ggl <- gglasso(x, 2 * y - 1, group = rep(1:10, each = 10), loss = "logit", lambda = res.gr$lambda, intercept = FALSE, eps = 1e-10))
#' max(abs(ggl$beta - res.gr$beta[[1]][-1,]))
#' 
#' system.time(ggl <- gglasso(x, 2 * y - 1, group = rep(1:10, each = 10), loss = "logit", lambda = res.gr$lambda, intercept = FALSE, eps = 1e-12))
#' max(abs(ggl$beta - res.gr$beta[[1]][-1,]))
#' 
oem <- function(x, 
                y, 
                family = c("gaussian", "binomial"),
                penalty = c("elastic.net", "lasso", "ols", "mcp", "scad", "grp.lasso"),
                lambda = numeric(0),
                nlambda = 100L,
                lambda.min.ratio = NULL,
                alpha = 1,
                gamma = 3,
                groups = numeric(0),
                penalty.factor = NULL,
                group.weights = NULL,
                standardize = FALSE,
                intercept = TRUE,
                maxit = 500L, 
                tol = 1e-7,
                irls.maxit = 100L,
                irls.tol = 1e-3) 
{
    family  <- match.arg(family)
    penalty <- match.arg(penalty, several.ok = TRUE)
    
    dims <- dim(x)
    n <- dims[1]
    p <- dims[2]
    y <- drop(y)
    y.vals <- unique(y)
    is.sparse <- FALSE
    if(inherits(x, "sparseMatrix")){##Sparse case
        is.sparse <- TRUE
        x <- as(x,"CsparseMatrix")
        x <- as(x,"dgCMatrix")
    }
    
    if (length(y) != n) {
        stop("x and y lengths do not match")
    }
    
    if (family == "binomial" & length(y.vals) > 2) {
        stop("y must be a binary outcome")
    }
    
    if (is.null(penalty.factor)) {
        penalty.factor <- rep(1, p)
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
                if (intercept & family != "gaussian")
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
                if (intercept & family != "gaussian")
                {
                    ## add group for zero term if it's not here
                    unique.groups <- sort(c(0, unique.groups))
                }
            }
        }
        
        
        
        if (intercept & family != "gaussian")
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
    
    if (length(lambda) > 0)
        
        
    groups <- as.integer(groups)
    unique.groups <- as.integer(unique.groups)
    nlambda <- as.integer(nlambda)
    alpha <- as.double(alpha)
    gamma <- as.double(gamma)
    tol     <- as.double(tol)
    irls.tol <- as.double(irls.tol)
    irls.maxit  <- as.integer(irls.maxit)
    maxit  <- as.integer(maxit)
    standardize <- as.logical(standardize)
    intercept <- as.logical(intercept)
    
    if(maxit <= 0 | irls.maxit <= 0)
    {
        stop("maxit and irls.maxit should be positive")
    }
    if(tol < 0 | irls.tol < 0)
    {
        stop("tol and irls.tol should be nonnegative")
    }
    
    
    options <- list(maxit      = maxit,
                    tol        = tol,
                    irls_maxit = irls.maxit,
                    irls_tol   = irls.tol)
    
    res <- switch(family,
                  "gaussian" = oemfit.gaussian(is.sparse,
                                               x, y, 
                                               family, 
                                               penalty, 
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
                                               options),
                  "binomial" = oemfit.binomial(is.sparse, 
                                               x, y, 
                                               family, 
                                               penalty, 
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
                                               options)
                  )
    
    class(res) <- c(class(res), "oemfit")
    res
}


oemfit.gaussian <- function(is.sparse, 
                            x, 
                            y, 
                            family, 
                            penalty, 
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
                            options)
{
    if (is.sparse)
    {
        ret <- .Call("oem_fit_sparse", 
                     x, y, 
                     family, 
                     penalty, 
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
                     options,
                     PACKAGE = "oem")
    } else 
    {
        ret <- .Call("oem_fit_dense", 
                     x, y, 
                     family, 
                     penalty, 
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
                            options)
{
    if (is.sparse)
    {
        ret <- .Call("oem_fit_logistic_sparse", 
                     x, y, 
                     family, 
                     penalty, 
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
                     options,
                     PACKAGE = "oem")
    } else 
    {
        ret <- .Call("oem_fit_logistic_dense", 
                     x, y, 
                     family, 
                     penalty, 
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
                     options,
                     PACKAGE = "oem")
    }
    class(ret) <- "oemfit_binomial"
    ret
}



