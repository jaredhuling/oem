
#' Orthogonalizing EM with precomputed XtX
#'
#' @param xtx input matrix equal to \code{crossprod(x) / nrow(x)}. 
#' where \code{x} is the design matrix.
#' It is highly recommended to scale by the number of rows in \code{x}.
#' If \code{xtx} is scaled, \code{xty} must also be scaled or else results may be meaningless!
#' @param xty numeric vector of length \code{nvars}. Equal to \code{crosprod(x, y) / nobs}. 
#' It is highly recommended to scale by the number of rows in \code{x}.
#' @param family \code{"gaussian"} for least squares problems, \code{"binomial"} for binary response. 
#' (only \code{gaussian} implemented currently)
#' @param penalty Specification of penalty type. Choices include:
#' \itemize{
#'    \item \code{"elastic.net"} - elastic net penalty, extra parameters: \code{"alpha"}
#'    \item \code{"lasso"} - lasso penalty
#'    \item \code{"ols"} - ordinary least squares
#'    \item \code{"mcp"} - minimax concave penalty, extra parameters: \code{"gamma"}
#'    \item \code{"scad"} - smoothly clipped absolute deviation, extra parameters: \code{"gamma"}
#'    \item \code{"mcp.net"} - minimax concave penalty + l2 penalty, extra parameters: 
#'    \code{"gamma"}, \code{"alpha"}
#'    \item \code{"scad.net"} - smoothly clipped absolute deviation + l2 penalty, extra parameters: 
#'    \code{"gamma"}, \code{"alpha"}
#'    \item \code{"grp.lasso"} - group lasso penalty
#'    \item \code{"grp.lasso.net"} - group lasso penalty + l2 penalty, extra parameters: \code{"alpha"}
#'    \item \code{"grp.mcp"} - group minimax concave penalty, extra parameters: \code{"gamma"}
#'    \item \code{"grp.scad"} - group smoothly clipped absolute deviation, extra parameters: \code{"gamma"}
#'    \item \code{"grp.mcp.net"} - group minimax concave penalty + l2 penalty, extra parameters: \code{"gamma"}, \code{"alpha"}
#'    \item \code{"grp.scad.net"} - group smoothly clipped absolute deviation + l2 penalty, extra parameters: \code{"gamma"}, \code{"alpha"}
#'    \item \code{"sparse.grp.lasso"} - sparse group lasso penalty (group lasso + lasso), extra parameters: \code{"tau"}
#' }
#' Careful consideration is required for the group lasso, group MCP, and group SCAD penalties. Groups as specified by the \code{groups} argument 
#' should be chosen in a sensible manner.
#' @param lambda A user supplied lambda sequence. By default, the program computes
#' its own lambda sequence based on \code{nlambda} and \code{lambda.min.ratio}. Supplying
#' a value of lambda overrides this.
#' @param nlambda The number of lambda values - default is 100.
#' @param lambda.min.ratio Smallest value for lambda, as a fraction of \code{lambda.max}, the (data derived) entry
#' value (i.e. the smallest value for which all coefficients are zero). The default
#' depends on the sample size nobs relative to the number of variables nvars. The default is 0.0001
#' @param alpha mixing value for \code{elastic.net}, \code{mcp.net}, \code{scad.net}, \code{grp.mcp.net}, \code{grp.scad.net}. 
#' penalty applied is (1 - alpha) * (ridge penalty) + alpha * (lasso/mcp/mcp/grp.lasso penalty)
#' @param gamma tuning parameter for SCAD and MCP penalties. must be >= 1
#' @param tau mixing value for \code{sparse.grp.lasso}. penalty applied is (1 - tau) * (group lasso penalty) + tau * (lasso penalty)
#' @param groups A vector of describing the grouping of the coefficients. See the example below. All unpenalized variables
#' should be put in group 0
#' @param scale.factor of length \code{nvars === ncol(xtx) == length(xty)} for scaling columns of \code{x}. The standard deviation
#' for each column of \code{x} is a common choice for \code{scale.factor}. Coefficients will be returned on original scale. Default is 
#' no scaling.
#' @param penalty.factor Separate penalty factors can be applied to each coefficient. 
#' This is a number that multiplies lambda to allow differential shrinkage. Can be 0 for some variables, 
#' which implies no shrinkage, and that variable is always included in the model. Default is 1 for all 
#' variables. 
#' @param group.weights penalty factors applied to each group for the group lasso. Similar to \code{penalty.factor}, 
#' this is a number that multiplies lambda to allow differential shrinkage. Can be 0 for some groups, 
#' which implies no shrinkage, and that group is always included in the model. Default is sqrt(group size) for all
#' groups. 
#' @param maxit integer. Maximum number of OEM iterations
#' @param tol convergence tolerance for OEM iterations
#' @param irls.maxit integer. Maximum number of IRLS iterations
#' @param irls.tol convergence tolerance for IRLS iterations. Only used if \code{family != "gaussian"}
#' @return An object with S3 class \code{"oem"}
#' @import Rcpp
#' @import Matrix
#' @import foreach
#' @export
#' @references Huling. J.D. and Chien, P. (2022), Fast Penalized Regression and Cross Validation for Tall Data with the oem Package.
#' Journal of Statistical Software 104(6), 1-24. doi:10.18637/jss.v104.i06
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
#'            penalty = c("lasso", "elastic.net", 
#'                         "ols", 
#'                         "mcp",       "scad", 
#'                         "mcp.net",   "scad.net",
#'                         "grp.lasso", "grp.lasso.net",
#'                         "grp.mcp",   "grp.scad",
#'                         "sparse.grp.lasso"), 
#'            standardize = FALSE, intercept = FALSE,
#'            groups = rep(1:20, each = 5))
#'            
#' xtx <- crossprod(x) / n.obs
#' xty <- crossprod(x, y) / n.obs
#' 
#' fit.xtx <- oem.xtx(xtx = xtx, xty = xty, 
#'                    penalty = c("lasso", "elastic.net", 
#'                                "ols", 
#'                                "mcp",       "scad", 
#'                                "mcp.net",   "scad.net",
#'                                "grp.lasso", "grp.lasso.net",
#'                                "grp.mcp",   "grp.scad",
#'                                "sparse.grp.lasso"), 
#'                    groups = rep(1:20, each = 5))    
#'                    
#' max(abs(fit$beta[[1]][-1,] - fit.xtx$beta[[1]]))
#' max(abs(fit$beta[[2]][-1,] - fit.xtx$beta[[2]]))       
#' 
#' layout(matrix(1:2, ncol = 2))
#' plot(fit.xtx)
#' plot(fit.xtx, which.model = 2)
#' 
oem.xtx <- function(xtx, 
                    xty, 
                    family = c("gaussian", "binomial"),
                    penalty = c("elastic.net", 
                                "lasso", 
                                "ols", 
                                "mcp",           "scad", 
                                "mcp.net",       "scad.net",
                                "grp.lasso",     "grp.lasso.net",
                                "grp.mcp",       "grp.scad",
                                "grp.mcp.net",   "grp.scad.net",
                                "sparse.grp.lasso"),
                    lambda = numeric(0),
                    nlambda = 100L,
                    lambda.min.ratio = NULL,
                    alpha = 1,
                    gamma = 3,
                    tau   = 0.5,
                    groups = numeric(0),
                    scale.factor   = numeric(0),
                    penalty.factor = NULL,
                    group.weights  = NULL,
                    maxit = 500L, 
                    tol = 1e-7,
                    irls.maxit = 100L,
                    irls.tol = 1e-3) 
{
    this.call    <- match.call()
    
    family       <- match.arg(family)
    
    ## don't default to fitting all penalties!
    ## only allow multiple penalties if the user
    ## explicitly chooses multiple penalties
    if ("penalty" %in% names(this.call))
    {
        penalty  <- match.arg(penalty, several.ok = TRUE)
    } else 
    {
        penalty  <- match.arg(penalty, several.ok = FALSE)
    }
    
    dims <- dim(xtx)
    
    if (is.null(dims))
    {
        stop("xtx must be a matrix")
    }
    
    if (dims[1] != dims[2]) stop("xtx must be a square matrix equal to X'X. do NOT provide design matrix")
    
    p <- dims[2]
    xty <- drop(xty)
    
    if (p != NROW(xty)) stop("xty must have length equal to the number of columns and rows of xtx. do NOT provide response vector")
    
    if(inherits(xtx, "sparseMatrix"))
    {
        stop("Sparse matrices not allowed")
    }
    
    if (p < 2)
    {
        stop("xtx must have at least two columns")
    }
    
    if (family == "binomial") stop("binomial not implemented yet")
    
    if (is.null(penalty.factor)) {
        penalty.factor <- rep(1, p)
    }
    
    varnames <- colnames(xtx)
    if(is.null(varnames)) varnames = paste("V", seq(p), sep="")
    
    
    if (length(penalty.factor) != p) {
        stop("penalty.factor must have same length as number of columns in x")
    }
    penalty.factor <- drop(penalty.factor)
    
    if (any(grep("grp", penalty) > 0)) {
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
            }
            group.weights <- drop(group.weights)
            if (length(group.weights) != length(unique.groups)) {
                stop("group.weights must have same length as the number of groups")
            }
            group.weights <- as.numeric(group.weights)
        } else {
            # default to sqrt(group size) for each group weight
            group.weights <- numeric(0)
            
        }
        
        
        
    } else {
        unique.groups <- numeric(0)
        group.weights <- numeric(0)
    }
    
    
    if (is.null(lambda.min.ratio)) {
        lambda.min.ratio <- 0.0001
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
    
    if (!is.list(lambda))
    {
        lambda <- sort(as.numeric(lambda), decreasing = TRUE)
        
        ## ensure is double type
        if (length(lambda) > 0)
        {
            lambda    <- as.double(lambda)
        }
        
        lambda <- rep(list(lambda), length(penalty))
        
    } else 
    {
        if (length(lambda) != length(penalty))
        {
            stop("If list of lambda vectors is provided, it must be 
                 the same length as the number of penalties fit")
        }
        nlambda.tmp <- length(lambda[[1]])
        for (l in 1:length(lambda))
        {
            
            ## check to make sure all things in the list are actually vectors
            if ( is.null(lambda[[l]]) || length(lambda[[l]]) < 1 )
            {
                stop("Provided lambda vector must have at least one value")
            }
            
            if (length(lambda[[l]]) != nlambda.tmp)
            {
                stop("All provided lambda vectors must have same length")
            }
            
            ## ensure is double type
            lambda[[l]] <- as.double(sort(as.numeric(lambda[[l]]), decreasing = TRUE))
            
        }
    }
    
    ##    ensure types are correct
    ##    before sending to c++
    
    groups        <- as.integer(groups)
    unique.groups <- as.integer(unique.groups)
    nlambda       <- as.integer(nlambda)
    alpha         <- as.double(alpha)
    gamma         <- as.double(gamma)
    tau           <- as.double(tau)
    tol           <- as.double(tol)
    irls.tol      <- as.double(irls.tol)
    irls.maxit    <- as.integer(irls.maxit)
    maxit         <- as.integer(maxit)
    
    if (length(scale.factor) > 0) 
    {
        if (length(scale.factor) != p) stop("scale.factor must be same length as xty (nvars)")
        scale.factor <- as.double(scale.factor)
    }
    
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
                    irls_tol     = irls.tol)
    
    res <- switch(family,
                  "gaussian" = oemfit.xtx.gaussian(xtx, xty, 
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
                                                   tau,
                                                   scale.factor,
                                                   penalty.factor,
                                                   options),
                  "binomial" = oemfit.xtx.binomial(xtx, xty, 
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
                                                   tau,
                                                   scale.factor,
                                                   penalty.factor,
                                                   options)
    )
    
    for (i in 1:length(penalty))
    {
        if (penalty[i] == "ols") res$beta[[i]] <- matrix(res$beta[[i]], ncol = 1)
        rownames(res$beta[[i]]) <- varnames
    }
    
    names(res$beta) <- penalty
    
    nz <- lapply(1:length(res$beta), function(m) 
        sapply(predict.oem(res, type = "nonzero", which.model = m), length)
    )
    
    res$nvars    <- p
    res$penalty  <- penalty
    res$family   <- family
    res$varnames <- varnames
    res$nzero    <- nz
    
    class(res)   <- c(class(res), "oem")
    res
}


oemfit.xtx.gaussian <- function(xtx, 
                                xty, 
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
                                tau,
                                scale.factor,
                                penalty.factor,
                                options)
{
    ret <- .Call("oem_xtx", 
                 xtx, 
                 xty, 
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
                 tau,
                 scale.factor,
                 penalty.factor,
                 options,
                 PACKAGE = "oem")
    
    class(ret) <- "oemfit_gaussian"
    ret
}


oemfit.xtx.binomial <- function(xtx, 
                                xty, 
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
                                tau,
                                scale.factor,
                                penalty.factor,
                                options)
{

    ret <- .Call("oem_xtx_logistic", 
                 xtx, 
                 xty, 
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
                 tau,
                 scale.factor,
                 penalty.factor,
                 options,
                 PACKAGE = "oem")
    
    class(ret) <- "oemfit_binomial"
    ret
}



