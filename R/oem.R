
#' @useDynLib oem
#' @import Rcpp
#' @exportPattern "^[[:alpha:]]+"
#' @export
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
                intercept = FALSE,
                maxit = 500L, 
                tol = 1e-5,
                irls.maxit = 100L,
                irls.tol = 1e-5) 
{
    family  <- match.arg(family)
    penalty <- match.arg(penalty, several.ok = TRUE)
    
    dims <- dim(x)
    n <- dims[1]
    p <- dims[2]
    y <- drop(y)
    
    is.sparse <- FALSE
    if(inherits(x, "sparseMatrix")){##Sparse case
        is.sparse <- TRUE
        x <- as(x,"CsparseMatrix")
        x <- as(x,"dgCMatrix")
    }
    
    if (length(y) != n) {
        stop("x and y lengths do not match")
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
        groups <- drop(groups)
        if (!is.null(group.weights))
        {
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
    
    fam_type <- paste0(family, is.sparse)
    
    
    res <- switch(fam_type,
                  "gaussianFALSE" = .Call("oem_fit_dense", 
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
                                          list(maxit      = maxit,
                                               tol        = tol,
                                               irls_maxit = irls.maxit,
                                               irls_tol   = irls.tol),
                                          PACKAGE = "oem"),
                  "gaussianTRUE" = .Call("oem_fit_sparse", 
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
                                         list(maxit      = maxit,
                                              tol        = tol,
                                              irls_maxit = irls.maxit,
                                              irls_tol   = irls.tol),
                                         PACKAGE = "oem"),
                  "binomialFALSE" = list(NULL),
                  "binomialTRUE"  = list(NULL))
    
    class(res) <- "oem.fit"
    res
}