
#' @useDynLib oem
#' @import Rcpp
#' @exportPattern "^[[:alpha:]]+"
#' @export
oem <- function(x, 
                y, 
                family = c("gaussian", "binomial"),
                penalty = c("elastic.net", "lasso", "ols", "mcp"),
                lambda = numeric(0),
                nlambda = 100L,
                lambda.min.ratio = NULL,
                alpha = 1,
                gamma = 3,
                penalty.factor = NULL,
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
    
    if (length(y) != n) {
        stop("x and y lengths do not match")
    }
    
    if (is.null(penalty.factor)) {
        penalty.factor <- rep(1, p)
    }
    
    if (length(penalty.factor) != p) {
        stop("penalty.factor must have same length as number of columns in x")
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
    
    res <- .Call("oem_fit_dense_tall", 
                 x, y, 
                 family, 
                 penalty, 
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
                 PACKAGE = "oem")
    class(res) <- "oem.fit"
    res
}