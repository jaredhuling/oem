## the code here is largely based on the code
## from the glmnet package (no reason to reinvent the wheel)

#' Prediction method for Orthogonalizing EM fitted objects
#'
#' @param object fitted "oem" model object
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix; can be sparse as in Matrix package. 
#' This argument is not used for type=c("coefficients","nonzero")
#' @param s Value(s) of the penalty parameter lambda at which predictions are required. Default is the entire sequence used to create 
#' the model.
#' @param which.model If multiple penalties are fit and returned in the same oem object, the which.model argument is used to 
#' specify which model to make predictions for. For example, if the oem object "oemobj" was fit with argument 
#' penalty = c("lasso", "grp.lasso"), then which.model = 2 provides predictions for the group lasso model.
#' @param type Type of prediction required. Type == "link" gives the linear predictors for the "binomial" model; for "gaussian" models it gives the fitted values. 
#' Type == "response" gives the fitted probabilities for "binomial". Type "coefficients" computes the coefficients at the requested values for s.
#' Type "class" applies only to "binomial" and produces the class label corresponding to the maximum probability.
#' @return An object depending on the type argument
#' @export
#' @examples
#' set.seed(123)
#' n.obs <- 1e4
#' n.vars <- 100
#' n.obs.test <- 1e3
#' 
#' true.beta <- c(runif(15, -0.5, 0.5), rep(0, n.vars - 15))
#' 
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + x %*% true.beta
#' x.test <- matrix(rnorm(n.obs.test * n.vars), n.obs.test, n.vars)
#' y.test <- rnorm(n.obs.test, sd = 3) + x.test %*% true.beta
#' 
#' fit <- oem(x = x, y = y, penalty = c("lasso", "grp.lasso"), groups = rep(1:10, each = 10), nlambda = 10)
#' 
#' preds.lasso <- predict(fit, newx = x.test, type = "response", which.model = 1)
#' preds.grp.lasso <- predict(fit, newx = x.test, type = "response", which.model = 2)
#' 
#' apply(preds.lasso,     2, function(x) mean((y.test - x) ^ 2))
#' apply(preds.grp.lasso, 2, function(x) mean((y.test - x) ^ 2))
#' 
predict.oemfit <- function(object, newx, s = NULL, which.model = 1,
                           type = c("link",
                                    "response",
                                    "coefficients",
                                    "nonzero",
                                    "class"), ...) 
{
    type <- match.arg(type)
    if(missing(newx)){
        if(!match(type, c("coefficients", "nonzero"), FALSE))stop("A value for 'newx' must be supplied")
    }
    nbeta <- object$beta[[which.model]]
    
    if(!is.null(s)){
        #vnames=dimnames(nbeta)[[1]]
        lambda <- object$lambda
        lamlist <- lambda.interp(object$lambda,s)
        nbeta <- nbeta[,lamlist$left,drop=FALSE]*lamlist$frac +nbeta[,lamlist$right,drop=FALSE]*(1-lamlist$frac)
        #dimnames(nbeta)=list(vnames,paste(seq(along=s)))
    }
    if (type == "coefficients") return(nbeta)
    if (type == "nonzero") {
        newbeta <- abs(as.matrix(object$beta)) > 0
        index <- 1:(dim(newbeta)[1])
        nzel <- function(x, index) if(any(x)) index[x] else NULL
        betaList <- apply(newbeta, 2, nzel, index)
        return(betaList)
    }
    
    newx <- as.matrix(newx)
    # add constant column if needed                                        
    if (ncol(newx) < nrow(nbeta))
        newx <- cbind(rep(1, nrow(newx)), newx)
    
    as.matrix(newx %*% nbeta)
}

#' Prediction method for Orthogonalizing EM fitted objects
#'
#' @param object fitted "oem" model object
#' @param which.model If multiple penalties are fit and returned in the same oem object, the which.model argument is used to 
#' specify which model to make predictions for. For example, if the oem object "oemobj" was fit with argument 
#' penalty = c("lasso", "grp.lasso"), then which.model = 2 provides predictions for the group lasso model.
#' @param xvar What is on the X-axis. "norm" plots against the L1-norm of the coefficients, "lambda" against the log-lambda sequence, and "dev" 
#' against the percent deviance explained.
#' @param ... other graphical parameters for the plot
#' @export
#' @examples
#' set.seed(123)
#' n.obs <- 1e4
#' n.vars <- 100
#' n.obs.test <- 1e3
#' 
#' true.beta <- c(runif(15, -0.5, 0.5), rep(0, n.vars - 15))
#' 
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + x %*% true.beta
#' 
#' fit <- oem(x = x, y = y, penalty = c("lasso", "grp.lasso"), groups = rep(1:10, each = 10))
#' 
#' layout(matrix(1:2, ncol = 2))
#' plot(fit, which.model = 1)
#' plot(fit, which.model = 2)
#' 
plot.oemfit <- function(x, which.model = 1,
                        xvar = c("norm", "lambda", "loglambda", "dev"),
                        xlab = iname, ylab = "Coefficients", 
                        ...) 
{
    xvar <- match.arg(xvar)
    nbeta <- as.matrix(x$beta[[which.model]])
    switch(xvar,
           "norm" = {
               index <- apply(abs(nbeta), 2, sum)
               iname <- "L1 Norm"
               xlim <- range(index)
           },
           "lambda" = {
               index <- x$lambda
               iname <- "Lambda"
               xlim <- rev(range(index))
           },
           "loglambda" = {
               index <- log(x$lambda)
               iname <- "Log Lambda"
               xlim <- rev(range(index))
           },
           "dev" = {
               index = x$sumSquare
               iname = "Sum of Squares"
               xlim <- range(index)
           }
    )
    matplot(index, t(nbeta), lty = 1, xlab = xlab, ylab = ylab, xlim = xlim,
            type = 'l', ...)
}



predict.oemfit_gaussian <- function(object, newx, s = NULL, which.model = 1,
                                    type = c("link", 
                                             "response",
                                             "coefficients",
                                             "nonzero"), ...)
{
    NextMethod("predict")
} 

predict.oemfit_binomial <- function(object, newx, s=NULL, which.model = 1,
                                    type=c("link", 
                                           "response", 
                                           "coefficients", 
                                           "class", 
                                           "nonzero"), ...)
{
    type <- match.arg(type)
    nfit <- NextMethod("predict")
    switch(type,
           response={
               prob=exp(-nfit)
               1 / (1 + prob)
           },
           class={
               cnum=ifelse(nfit > 0, 2, 1)
               clet=object$classnames[cnum]
               if(is.matrix(cnum))clet=array(clet,dim(cnum),dimnames(cnum))
               clet
           },
           nfit
    )
}  
