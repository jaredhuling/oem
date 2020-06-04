
#' Deprecated functions
#'
#' These functions have been renamed and deprecated in \pkg{oem}:
#' \code{oemfit()} (use \code{\link{oem}()}), \code{cv.oemfit()}
#' (use \code{\link{cv.oem}()}), \code{print.oemfit()}, 
#' \code{plot.oemfit()}, \code{predict.oemfit()}, and 
#' \code{coef.oemfit()}.
#' @rdname deprecated
#' @aliases oem-deprecated
#' 
#' @param formula an object of 'formula' (or one that can be coerced to
#' that class): a symbolic description of the model to be fitted. The
#' details of model specification are given under 'Details'
#' @param data an optional data frame, list or environment (or object
#' coercible by 'as.data.frame' to a data frame) containing the
#' variables in the model.  If not found in 'data', the
#' variables are taken from 'environment(formula)', typically
#' the environment from which 'oemfit' is called.
#' @param lambda A user supplied \code{lambda} sequence. Typical usage is
#' to have the program compute its own \code{lambda} sequence based on
#' \code{nlambda} and \code{lambda.min.ratio}. Supplying a value of
#' \code{lambda} overrides this. WARNING: use with care. Do not supply a
#' single value for \code{lambda} (for predictions after CV use \code{predict()} 
#' instead).  Supply instead a decreasing sequence of \code{lambda}
#' values. \code{oemfit} relies on its warms starts for speed, and its
#' often faster to fit a whole path than compute a single fit.
#' @param nlambda The number of \code{lambda} values - default is 100.
#' @param lambda.min.ratio Smallest value for \code{lambda}, as a fraction of
#' \code{lambda.max}, the (data derived) entry value (i.e. the smallest
#' value for which all coefficients are zero). The default depends on the
#' sample size \code{nobs} relative to the number of variables
#' \code{nvars}. If \code{nobs > nvars}, the default is \code{0.0001},
#' close to zero.  If \code{nobs < nvars}, the default is \code{0.01}.
#' A very small value of
#' \code{lambda.min.ratio} will lead to a saturated fit in the \code{nobs <
#' nvars} case.
#' @param tolerance Convergence tolerance for OEM. Each inner
#' OEM loop continues until the maximum change in the
#' objective after any coefficient update is less than \code{tolerance}.
#' Defaults value is \code{1E-3}.
#' @param maxIter Maximum number of passes over the data for all lambda
#' values; default is 1000.
#' @param standardized Logical flag for x variable standardization, prior to
#' fitting the model sequence. The coefficients are always returned on
#' the original scale. Default is \code{standardize=TRUE}.
#' If variables are in the same units already, you might not wish to
#' standardize.
#' @param numGroup Integer value for the number of groups to use for OEM
#' fitting. Default is 1.
#' @param penalty type in lower letters. Different types include
#' 'lasso', 'scad', 'ols' (ordinary least square), 'elastic-net',
#' 'ngarrote' (non-negative garrote) and 'mcp'.
#' @param alpha alpha value for scad and mcp.
#' @param evaluate debugging argument
#' @param condition Debugging for different ways of calculating OEM.
#' @details The sequence of models implied by 'lambda' is fit by OEM algorithm. 
#' @author Bin Dai
#' @export
oemfit <- function(formula, data = list(), lambda = NULL, nlambda = 100,
                   lambda.min.ratio = NULL, tolerance = 1e-3,
                   maxIter = 1000, standardized = TRUE, numGroup = 1,
                   penalty = c("lasso", "scad", "ols", "elastic.net",
                               "ngarrote", "mcp"), alpha = 3,
                   evaluate = 0, condition = -1) {
    # prepare the generic arguments
    .Deprecated("oem")
    this.call <- match.call()
    penalty <- match.arg(penalty)
    mf <- model.frame(formula = formula, data = data)
    x <- model.matrix(attr(mf, "terms"), data = mf)
    y <- model.response(mf)
    nobs <- nrow(x)
    nlambda <- as.integer(nlambda)
    tolerance <- as.double(tolerance)
    maxIter <- as.integer(maxIter)
    numGroup <- as.integer(numGroup)
    
    if (!standardized) {
        meanx <- apply(x, 2, mean)
        normx <- sqrt(apply((t(x) - meanx)^2, 1, sum) / nobs)
        nz <- which(normx > .0001)
        xx <- scale(x[,nz], meanx[nz], normx[nz])
        yy <- y - mean(y)
    } else {
        xx <- x
        yy <- y
    }
    nvars <- as.integer(ncol(xx))
    
    # NOTES: reset lambda.max here
    lambda.max <- max(abs(t(xx) %*% yy / nobs) * 1.1)
    if (is.null(lambda)) {
        if (is.null(lambda.min.ratio)) 
            lambda.min.ratio = ifelse(nobs < nvars, .05, 1e-3)
        if (lambda.min.ratio >= 1)
            stop("lambda.min.ratio should be less than 1")
        wlambda <- exp( seq(log(as.double(lambda.max)),
                            log(as.double(lambda.min.ratio)),
                            log(as.double(lambda.min.ratio / lambda.max))
                            /nlambda) )
        wlambda <- wlambda[1:nlambda]
    } else {
        if (any(lambda < 0)) stop("lambda should be non-negative")
        wlambda <- as.double(rev(sort(lambda)))
        nlambda <- length(wlambda)
    }
    method <- as.integer(switch(penalty,
                                ols = 0,
                                lasso = 1,
                                scad = 2,
                                elastic.net = 3,
                                ngarrote = 4,
                                mcp = 5))
    
    # determine which condition to calculate
    if (condition < 0)
        condition = ifelse(2 * nobs <= nvars, 0, 1)
    
    if (penalty == "ols") wlambda = 0
    result <- .Call("oemfit", xx, yy, maxIter, tolerance, wlambda,
                    method, numGroup, as.double(alpha), as.integer(evaluate),
                    as.integer(condition),
                    PACKAGE = "oem")
    result$lambda <- wlambda
    result$call <- this.call
    result$sumSquare <- apply( (yy - xx %*% result$beta)^2, 2, sum) / nobs
    
    # unstandadize
    if (!standardized && nrow(beta) == nz + 1) {
        beta <- matrix(0, ncol(xx) + 1, length(wlambda))
        beta[nz+1,] <- result$beta / normx[nz]
        beta[1,] <- mean(y) - crossprod(meanx, beta[-1,, drop = FALSE])
        result$beta <- beta
    }
    
    class(result) <- c("oemfit", class(result))
    result
}



#' @rdname deprecated
#' @param type.measure type.measure measure to evaluate for cross-validation. 
#' \code{type.measure = "mse"} (mean squared error) or 
#' \code{type.measure = "mae"} (mean absolute error)
#' @param ... arguments to be passed to \code{oemfit()}
#' @param nfolds number of folds for cross-validation. default is 10. 
#' @param foldid an optional vector of values between 1 and nfold specifying which fold each observation belongs to.
#' @importFrom stats model.frame model.matrix model.response
#' @export
cv.oemfit <- function(formula, data = list(), lambda = NULL,
                      type.measure = c('mse', 'mae'), ...,
                      nfolds = 10, foldid,
                      penalty = c("lasso", "scad", "elastic.net",
                                  "ngarrote", "mcp")) {
    
    .Deprecated("cv.oem")
    this.call <- match.call()
    mf <- model.frame(formula = formula, data = data)
    x <- model.matrix(attr(mf, "terms"), data = mf)
    y <- model.response(mf)
    N <- nrow(x)
    y <- drop(y) # only vector form is allowed
    if (missing(type.measure)) type.measure = "mse"
    else type.measure <- match.arg(type.measure)
    if (!is.null(lambda) && length(lambda) < 2)
        stop ("Need more than on value of lambda for cv.oemfit")
    oemfit.object <- oemfit(formula, data = data, lambda = lambda, penalty = penalty)
    lambda <- oemfit.object$lambda
    if (missing(foldid)) foldid <- sample(rep(seq(nfolds), length = N))
    else nfolds <- max(foldid)
    nz <- sapply(predict(oemfit.object, type = 'nonzero'), length)
    if (nfolds < 3) stop("nfolds must be greater than 3; nfolds = 10 is recommended")
    outlist <- as.list(seq(nfolds))
    ######################3
    # fit the n-fold model and store
    for (i in seq(nfolds)) {
        index <- foldid == i
        y_sub <- y[!index]
        xx <- x[!index,, drop = FALSE]
        outlist[[i]] <- oemfit(y_sub ~ xx - 1, lambda = lambda, penalty = penalty)
    }
    # Use type.measure to evaluate
    typenames <- c(mse = "Mean-Squared Error", mae = "Mean Absolute Error")
    if (!match(type.measure, c("mse", "mae"), FALSE)){
        warning ("Only 'mse' and 'mae' available; 'mse' used")
        type.measure <- "mse"
    }
    predmat <- matrix(NA, length(y), length(lambda))
    nlams <- double(nfolds)
    for (i in seq(nfolds)) {
        index <- foldid == i
        fitobj <- outlist[[i]]
        preds <- predict(fitobj, x[index,, drop = FALSE])
        nlami <- length(outlist[[i]]$lambda)
        predmat[index, seq(nlami)] <- preds
        nlams[i] <- nlami
    }
    N.cv <- length(y) - apply(is.na(predmat), 2, sum)
    cvraw <- switch(type.measure,
                    'mse' = (y - predmat)^2,
                    'mae' = abs(y - predmat)
    )
    
    cvm <- apply(cvraw, 2, mean, na.rm = TRUE)
    cvsd <- sqrt(apply(scale(cvraw, cvm, FALSE)^2, 2, mean, na.rm = TRUE)
                 /(N.cv - 1))
    out <- list(lambda = lambda, cvm = cvm, cvsd = cvsd,
                cvup = cvm + cvsd, cvlo = cvm - cvsd,
                nzero = nz, fit = oemfit.object,
                name = typenames[type.measure])
    lamin <- getmin.old(lambda, cvm, cvsd)
    obj <- c(out, as.list(lamin))
    obj$penalty <- penalty
    class(obj) <- "cv.oemfit"
    obj
}


getmin.old <- function(lambda, cvm, cvsd){
    cvmin <- min(cvm)
    idmin <- cvm <= cvmin
    lambda.min <- max(lambda[idmin])
    idmin <- match(lambda.min, lambda)
    semin <- (cvm + cvsd)[idmin]
    idmin <- cvm < semin
    lambda.1se <- max(lambda[idmin])
    list(lambda.min = lambda.min, lambda.1se = lambda.1se)
}


coef.oemfit <- function(object, s = NULL, ...) {
    predict(object, s = s, type = "coefficients")
}

#' @rdname deprecated
#' @param x fitted \code{oemfit} object
#' @param xvar what is on the X-axis. "norm" plots against the L1-norm of the coefficients,
#' "lambda" against the log-lambda sequence, and "dev" against the percent deviance
#' explained.
#' @param xlab x-axis label
#' @param ylab y-axis label
#' @method plot oemfit
#' @importFrom grDevices rainbow
#' @importFrom graphics title strwidth par
#' @export
plot.oemfit <- function(x, xvar = c("norm", "lambda", "loglambda", "dev"),
                        xlab = iname, ylab = "Coefficients",
                        ...) {
    xvar <- match.arg(xvar)
    nbeta <- as.matrix(x$beta)
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


#' @rdname deprecated
#' @param object fitted \code{oemfit} object
#' @param newx matrix of new values for x at which predictions are to be
#' made. Must be a matrix.
#' @param s Value(s) of the penalty parameter lambda at which predictions
#' are required. Default is the entire sequence used to create the model.
#' @param type not used.
#' @export
predict.oemfit <- function(object, newx, s = NULL,
                           type = c("response",
                                    "coefficients",
                                    "nonzero"), ...) {
    type <- match.arg(type)
    nbeta <- object$beta
    if(!is.null(s)){
        lambda <- object$lambda
        lamlist <- lambda.interp(object$lambda,s)
        nbeta <- nbeta[,lamlist$left,drop=FALSE]*lamlist$frac +nbeta[,lamlist$right,drop=FALSE]*(1-lamlist$frac)
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
    nfit <- as.matrix(newx %*% nbeta)
}

#' @rdname deprecated
#' @param digits significant digits in print out.
#' @method print oemfit
#' @export
print.oemfit <- function(x, digits = max(3, getOption("digits") - 3), ...) {
    cat("\nCall: ", deparse(x$call))
    cat("\nPenalty:", x$penalty, "\n\n")
    print(cbind(Df = x$df, sumSquare = signif(x$sumSquare, digits),
                Lambda = signif(x$lambda, digits)))
}


