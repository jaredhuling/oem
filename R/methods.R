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
#' @param ... not used 
#' @importFrom graphics abline abline axis matplot points segments
#' @importFrom methods as
#' @importFrom stats approx predict quantile runif weighted.mean
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
#' fit <- oem(x = x, y = y, 
#'            penalty = c("lasso", "grp.lasso"), 
#'            groups = rep(1:10, each = 10), 
#'            nlambda = 10)
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
    
    num.models <- length(object$beta)
    if (which.model > num.models)
    {
        err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
        stop(err.txt)
    }
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
        newbeta <- abs(as.matrix(object$beta[[which.model]])) > 0
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

#' Plot method for Orthogonalizing EM fitted objects
#'
#' @param x fitted "oem" model object
#' @param which.model If multiple penalties are fit and returned in the same oem object, the which.model argument is used to 
#' specify which model to plot. For example, if the oem object "oemobj" was fit with argument 
#' penalty = c("lasso", "grp.lasso"), then which.model = 2 provides a plot for the group lasso model.
#' @param xvar What is on the X-axis. "norm" plots against the L1-norm of the coefficients, "lambda" against the log-lambda sequence, and "dev" 
#' against the percent deviance explained.
#' @param labsize size of labels for variable names. If labsize = 0, then no variable names will be plotted
#' @param xlab label for x-axis
#' @param ylab label for y-axis
#' @param ... other graphical parameters for the plot
#' @rdname plot
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
                        labsize = 0.6,
                        xlab = iname, ylab = "Coefficients", 
                        ...) 
{
    num.models <- length(x$beta)
    if (which.model > num.models)
    {
        err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
        stop(err.txt)
    }
    
    main.txt <- x$penalty[which.model]
    
    xvar <- match.arg(xvar)
    nbeta <- as.matrix(x$beta[[which.model]])
    remove <- apply(nbeta, 1, function(betas) all(betas == 0) )
    switch(xvar,
           "norm" = {
               index    <- apply(abs(nbeta), 2, sum)
               iname    <- expression(L[1] * " Norm")
               xlim     <- range(index)
               approx.f <- 1
           },
           "lambda" = {
               index    <- x$lambda
               iname    <- expression(lambda)
               xlim     <- rev(range(index))
               approx.f <- 0
           },
           "loglambda" = {
               index    <- log(x$lambda)
               iname    <- expression(log(lambda))
               xlim     <- rev(range(index))
               approx.f <- 1
           },
           "dev" = {
               index    <- x$sumSquare
               iname    <- "Sum of Squares"
               xlim     <- range(index)
               approx.f <- 1
           }
    )
    if (all(remove)) stop("All beta estimates are zero for all values of lambda. No plot returned.")
    
    matplot(index, t(nbeta[!remove,,drop=FALSE]), 
            lty = 1, xlab = xlab, 
            col=rainbow(sum(!remove)),
            ylab = ylab, xlim = xlim,
            type = 'l', ...)
    
    atdf <- pretty(index, n = 10L)
    plotnz <- approx(x = index, y = x$nzero[[which.model]], xout = atdf, rule = 2, method = "constant", f = approx.f)$y
    axis(side=3, at = atdf, labels = plotnz, tick=FALSE, line=0)
    title(main.txt, line = 2.5)
    
    
    
    # Adjust the margins to make sure the labels fit
    labwidth <- ifelse(labsize > 0, max(strwidth(rownames(nbeta[!remove,]), "inches", labsize)), 0)
    margins <- par("mai")
    par("mai" = c(margins[1:3], max(margins[4], labwidth*1.4)))
    if ( labsize > 0 && !is.null(rownames(nbeta)) ) 
    {
        take <- which(!remove)
        for (i in 1:sum(!remove)) {
            j <- take[i]
            axis(4, at = nbeta[j, ncol(nbeta)], labels = rownames(nbeta)[j],
                 las=1, cex.axis=labsize, col.axis=rainbow(sum(!remove))[i], 
                 lty = (i - 1) %% 5 + 1, col = rainbow(sum(!remove))[i])
        }
    }
    par("mai"=margins)
}


#' @param sign.lambda Either plot against log(lambda) (default) or its negative if sign.lambda=-1.
#' @rdname plot
#' @method plot cv.oem
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
#' fit <- cv.oem(x = x, y = y, penalty = c("lasso", "grp.lasso"), groups = rep(1:10, each = 10))
#' 
#' layout(matrix(1:2, ncol = 2))
#' plot(fit, which.model = 1)
#' plot(fit, which.model = 2)
#' 
plot.cv.oem <- function(x, which.model = 1, sign.lambda = 1, ...)
{
    # modified from glmnet
    object = x
    num.models <- length(object$cvm)
    if (which.model > num.models)
    {
        err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
        stop(err.txt)
    }
    
    main.txt <- x$oem.fit$penalty[which.model]
    
    xlab=expression(log(lambda))
    if(sign.lambda<0)xlab=paste("-",xlab,sep="")
    plot.args=list(x    = sign.lambda * log(object$lambda),
                   y    = object$cvm[[which.model]],
                   ylim = range(object$cvup[[which.model]], object$cvlo[[which.model]]),
                   xlab = xlab,
                   ylab = object$name,
                   type = "n")
    new.args=list(...)
    if(length(new.args))plot.args[names(new.args)]=new.args
    do.call("plot", plot.args)
    error.bars(sign.lambda * log(object$lambda), 
               object$cvup[[which.model]], 
               object$cvlo[[which.model]], width = 0.005)
    points(sign.lambda*log(object$lambda), object$cvm[[which.model]], pch=20, col="dodgerblue")
    axis(side=3,at=sign.lambda*log(object$lambda),labels = paste(object$nzero[[which.model]]), tick=FALSE, line=0)
    abline(v = sign.lambda * log(object$lambda.min.models[which.model]), lty=2, lwd = 2, col = "firebrick1")
    abline(v = sign.lambda * log(object$lambda.1se.models[which.model]), lty=2, lwd = 2, col = "firebrick1")
    title(main.txt, line = 2.5)
    invisible()
}


#' @export 
predict.oemfit_gaussian <- function(object, newx, s = NULL, which.model = 1,
                                    type = c("link", 
                                             "response",
                                             "coefficients",
                                             "nonzero"), ...)
{
    NextMethod("predict")
} 


#' @export
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


#' log likelihood function for fitted oem objects
#'
#' @param object fitted "oem" model object.
#' @param which.model If multiple penalties are fit and returned in the same oem object, the which.model argument is used to 
#' specify which model to plot. For example, if the oem object "oemobj" was fit with argument 
#' penalty = c("lasso", "grp.lasso"), then which.model = 2 provides a plot for the group lasso model.
#' @param ... not used
#' @rdname logLik
#' @export
#' @examples
#' set.seed(123)
#' n.obs <- 2000
#' n.vars <- 50
#' 
#' true.beta <- c(runif(15, -0.25, 0.25), rep(0, n.vars - 15))
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + x %*% true.beta
#'
#' fit <- oem(x = x, y = y, penalty = "lasso", compute.loss = TRUE)
#'
#' logLik(fit)
#'
logLik.oemfit <- function(object, which.model = 1, ...) {
    # taken from ncvreg. Thanks to Patrick Breheny.
    n <- as.numeric(object$nobs)
    
    num.models <- length(object$beta)
    if (which.model > num.models)
    {
        err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
        stop(err.txt)
    }
    
    if (all(object$loss[[which.model]] == 1e99))
    {
        stop("oem object needed compute.loss set to TRUE. logLik not returned")
    }
    
    if (object$family == "gaussian")
    {
        
        resid.ss <- object$loss[[which.model]]
        logL <- -0.5 * n * (log(2 * pi) - log(n) + log(resid.ss)) - 0.5 * n
    } else if (object$family == "binomial")
    {
        logL <- -1 * object$loss[[which.model]]
    } else if (object$family == "poisson")
    {
        stop("poisson not complete yet")
        #y <- object$y
        #ind <- y != 0
        #logL <- -object$loss + sum(y[ind] * log(y[ind])) - sum(y) - sum(lfactorial(y))
    } else if (object$family == "coxph")
    {
        logL <- -1e99
    }
    
    logL
}


#' log likelihood function for fitted cross validation oem objects
#'
#' @rdname logLik
#' @method logLik cv.oem
#' @export 
#' @examples
#'
#' fit <- cv.oem(x = x, y = y, penalty = "lasso", compute.loss = TRUE)
#'
#' logLik(fit)
#'
logLik.cv.oem <- function(object, which.model = 1, ...) {
    
    object <- object$oem.fit
    # taken from ncvreg. Thanks to Patrick Breheny.
    n <- as.numeric(object$nobs)
    
    num.models <- length(object$beta)
    if (which.model > num.models)
    {
        err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
        stop(err.txt)
    }
    
    if (all(object$loss[[which.model]] == 1e99))
    {
        stop("oem object needed compute.loss set to TRUE. logLik not returned")
    }
    
    if (object$family == "gaussian")
    {
        
        resid.ss <- object$loss[[which.model]]
        logL <- -0.5 * n * (log(2 * pi) - log(n) + log(resid.ss)) - 0.5 * n
    } else if (object$family == "binomial")
    {
        logL <- -1 * object$loss[[which.model]]
    } else if (object$family == "poisson")
    {
        stop("poisson not complete yet")
        #y <- object$y
        #ind <- y != 0
        #logL <- -object$loss + sum(y[ind] * log(y[ind])) - sum(y) - sum(lfactorial(y))
    } else if (object$family == "coxph")
    {
        logL <- -1e99
    }
    
    logL
}


## the code here is largely based on the code
## from the glmnet package (no reason to reinvent the wheel)



#' Prediction function for fitted cross validation oem objects
#'
#' @param object fitted "cv.oem" model object
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix; can be sparse as in Matrix package. 
#' This argument is not used for type=c("coefficients","nonzero")
#' @param s Value(s) of the penalty parameter lambda at which predictions are required. Default is the entire sequence used to create 
#' the model. For predict.cv.oem, can also specify "lambda.1se" or "lambda.min" for best lambdas estimated by cross validation
#' @param which.model If multiple penalties are fit and returned in the same oem object, the which.model argument is used to 
#' specify which model to make predictions for. For example, if the oem object "oemobj" was fit with argument 
#' penalty = c("lasso", "grp.lasso"), then which.model = 2 provides predictions for the group lasso model. For 
#' predict.cv.oem, can specify
#' "best.model" to use the best model as estimated by cross-validation
#' @param ... used to pass the other arguments for predict.oemfit
#' @return An object depending on the type argument
#' @method predict cv.oem
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
#' fit <- cv.oem(x = x, y = y, 
#'               penalty = c("lasso", "grp.lasso"), 
#'               groups = rep(1:10, each = 10), 
#'               nlambda = 10)
#' 
#' preds.best <- predict(fit, newx = x.test, type = "response", which.model = "best.model")
#' 
#' apply(preds.best, 2, function(x) mean((y.test - x) ^ 2))
predict.cv.oem <- function(object, newx, which.model = "best.model",
                           s=c("lambda.1se","lambda.min"),...)
{
    if(is.numeric(s))lambda=s
    else 
        if(is.character(s)){
            s=match.arg(s)
            lambda=object[[s]]
        }
    if( is.numeric(which.model) )
    {
        mod.num <- as.integer(which.model)
        
        num.models <- length(object$cvm)
        if (mod.num > num.models)
        {
            err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
            stop(err.txt)
        }
    }
    else if(is.character(which.model))
    {
        mod.num <- object[["model.min"]]
    }
    
    else stop("Invalid form for s")
    predict(object$oem.fit, newx, s=lambda, which.model = mod.num, ...)
}









#' Plot method for Orthogonalizing EM fitted objects
#'
#' @param x fitted "oem" model object
#' @param which.model If multiple penalties are fit and returned in the same oem object, the which.model argument is used to 
#' specify which model to plot. For example, if the oem object "oemobj" was fit with argument 
#' penalty = c("lasso", "grp.lasso"), then which.model = 2 provides a plot for the group lasso model.
#' @param type one of "cv" or "coefficients". type = "cv" will produce a plot of cross validation results like plot.cv.oem. 
#' type = "coefficients" will produce a coefficient path plot like plot.oemfit
#' @param xvar What is on the X-axis. "norm" plots against the L1-norm of the coefficients, "lambda" against the log-lambda sequence, and "dev" 
#' against the percent deviance explained.
#' @param labsize size of labels for variable names. If labsize = 0, then no variable names will be plotted
#' @param xlab label for x-axis
#' @param ylab label for y-axis
#' @param sign.lambda Either plot against log(lambda) (default) or its negative if sign.lambda=-1.
#' @param ... other graphical parameters for the plot
#' @rdname plot
#' @method plot xval.oem
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
#' fit <- xval.oem(x = x, y = y, penalty = c("lasso", "grp.lasso"), groups = rep(1:10, each = 10))
#' 
#' layout(matrix(1:4, ncol = 2))
#' plot(fit, which.model = 1)
#' plot(fit, which.model = 2)
#' 
#' plot(fit, which.model = 1, type = "coef")
#' plot(fit, which.model = 2, type = "coef")
#' 
plot.xval.oem <- function(x, which.model = 1,
                          type = c("cv", "coefficients"),
                          xvar = c("norm", "lambda", "loglambda", "dev"),
                          labsize = 0.6,
                          xlab = iname, ylab = "Coefficients", 
                          sign.lambda = 1,
                          ...) 
{
    type       <- match.arg(type)
    num.models <- length(x$beta)
    if (which.model > num.models)
    {
        err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
        stop(err.txt)
    }
    
    main.txt <- x$penalty[which.model]
    
    if (type == "coefficients")
    {
        xvar <- match.arg(xvar)
        nbeta <- as.matrix(x$beta[[which.model]])
        remove <- apply(nbeta, 1, function(betas) all(betas == 0) )
        switch(xvar,
               "norm" = {
                   index    <- apply(abs(nbeta), 2, sum)
                   iname    <- expression(L[1] * " Norm")
                   xlim     <- range(index)
                   approx.f <- 1
               },
               "lambda" = {
                   index    <- x$lambda
                   iname    <- expression(lambda)
                   xlim     <- rev(range(index))
                   approx.f <- 0
               },
               "loglambda" = {
                   index    <- log(x$lambda)
                   iname    <- expression(log(lambda))
                   xlim     <- rev(range(index))
                   approx.f <- 1
               },
               "dev" = {
                   index    <- x$sumSquare
                   iname    <- "Sum of Squares"
                   xlim     <- range(index)
                   approx.f <- 1
               }
        )
        if (all(remove)) stop("All beta estimates are zero for all values of lambda. No plot returned.")
        
        matplot(index, t(nbeta[!remove,,drop=FALSE]), 
                lty = 1, xlab = xlab, 
                col=rainbow(sum(!remove)),
                ylab = ylab, xlim = xlim,
                type = 'l', ...)
        
        atdf <- pretty(index, n = 10L)
        plotnz <- approx(x = index, y = x$nzero[[which.model]], xout = atdf, rule = 2, method = "constant", f = approx.f)$y
        axis(side=3, at = atdf, labels = plotnz, tick=FALSE, line=0)
        title(main.txt, line = 2.5)
        
        
        
        # Adjust the margins to make sure the labels fit
        labwidth <- ifelse(labsize > 0, max(strwidth(rownames(nbeta[!remove,]), "inches", labsize)), 0)
        margins <- par("mai")
        par("mai" = c(margins[1:3], max(margins[4], labwidth*1.4)))
        if ( labsize > 0 && !is.null(rownames(nbeta)) ) 
        {
            take <- which(!remove)
            for (i in 1:sum(!remove)) {
                j <- take[i]
                axis(4, at = nbeta[j, ncol(nbeta)], labels = rownames(nbeta)[j],
                     las=1, cex.axis=labsize, col.axis=rainbow(sum(!remove))[i], 
                     lty = (i - 1) %% 5 + 1, col = rainbow(sum(!remove))[i])
            }
        }
        par("mai"=margins)
    } else if (type == "cv")
    {
        xlab=expression(log(lambda))
        if(sign.lambda<0)xlab=paste("-",xlab,sep="")
        plot.args=list(x    = sign.lambda * log(x$lambda),
                       y    = x$cvm[[which.model]],
                       ylim = range(x$cvup[[which.model]], x$cvlo[[which.model]]),
                       xlab = xlab,
                       ylab = x$name,
                       type = "n")
        new.args=list(...)
        if(length(new.args))plot.args[names(new.args)]=new.args
        do.call("plot", plot.args)
        error.bars(sign.lambda * log(x$lambda), 
                   x$cvup[[which.model]], 
                   x$cvlo[[which.model]], width = 0.005)
        points(sign.lambda*log(x$lambda), x$cvm[[which.model]], pch=20, col="dodgerblue")
        axis(side=3,at=sign.lambda*log(x$lambda),labels = paste(x$nzero[[which.model]]), tick=FALSE, line=0)
        abline(v = sign.lambda * log(x$lambda.min.models[which.model]), lty=2, lwd = 2, col = "firebrick1")
        abline(v = sign.lambda * log(x$lambda.1se.models[which.model]), lty=2, lwd = 2, col = "firebrick1")
        title(main.txt, line = 2.5)
    }
}



