## the code here is largely based on the code
## from the glmnet package (no reason to reinvent the wheel)

#' Prediction method for Orthogonalizing EM fitted objects
#'
#' @param object fitted "oem" model object
#' @param newx Matrix of new values for \code{x} at which predictions are to be made. Must be a matrix; can be sparse as in the 
#' \code{CsparseMatrix} objects of the \pkg{Matrix} package. 
#' This argument is not used for \code{type=c("coefficients","nonzero")}
#' @param s Value(s) of the penalty parameter lambda at which predictions are required. Default is the entire sequence used to create 
#' the model.
#' @param which.model If multiple penalties are fit and returned in the same oem object, the \code{which.model} argument is used to 
#' specify which model to make predictions for. For example, if the oem object \code{oemobj} was fit with argument 
#' \code{penalty = c("lasso", "grp.lasso")}, then which.model = 2 provides predictions for the group lasso model.
#' @param type Type of prediction required. \code{type = "link"} gives the linear predictors for the \code{"binomial"} model; for \code{"gaussian"} models it gives the fitted values. 
#' \code{type = "response"} gives the fitted probabilities for \code{"binomial"}. \code{type = "coefficients"} computes the coefficients at the requested values for \code{s}.
#' \code{type = "class"} applies only to \code{"binomial"} and produces the class label corresponding to the maximum probability.
#' @param ... not used 
#' @importFrom graphics abline abline axis matplot mtext points segments 
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
predict.oem <- function(object, newx, s = NULL, which.model = 1,
                        type = c("link",
                                 "response",
                                 "coefficients",
                                 "nonzero",
                                 "class"), ...) 
{
    type <- match.arg(type)
    
    num.models <- length(object$beta)
    which.model <- which.model[1]
    pen.names   <- names(object$beta)
    
    if (!is.character(which.model))
    {
        if (which.model > num.models)
        {
            err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
            stop(err.txt)
        }
    } else 
    {
        if (!(which.model %in% pen.names))
        {
            err.txt <- paste0("Model ", which.model, " specified, but ", which.model, " not computed.")
            stop(err.txt)
        }
        which.model <- match(which.model, pen.names)
    }
    
    
    if(missing(newx)){
        if(!match(type, c("coefficients", "nonzero"), FALSE))stop("A value for 'newx' must be supplied")
    }
    nbeta <- object$beta[[which.model]]
    
    if(!is.null(s))
    {
        #vnames=dimnames(nbeta)[[1]]
        lambda <- object$lambda[[which.model]]
        lamlist <- lambda.interp(object$lambda[[which.model]], s)
        nbeta <- nbeta[,lamlist$left,drop=FALSE]*lamlist$frac +nbeta[,lamlist$right,drop=FALSE]*(1-lamlist$frac)
        #dimnames(nbeta)=list(vnames,paste(seq(along=s)))
    }
    if (type == "coefficients") return(nbeta)
    if (type == "nonzero") 
    {
        nbeta[1,] <- 0 ## rem intercept
        newbeta <- abs(as.matrix(nbeta)) > 0
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
#' specify which model to plot. For example, if the oem object \code{"oemobj"} was fit with argument 
#' \code{penalty = c("lasso", "grp.lasso")}, then \code{which.model = 2} provides a plot for the group lasso model.
#' @param xvar What is on the X-axis. \code{"norm"} plots against the L1-norm of the coefficients, \code{"lambda"} against the log-lambda sequence, and \code{"dev"}
#' against the percent deviance explained.
#' @param labsize size of labels for variable names. If labsize = 0, then no variable names will be plotted
#' @param xlab label for x-axis
#' @param ylab label for y-axis
#' @param main main title for plot
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
plot.oem <- function(x, which.model = 1,
                     xvar = c("norm", "lambda", "loglambda", "dev"),
                     labsize = 0.6,
                     xlab = iname, ylab = NULL, 
                     main = x$penalty[which.model],
                     ...) 
{
    num.models <- length(x$beta)
    
    
    which.model <- which.model[1]
    pen.names   <- names(x$beta)
    
    if (!is.character(which.model))
    {
        if (which.model > num.models)
        {
            err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
            stop(err.txt)
        }
    } else 
    {
        if (!(which.model %in% pen.names))
        {
            err.txt <- paste0("Model ", which.model, " specified, but ", which.model, " not computed.")
            stop(err.txt)
        }
        which.model <- match(which.model, pen.names)
    }
    
    

    xvar <- match.arg(xvar)
    nbeta <- as.matrix(x$beta[[which.model]][-1,]) ## remove intercept
    remove <- apply(nbeta, 1, function(betas) all(betas == 0) )
    switch(xvar,
           "norm" = {
               index    <- apply(abs(nbeta), 2, sum)
               iname    <- expression(L[1] * " Norm")
               xlim     <- range(index)
               approx.f <- 1
           },
           "lambda" = {
               index    <- x$lambda[[which.model]]
               iname    <- expression(lambda)
               xlim     <- rev(range(index))
               approx.f <- 0
           },
           "loglambda" = {
               index    <- log(x$lambda[[which.model]])
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
    
    
    cols <- rainbow(sum(!remove))
    
    ## create sequence that grabs one of ROYGBIV and repeats with
    ## an increment up the rainbow spectrum with each step from 1:7 on ROYGBIV
    n.cols <- 7L
    scramble.seq <- rep(((1:n.cols) - 1) * (length(cols) %/% (n.cols)) + 1, length(cols) %/% n.cols)[1:length(cols)] + 
        (((0:(length(cols)-1)) %/% n.cols))
    
    scramble.seq[is.na(scramble.seq)] <- which(!(1:length(cols) %in% scramble.seq))
    colseq <- cols[scramble.seq]
    

    matplot(index, t(nbeta[!remove,,drop=FALSE]), 
            lty = 1, 
            xlab = xlab, 
            ylab = "",
            col = colseq,
            xlim = xlim,
            type = 'l', ...)
    
    if (is.null(ylab)) 
    {
        mtext(expression(hat(beta)), side = 2, cex = par("cex"), line = 3, las = 1)
    } else 
    {
        mtext(ylab, side = 2, cex = par("cex"), line = 3)
        ylab = ""
    }
    
    atdf <- pretty(index, n = 10L)
    plotnz <- approx(x = index, y = x$nzero[[which.model]], xout = atdf, rule = 2, method = "constant", f = approx.f)$y
    axis(side=3, at = atdf, labels = plotnz, tick=FALSE, line=0, ...)
    
    title(main, line = 2.5, ...)
    
    
    
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
                 las=1, cex.axis=labsize, col.axis = colseq[i], 
                 lty = (i - 1) %% 5 + 1, col = colseq[i], ...)
        }
    }
    par("mai"=margins)
}


#' @param sign.lambda Either plot against log(lambda) (default) or its negative if \code{sign.lambda = -1}.
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
#' plot(fit, which.model = "grp.lasso")
#' 
plot.cv.oem <- function(x, which.model = 1, sign.lambda = 1, ...)
{
    # modified from glmnet
    object = x
    num.models <- length(object$cvm)
    
    which.model <- which.model[1]
    pen.names   <- names(object$oem.fit$beta)
    
    if (!is.character(which.model))
    {
        if (which.model > num.models)
        {
            err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
            stop(err.txt)
        }
    } else 
    {
        if (!(which.model %in% pen.names))
        {
            err.txt <- paste0("Model ", which.model, " specified, but ", which.model, " not computed.")
            stop(err.txt)
        }
        which.model <- match(which.model, pen.names)
    }
    
    main.txt <- x$oem.fit$penalty[which.model]
    
    xlab <- expression(log(lambda))
    if(sign.lambda<0)xlab=paste("-",xlab,sep="")
    plot.args=list(x    = sign.lambda * log(object$lambda[[which.model]]),
                   y    = object$cvm[[which.model]],
                   ylim = range(object$cvup[[which.model]], object$cvlo[[which.model]]),
                   xlab = xlab,
                   ylab = object$name,
                   type = "n")
    new.args=list(...)
    if(length(new.args))plot.args[names(new.args)]=new.args
    do.call("plot", plot.args)
    error.bars(sign.lambda * log(object$lambda[[which.model]]), 
               object$cvup[[which.model]], 
               object$cvlo[[which.model]], width = 0.005)
    points(sign.lambda*log(object$lambda[[which.model]]), object$cvm[[which.model]], pch=20, col="dodgerblue")
    axis(side=3,at=sign.lambda*log(object$lambda[[which.model]]),labels = paste(object$nzero[[which.model]]), tick=FALSE, line=0, ...)
    abline(v = sign.lambda * log(object$lambda.min.models[which.model]), lty=2, lwd = 2, col = "firebrick1")
    abline(v = sign.lambda * log(object$lambda.1se.models[which.model]), lty=2, lwd = 2, col = "firebrick1")
    title(main.txt, line = 2.5, ...)
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

#' @export 
predict.oemfit_xval_gaussian <- function(object, newx, s = NULL, which.model = 1,
                                         type = c("link", 
                                                  "response",
                                                  "coefficients",
                                                  "nonzero"), ...)
{
    NextMethod("predict")
} 


#' @export
predict.oemfit_xval_binomial <- function(object, newx, s=NULL, which.model = 1,
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
#' @param which.model If multiple penalties are fit and returned in the same \code{oem} object, the \code{which.model} argument is used to 
#' specify which model to plot. For example, if the oem object \code{"oemobj"} was fit with argument 
#' \code{penalty = c("lasso", "grp.lasso")}, then \code{which.model = 2} provides a plot for the group lasso model.
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
#' fit <- oem(x = x, y = y, penalty = c("lasso", "mcp"), compute.loss = TRUE)
#'
#' logLik(fit)
#' 
#' logLik(fit, which.model = "mcp")
#'
logLik.oem <- function(object, which.model = 1, ...) {
    # taken from ncvreg. Thanks to Patrick Breheny.
    n <- as.numeric(object$nobs)
    
    num.models <- length(object$beta)
    
    which.model <- which.model[1]
    pen.names   <- names(object$beta)
    
    if (!is.character(which.model))
    {
        if (which.model > num.models)
        {
            err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
            stop(err.txt)
        }
    } else 
    {
        if (!(which.model %in% pen.names))
        {
            err.txt <- paste0("Model ", which.model, " specified, but ", which.model, " not computed.")
            stop(err.txt)
        }
        which.model <- match(which.model, pen.names)
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


#' log likelihood function for fitted cross validation \code{oem} objects
#'
#' @rdname logLik
#' @method logLik cv.oem
#' @export 
#' @examples
#'
#' fit <- cv.oem(x = x, y = y, penalty = c("lasso", "mcp"), compute.loss = TRUE,
#'               nlambda = 25)
#'
#' logLik(fit)
#' 
#' logLik(fit, which.model = "mcp")
#'
logLik.cv.oem <- function(object, which.model = 1, ...) {
    
    object <- object$oem.fit
    # taken from ncvreg. Thanks to Patrick Breheny.
    n <- as.numeric(object$nobs)
    
    num.models <- length(object$beta)
    
    which.model <- which.model[1]
    pen.names   <- names(object$beta)
    
    if (!is.character(which.model))
    {
        if (which.model > num.models)
        {
            err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
            stop(err.txt)
        }
    } else 
    {
        if (!(which.model %in% pen.names))
        {
            err.txt <- paste0("Model ", which.model, " specified, but ", which.model, " not computed.")
            stop(err.txt)
        }
        which.model <- match(which.model, pen.names)
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


#' log likelihood function for fitted cross validation \code{oem} objects
#'
#' @rdname logLik
#' @method logLik xval.oem
#' @export 
#' @examples
#'
#' fit <- xval.oem(x = x, y = y, penalty = c("lasso", "mcp"), compute.loss = TRUE, 
#'                 nlambda = 25)
#'
#' logLik(fit)
#' 
#' logLik(fit, which.model = "mcp")
#'
logLik.xval.oem <- function(object, which.model = 1, ...) {
    
    # taken from ncvreg. Thanks to Patrick Breheny.
    n <- as.numeric(object$nobs)
    
    num.models <- length(object$beta)
    
    which.model <- which.model[1]
    pen.names   <- names(object$beta)
    
    if (!is.character(which.model))
    {
        if (which.model > num.models)
        {
            err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
            stop(err.txt)
        }
    } else 
    {
        if (!(which.model %in% pen.names))
        {
            err.txt <- paste0("Model ", which.model, " specified, but ", which.model, " not computed.")
            stop(err.txt)
        }
        which.model <- match(which.model, pen.names)
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
#' @param object fitted \code{"cv.oem"} model object
#' @param newx Matrix of new values for \code{x} at which predictions are to be made. Must be a matrix; can be sparse as in the 
#' \code{CsparseMatrix} objects of the \pkg{Matrix} package
#' This argument is not used for \code{type = c("coefficients","nonzero")}
#' @param s Value(s) of the penalty parameter lambda at which predictions are required. Default is the entire sequence used to create 
#' the model. For \code{predict.cv.oem()}, can also specify \code{"lambda.1se"} or \code{"lambda.min"} for best lambdas estimated by cross validation
#' @param which.model If multiple penalties are fit and returned in the same \code{oem} object, the \code{which.model} argument is used to 
#' specify which model to make predictions for. For example, if the oem object \code{"oemobj"} was fit with argument 
#' \code{penalty = c("lasso", "grp.lasso")}, then \code{which.model = 2} provides predictions for the group lasso model. For 
#' \code{predict.cv.oem()}, can specify
#' \code{"best.model"} to use the best model as estimated by cross-validation
#' @param ... used to pass the other arguments for predict.oem
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
#' 
#' preds.gl <- predict(fit, newx = x.test, type = "response", which.model = "grp.lasso")
#' 
#' apply(preds.gl, 2, function(x) mean((y.test - x) ^ 2))
#' 
#' preds.l <- predict(fit, newx = x.test, type = "response", which.model = 1)
#' 
#' apply(preds.l, 2, function(x) mean((y.test - x) ^ 2))
predict.cv.oem <- function(object, newx, which.model = "best.model",
                           s=c("lambda.min", "lambda.1se"), ...)
{
    which.model <- which.model[1]
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
        if (which.model == "best.model")
        {
            mod.num <- object[["model.min"]]
        } else 
        {
            pen.names   <- names(object$oem.fit$beta)
            
            if (!(which.model %in% pen.names))
            {
                err.txt <- paste0("Model ", which.model, " specified, but ", which.model, " not computed.")
                stop(err.txt)
            }
            mod.num <- match(which.model, pen.names)
        }
    }
    
    else stop("Invalid form for s")
    predict(object$oem.fit, newx, s=lambda, which.model = mod.num, ...)
}



#' Prediction function for fitted cross validation oem objects
#'
#' @param object fitted "cv.oem" model object
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix; can be sparse as in the 
#' \code{CsparseMatrix} objects of the \pkg{Matrix} package
#' This argument is not used for type=c("coefficients","nonzero")
#' @param s Value(s) of the penalty parameter \code{lambda} at which predictions are required. Default is the entire sequence used to create 
#' the model. For predict.cv.oem, can also specify \code{"lambda.1se"} or \code{"lambda.min"} for best lambdas estimated by cross validation
#' @param which.model If multiple penalties are fit and returned in the same \code{oem} object, the \code{which.model} argument is used to 
#' specify which model to make predictions for. For example, if the oem object "oemobj" was fit with argument 
#' \code{penalty = c("lasso", "grp.lasso")}, then \code{which.model = 2} provides predictions for the group lasso model. For 
#' \code{predict.cv.oem()}, can specify
#' \code{"best.model"} to use the best model as estimated by cross-validation
#' @param ... used to pass the other arguments for \code{predict.oem()}
#' @return An object depending on the type argument
#' @method predict xval.oem
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
#' fit <- xval.oem(x = x, y = y, 
#'                 penalty = c("lasso", "grp.lasso"), 
#'                 groups = rep(1:10, each = 10), 
#'                 nlambda = 10)
#' 
#' preds.best <- predict(fit, newx = x.test, type = "response", which.model = "best.model")
#' 
#' apply(preds.best, 2, function(x) mean((y.test - x) ^ 2))
#' 
#' preds.gl <- predict(fit, newx = x.test, type = "response", which.model = "grp.lasso")
#' 
#' apply(preds.gl, 2, function(x) mean((y.test - x) ^ 2))
#' 
#' preds.l <- predict(fit, newx = x.test, type = "response", which.model = 1)
#' 
#' apply(preds.l, 2, function(x) mean((y.test - x) ^ 2))
predict.xval.oem <- function(object, newx, which.model = "best.model",
                             s = c("lambda.min", "lambda.1se"),...)
{
    
    which.model <- which.model[1]
    if(is.numeric(s))lambda=s
    else 
        if(is.character(s)){
            s=match.arg(s)
            lambda=object[[s]]
        }
    else stop("Invalid form for s")
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
        if (which.model == "best.model")
        {
            mod.num <- object[["model.min"]]
        } else 
        {
            pen.names   <- names(object$beta)
            
            if (!(which.model %in% pen.names))
            {
                err.txt <- paste0("Model ", which.model, " specified, but ", which.model, " not computed.")
                stop(err.txt)
            }
            mod.num <- match(which.model, pen.names)
        }
    }
    else stop("Invalid form for which.model")
    predict.oem(object, newx, s=lambda, which.model = mod.num, ...)
}






#' Plot method for Orthogonalizing EM fitted objects
#'
#' @param type one of \code{"cv"} or \code{"coefficients"}. \code{type = "cv"} will produce a plot of cross validation results like plot.cv.oem. 
#' \code{type = "coefficients"} will produce a coefficient path plot like \code{plot.oem()}
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
                          xlab = iname, ylab = NULL, 
                          main = x$penalty[which.model],
                          sign.lambda = 1,
                          ...) 
{
    type       <- match.arg(type)
    num.models <- length(x$beta)
    
    which.model <- which.model[1]
    pen.names   <- names(x$beta)
    
    if (!is.character(which.model))
    {
        if (which.model > num.models)
        {
            err.txt <- paste0("Model ", which.model, " specified, but only ", num.models, " were computed.")
            stop(err.txt)
        }
    } else 
    {
        if (!(which.model %in% pen.names))
        {
            err.txt <- paste0("Model ", which.model, " specified, but ", which.model, " not computed.")
            stop(err.txt)
        }
        which.model <- match(which.model, pen.names)
    }
    
    
    if (type == "coefficients")
    {
        xvar <- match.arg(xvar)
        nbeta <- as.matrix(x$beta[[which.model]][-1,]) ## remove intercept from plot
        remove <- apply(nbeta, 1, function(betas) all(betas == 0) )
        switch(xvar,
               "norm" = {
                   index    <- apply(abs(nbeta), 2, sum)
                   iname    <- expression(L[1] * " Norm")
                   xlim     <- range(index)
                   approx.f <- 1
               },
               "lambda" = {
                   index    <- x$lambda[[which.model]]
                   iname    <- expression(lambda)
                   xlim     <- rev(range(index))
                   approx.f <- 0
               },
               "loglambda" = {
                   index    <- log(x$lambda[[which.model]])
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
        
        cols <- rainbow(sum(!remove))
        
        ## create sequence that grabs one of ROYGBIV and repeats with
        ## an increment up the rainbow spectrum with each step from 1:7 on ROYGBIV
        n.cols <- 7L
        scramble.seq <- rep(((1:n.cols) - 1) * (length(cols) %/% (n.cols)) + 1, length(cols) %/% n.cols)[1:length(cols)] + 
            (((0:(length(cols)-1)) %/% n.cols))
        
        scramble.seq[is.na(scramble.seq)] <- which(!(1:length(cols) %in% scramble.seq))
        colseq <- cols[scramble.seq]
        
        
        matplot(index, t(nbeta[!remove,,drop=FALSE]), 
                lty = 1, 
                xlab = xlab, 
                ylab = "",
                col = colseq,
                xlim = xlim,
                type = 'l', ...)
        
        if (is.null(ylab)) 
        {
            mtext(expression(hat(beta)), side = 2, cex = par("cex"), line = 3, las = 1)
        } else 
        {
            mtext(ylab, side = 2, cex = par("cex"), line = 3)
            ylab = ""
        }
        
        
        atdf <- pretty(index, n = 10L)
        plotnz <- approx(x = index, y = x$nzero[[which.model]], xout = atdf, rule = 2, method = "constant", f = approx.f)$y
        axis(side=3, at = atdf, labels = plotnz, tick=FALSE, line=0, ...)
        title(main, line = 2.5, ...)
        
        
        
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
                     las=1, cex.axis=labsize, col.axis = colseq[i], 
                     lty = (i - 1) %% 5 + 1, col = colseq[i], ...)
            }
        }
        par("mai"=margins)
    } else if (type == "cv")
    {
        xlab=expression(log(lambda))
        if(sign.lambda<0)xlab=paste("-",xlab,sep="")
        plot.args=list(x    = sign.lambda * log(x$lambda[[which.model]]),
                       y    = x$cvm[[which.model]],
                       ylim = range(x$cvup[[which.model]], x$cvlo[[which.model]]),
                       xlab = xlab,
                       ylab = x$name,
                       type = "n")
        new.args=list(...)
        if(length(new.args))plot.args[names(new.args)]=new.args
        do.call("plot", plot.args)
        error.bars(sign.lambda * log(x$lambda[[which.model]]), 
                   x$cvup[[which.model]], 
                   x$cvlo[[which.model]], width = 0.005)
        points(sign.lambda*log(x$lambda[[which.model]]), x$cvm[[which.model]], pch=20, col="dodgerblue")
        axis(side=3,at=sign.lambda*log(x$lambda[[which.model]]),labels = paste(x$nzero[[which.model]]), tick=FALSE, line=0, ...)
        abline(v = sign.lambda * log(x$lambda.min.models[which.model]), lty=2, lwd = 2, col = "firebrick1")
        abline(v = sign.lambda * log(x$lambda.1se.models[which.model]), lty=2, lwd = 2, col = "firebrick1")
        title(main, line = 2.5, ...)
    }
}



#' summary method for cross validation Orthogonalizing EM fitted objects
#'
#' @param object fitted \code{"cv.oem"} object
#' @param ... not used
#' @rdname summary
#' @method summary cv.oem
#' @export
summary.cv.oem <- function(object, ...) {
    ## modified from ncvreg
    #S <- pmax(object$null.dev - object$cve, 0)
    #rsq <- S/object$null.dev
    #snr <- S/object$cve
    nvars <- lapply(object$oem.fit$beta, function(x) apply(x, 2, function(xx) sum(xx != 0)))
    model <- switch(object$oem.fit$family, gaussian="linear", binomial="logistic")
    val <- list(penalty=object$oem.fit$penalty, model=model, n=object$oem.fit$nobs, 
                p=object$oem.fit$nvars, lambda.min.models=object$lambda.min.models, 
                lambda=object$lambda, cve=object$cvm, nvars=nvars, type.measure = object$name)
    if (object$oem.fit$family=="gaussian") val$sigma <- lapply(object$cvm, sqrt)
    #if (object$oem.fit$family=="binomial") val$pe <- object$pe
    structure(val, class="summary.cv.oem")
}

#' summary method for cross validation Orthogonalizing EM fitted objects
#'
#' @rdname summary
#' @method summary xval.oem
#' @export
summary.xval.oem <- function(object, ...) {
    ## modified from ncvreg
    #S <- pmax(object$null.dev - object$cve, 0)
    #rsq <- S/object$null.dev
    #snr <- S/object$cve
    nvars <- lapply(object$beta, function(x) apply(x, 2, function(xx) sum(xx != 0)))
    model <- switch(object$family, gaussian="linear", binomial="logistic")
    val <- list(penalty=object$penalty, model=model, n=object$nobs, 
                p=object$nvars, lambda.min.models=object$lambda.min.models, 
                lambda=object$lambda, cve=object$cvm, nvars=nvars, type.measure = object$name)
    if (object$family=="gaussian") val$sigma <- lapply(object$cvm, sqrt)
    #if (object$oem.fit$family=="binomial") val$pe <- object$pe
    structure(val, class="summary.cv.oem")
}

#' print method for \code{summary.cv.oem} objects
#'
#' @param x a "summary.cv.oem" object
#' @param digits digits to display
#' @param ... not used
#' @rdname print
#' @method print summary.cv.oem
#' @export
print.summary.cv.oem <- function(x, digits, ...) {
    ## modified from ncvreg
    digits <- if (missing(digits)) digits <- c(2, 4, 2, 2, 3) else rep(digits, length.out=5)
    for (m in 1:length(x$penalty))
    {
        cat(x$penalty[m], "-penalized ", x$model, " regression with n=", x$n, ", p=", x$p, "\n", sep="")
        cat("At minimum cross-validation error (lambda=", formatC(x$lambda.min.models[m], digits[2], format="f"), "):\n", sep="")
        cat("-------------------------------------------------\n")
        cat("  Nonzero coefficients: ", x$nvars[[m]][ which.min(x$cve[[m]]) ], "\n", sep="")
        cat(paste0("  Cross-validation error (", x$type.measure, "): "), 
            formatC(min(x$cve[[m]]), digits[1], format="f"), "\n", sep="")
        #cat("  R-squared: ", formatC(max(x$r.squared), digits[3], format="f"), "\n", sep="")
        #cat("  Signal-to-noise ratio: ", formatC(max(x$snr), digits[4], format="f"), "\n", sep="")
        #if (x$model == "logistic") cat("  Prediction error: ", formatC(x$pe[x$min], digits[5], format="f"), "\n", sep="")
        if (x$model == "linear") cat("  Scale estimate (sigma): ", formatC(sqrt(x$cve[[m]][ which.min(x$cve[[m]]) ]), digits[5], format="f"), "\n\n", sep="")
        
        if (m < length(x$penalty))
            cat("<===============================================>\n\n")
    }
}
