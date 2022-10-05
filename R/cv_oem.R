


#' Cross validation for Orthogonalizing EM 
#'
#' @param x input matrix of dimension n x p or \code{CsparseMatrix} objects of the \pkg{Matrix} (sparse not yet implemented. 
#' Each row is an observation, each column corresponds to a covariate. The cv.oem() function
#' is optimized for n >> p settings and may be very slow when p > n, so please use other packages
#' such as \code{glmnet}, \code{ncvreg}, \code{grpreg}, or \code{gglasso} when p > n or p approx n.
#' @param y numeric response vector of length nobs.
#' @param penalty Specification of penalty type in lowercase letters. Choices include \code{"lasso"}, 
#' \code{"ols"} (Ordinary least squares, no penaly), \code{"elastic.net"}, \code{"scad"}, \code{"mcp"}, \code{"grp.lasso"}
#' @param weights observation weights. defaults to 1 for each observation (setting weight vector to 
#' length 0 will default all weights to 1)
#' @param lambda A user supplied lambda sequence. By default, the program computes
#' its own lambda sequence based on nlambda and lambda.min.ratio. Supplying
#' a value of lambda overrides this.
#' @param type.measure measure to evaluate for cross-validation. The default is \code{type.measure = "deviance"}, 
#' which uses squared-error for gaussian models (a.k.a \code{type.measure = "mse"} there), deviance for logistic
#' regression. \code{type.measure = "class"} applies to binomial only. \code{type.measure = "auc"} is for two-class logistic 
#' regression only. \code{type.measure = "mse"} or \code{type.measure = "mae"} (mean absolute error) can be used by all models;
#' they measure the deviation from the fitted mean to the response.
#' @param nfolds number of folds for cross-validation. default is 10. 3 is smallest value allowed. 
#' @param foldid an optional vector of values between 1 and nfold specifying which fold each observation belongs to.
#' @param grouped Like in \pkg{glmnet}, this is an experimental argument, with default \code{TRUE}, and can be ignored by most users. 
#' For all models, this refers to computing nfolds separate statistics, and then using their mean and estimated standard 
#' error to describe the CV curve. If \code{grouped = FALSE}, an error matrix is built up at the observation level from the 
#' predictions from the \code{nfold} fits, and then summarized (does not apply to \code{type.measure = "auc"}). 
#' @param keep If \code{keep = TRUE}, a prevalidated list of arrasy is returned containing fitted values for each observation 
#' and each value of lambda for each model. This means these fits are computed with this observation and the rest of its
#' fold omitted. The folid vector is also returned. Default is \code{keep = FALSE}
#' @param parallel If TRUE, use parallel foreach to fit each fold. Must register parallel before hand, such as \pkg{doMC}.
#' @param ncores Number of cores to use. If \code{parallel = TRUE}, then ncores will be automatically set to 1 to prevent conflicts
#' @param ... other parameters to be passed to \code{"oem"} function
#' @return An object with S3 class \code{"cv.oem"} 
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
#' fit <- cv.oem(x = x, y = y, 
#'               penalty = c("lasso", "grp.lasso"), 
#'               groups = rep(1:20, each = 5))
#' 
#' layout(matrix(1:2, ncol = 2))
#' plot(fit)
#' plot(fit, which.model = 2)
cv.oem <- function (x, y, penalty = c("elastic.net", 
                                      "lasso", 
                                      "ols", 
                                      "mcp",           "scad", 
                                      "mcp.net",       "scad.net",
                                      "grp.lasso",     "grp.lasso.net",
                                      "grp.mcp",       "grp.scad",
                                      "grp.mcp.net",   "grp.scad.net",
                                      "sparse.grp.lasso"),
                    weights = numeric(0), lambda = NULL, 
                    type.measure = c("mse", "deviance", "class", "auc", "mae"), nfolds = 10, foldid = NULL, 
                    grouped = TRUE, keep = FALSE, parallel = FALSE, ncores = -1, ...) 
{
    ## code modified from "glmnet" package
    
    this.call    <- match.call()

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
    
    if (missing(type.measure)) 
        type.measure = "default"
    else type.measure = match.arg(type.measure)
    if (!is.null(lambda) && length(lambda) < 2) 
        stop("Need more than one value of lambda for cv.oem")
    N = nrow(x)
    if (length(weights)) 
        weights = as.double(weights)
    y = drop(y)
    
    if (parallel & ncores != 1)
    {
        ncores <- 1
    }
    
    oem.call = match.call(expand.dots = TRUE)
    which = match(c("type.measure", "nfolds", "foldid", "grouped", 
                    "keep"), names(oem.call), FALSE)
    if (any(which)) 
        oem.call = oem.call[-which]
    oem.call[[1]] = as.name("oem")
    oem.object = oem(x, y, penalty = penalty, 
                     weights = weights, 
                     lambda = lambda, 
                     ncores = ncores, ...)
    oem.object$call = oem.call
    
    
    ###Next line is commented out so each call generates its own lambda sequence
    # lambda=oem.object$lambda
    #if (inherits(oem.object, "multnet") && !oem.object$grouped) {
    #    nz = predict(oem.object, type = "nonzero", which.model = m)
    #    nz = sapply(nz, function(x) sapply(x, length))
    #    nz = ceiling(apply(nz, 1, median))
    #}
    #else 
    nz = lapply(1:length(oem.object$beta), function(m) 
        sapply(predict(oem.object, type = "nonzero", which.model = m), length)
        )
    if (is.null(foldid)) 
        foldid = sample(rep(seq(nfolds), length = N))
    else nfolds = max(foldid)
    if (nfolds < 3) 
        stop("nfolds must be bigger than 3; nfolds=10 recommended")
    outlist = as.list(seq(nfolds))
    if (parallel) {
        outlist = foreach(i = seq(nfolds), .packages = c("oem")) %dopar% 
        {
            which = foldid == i
            if (is.matrix(y)) 
                y_sub = y[!which, ]
            else y_sub = y[!which]
            if (length(weights))
            {
                oem(x[!which, , drop = FALSE], y_sub, 
                    penalty = penalty, 
                    lambda = lambda, 
                    weights = weights[!which], 
                    ncores = 1, 
                    ...)
            } else 
            {
                oem(x[!which, , drop = FALSE], y_sub, 
                    penalty = penalty, 
                    lambda = lambda, 
                    ncores = 1, 
                    ...)
            }
        }
    }
    else {
        for (i in seq(nfolds)) {
            which = foldid == i
            if (is.matrix(y)) 
                y_sub = y[!which, ]
            else y_sub = y[!which]
            if (length(weights))
            {
                outlist[[i]] = oem(x[!which, , drop = FALSE], 
                                   y_sub, penalty = penalty, 
                                   lambda = lambda, 
                                   weights = weights[!which], 
                                   ncores = ncores, ...)
            } else 
            {
                outlist[[i]] = oem(x[!which, , drop = FALSE], 
                                   y_sub, penalty = penalty, 
                                   lambda = lambda, 
                                   ncores = ncores, ...)
            }
        }
    }
    fun <- paste("cv", class(oem.object)[[1]], sep = ".")
    lambda <- oem.object$lambda
    
    if (!length(weights))
    {
        weights <- rep(1, N)
    }
    
    cvstuff <- do.call(fun, list(outlist, lambda, x, y, weights, 
                                 foldid, type.measure, grouped, keep))
    cvm  <- cvstuff$cvm
    cvsd <- cvstuff$cvsd
    nas.list <- vector(mode = "list", length = length(cvm))
    for (m in 1:length(cvm))
    {
        nas.list[[m]] <- is.na(cvsd[[m]]) | is.nan(cvsd[[m]])
    }
    
    nas <- Reduce("+", nas.list) > 0
    if(any(nas)){
        
        for (m in 1:length(cvm))
        {
            cvm[[m]]    <- cvm[[m]][!nas]
            cvsd[[m]]   <- cvsd[[m]][!nas]
            nz[[m]]     <- nz[[m]][!nas]
            lambda[[m]] <- lambda[[m]][!nas]
        }
    }
    
    
    cvname = cvstuff$name
    out = list(lambda = lambda, cvm = cvm, cvsd = cvsd, 
               cvup = lapply(1:length(cvm), function(m) cvm[[m]] + cvsd[[m]]), 
               cvlo = lapply(1:length(cvm), function(m) cvm[[m]] - cvsd[[m]]), 
               nzero = nz, name = cvname, oem.fit = oem.object)
    if (keep) 
        out = c(out, list(fit.preval = cvstuff$fit.preval, foldid = foldid))
    lamin <- if(cvname == "AUC") getmin(lambda, lapply(cvm, function(ccvvmm) -ccvvmm), cvsd)
    else getmin(lambda, cvm, cvsd)
    obj <- c(out, as.list(lamin))
    obj$best.model <- penalty[obj$model.min]
    obj$penalty <- penalty
    class(obj) <- "cv.oem"
    obj
}


cv.oemfit_binomial <- function (outlist, lambda, x, y, weights, foldid, type.measure, grouped, keep = FALSE) 
{
    ## code modified from "glmnet" package
    typenames = c(mse = "Mean-Squared Error", mae = "Mean Absolute Error", 
                  deviance = "Binomial Deviance", auc = "AUC", class = "Misclassification Error")
    if (type.measure == "default") 
        type.measure = "deviance"
    if (!match(type.measure, c("mse", "mae", "deviance", "auc", 
                               "class"), FALSE)) 
    {
        warning("Only 'deviance', 'class', 'auc', 'mse' or 'mae'  available for binomial models; 'deviance' used")
        type.measure = "deviance"
    }
    prob_min = 1e-05
    prob_max = 1 - prob_min
    nc = dim(y)
    if (is.null(nc)) 
    {
        y    <- as.factor(y)
        ntab <- table(y)
        nc   <- as.integer(length(ntab))
        y    <- diag(nc)[as.numeric(y), ]
    }
    N <- nrow(y)
    nfolds <- max(foldid)
    nmodels <- length(outlist[[1]]$beta)
    if ((N/nfolds < 10) && type.measure == "auc") 
    {
        warning("Too few (< 10) observations per fold for type.measure='auc' in cv.lognet; changed to type.measure='deviance'. Alternatively, use smaller value for nfolds", 
                call. = FALSE)
        type.measure = "deviance"
    }
    if ((N/nfolds < 3) && grouped) 
    {
        warning("Option grouped=FALSE enforced in cv.glmnet, since < 3 observations per fold", 
                call. = FALSE)
        grouped = FALSE
    }
    
    mlami <- which_lam <- vector(mode = "list", length = nmodels)
    
    for (m in 1:nmodels)
    {
        mlami[[m]]     <- max(sapply(outlist, function(obj) min(obj$lambda[[m]])))
        which_lam[[m]] <- lambda[[m]] >= mlami[[m]]
    }
    
    predmat  <- matrix(NA, nrow(y), length(lambda[[1]]))
    predlist <- rep(list(predmat), nmodels)
    nlams    <- double(nfolds)
    for (i in seq(nfolds)) 
    {
        which  <- foldid == i
        fitobj <- outlist[[i]]
        for (m in 1:nmodels)
        {
            preds <- predict(fitobj, x[which, , drop = FALSE], s = lambda[[m]][which_lam[[m]]], 
                             which.model = m,
                             type = "response")
            nlami <- sum(which_lam[[m]])
            predlist[[m]][which, seq(nlami)] = preds
        }
        nlams[i] = nlami
    }
    if (type.measure == "auc") 
    {
        cvraw <- rep(list(matrix(NA, nfolds, length(lambda[[1]]))), nmodels)
        N     <- vector(mode = "list", length = nmodels)
        for (m in 1:nmodels)
        {
            good <- matrix(0, nfolds, length(lambda[[1]]))
            for (i in seq(nfolds)) 
            {
                good[i, seq(nlams[i])] = 1
                which <- foldid == i
                for (j in seq(nlams[i])) 
                {
                    cvraw[[m]][i, j] = auc.mat(y[which, ], predlist[[m]][which, j], weights[which])
                }
            }
            N[[m]] = apply(good, 2, sum)
        }
        weights = tapply(weights, foldid, sum)
        weights = rep(list(weights), nmodels)
    } else 
    {
        ywt <- apply(y, 1, sum)
        y <- y / ywt
        weights = weights * ywt
        N <- lapply(1:nmodels, function(xx) nrow(y) - apply(is.na(predlist[[xx]]), 2, sum))
        
        cvraw = lapply(1:nmodels, function(xx) 
            cvraw = switch(type.measure, mse = (y[, 1] - (1 - predlist[[xx]]))^2 + 
                               (y[, 2] - predlist[[xx]])^2, 
                           mae = abs(y[, 1] - (1 - predlist[[xx]])) + 
                               abs(y[, 2] - predlist[[xx]]), 
                           deviance = {
                               predmat = pmin(pmax(predlist[[xx]], prob_min), prob_max)
                               lp = y[, 1] * log(1 - predmat) + y[, 2] * log(predmat)
                               ly = log(y)
                               ly[y == 0] = 0
                               ly = drop((y * ly) %*% c(1, 1))
                               2 * (ly - lp)
                           }, class = y[, 1] * (predlist[[xx]] > 0.5) + y[, 2] * (predlist[[xx]] <= 0.5))
        )
        if (grouped) {
            cvob    <- lapply(1:nmodels, function(xx) cvcompute(cvraw[[xx]], weights, foldid, nlams))
            cvraw   <- lapply(cvob, function(x) x$cvraw)
            weights <- lapply(cvob, function(x) x$weights)
            N       <- lapply(cvob, function(x) x$N)
        } else 
        {
            weights <- rep(list(weights), nmodels)
        }
    }
    cvm  <- lapply(1:length(cvraw), function(m) apply(cvraw[[m]], 2, weighted.mean, w = weights[[m]], na.rm = TRUE))
    cvsd <- lapply(1:length(cvraw), function(m) sqrt(apply(scale(cvraw[[m]], cvm[[m]], FALSE)^2, 2, weighted.mean, 
                                                   w = weights[[m]], na.rm = TRUE)/(N[[m]] - 1)))
    out  <- list(cvm = cvm, cvsd = cvsd, name = typenames[type.measure])
    if (keep) 
        out$fit.preval = predlist
    out
}


cv.oemfit_gaussian <- function (outlist, lambda, x, y, weights, foldid, type.measure, grouped, keep = FALSE) 
{
    ## code modified from "glmnet" package
    typenames = c(deviance = "Mean-Squared Error", mse = "Mean-Squared Error", 
                  mae = "Mean Absolute Error")
    if (type.measure == "default") 
        type.measure = "mse"
    if (!match(type.measure, c("mse", "mae", "deviance"), FALSE)) {
        warning("Only 'mse', 'deviance' or 'mae'  available for Gaussian models; 'mse' used")
        type.measure = "mse"
    }
    
    nmodels <- length(outlist[[1]]$beta)
    
    ## We dont want to extrapolate smaller lambdas
    mlami <- which_lam <- vector(mode = "list", length = nmodels)
    
    for (m in 1:nmodels)
    {
        mlami[[m]]     <- max(sapply(outlist, function(obj) min(obj$lambda[[m]])))
        which_lam[[m]] <- lambda[[m]] >= mlami[[m]]
    }
    
    predmat  <- matrix(NA, length(y), length(lambda[[1]]))
    predlist <- rep(list(predmat), nmodels)
    nfolds   <- max(foldid)
    nlams    <- double(nfolds)
    for (i in seq(nfolds)) 
    {
        which  <- foldid == i
        fitobj <- outlist[[i]]
        x.tmp  <- x[which, , drop = FALSE]
        for (m in 1:nmodels)
        {
            preds <- predict(fitobj, x.tmp, s = lambda[[m]][which_lam[[m]]], which.model = m)
            
            
            nlami <- sum(which_lam[[m]])
            predlist[[m]][which, seq(nlami)] <- preds
        }
        nlams[i] <- nlami
    }
    
    N <- lapply(1:nmodels, function(xx) length(y) - apply(is.na(predlist[[xx]]), 2, sum))
    cvraw = switch(type.measure, mse = (y - predmat)^2, deviance = (y - 
                                                                        predmat)^2, mae = abs(y - predmat))
    
    
    cvraw = lapply(1:nmodels, function(xx) 
        switch(type.measure, mse = (y - predlist[[xx]])^2, 
               deviance = (y - predlist[[xx]])^2, 
               mae = abs(y - predlist[[xx]]))
    )
    
    if ((length(y)/nfolds < 3) && grouped) 
    {
        warning("Option grouped=FALSE enforced in cv.glmnet, since < 3 observations per fold", 
                call. = FALSE)
        grouped = FALSE
    }
    if (grouped) 
    {
        cvob    <- lapply(1:nmodels, function(xx) cvcompute(cvraw[[xx]], weights, foldid, nlams))
        cvraw   <- lapply(cvob, function(x) x$cvraw)
        weights <- lapply(cvob, function(x) x$weights)
        N       <- lapply(cvob, function(x) x$N)
    }
    cvm  <- lapply(1:length(cvraw), function(m) apply(cvraw[[m]], 2, weighted.mean, w = weights[[m]], na.rm = TRUE))
    cvsd <- lapply(1:length(cvraw), function(m) sqrt(apply(scale(cvraw[[m]], cvm[[m]], FALSE)^2, 2, weighted.mean, 
                                                           w = weights[[m]], na.rm = TRUE)/(N[[m]] - 1)))
    out  <- list(cvm = cvm, cvsd = cvsd, name = typenames[type.measure])
    if (keep) 
        out$fit.preval <- predlist
    out
}