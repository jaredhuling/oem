


#' Orthogonalizing EM
#'
#' @param x input matrix or SparseMatrix (sparse not yet implemented. 
#' Each row is an observation, each column corresponds to a covariate
#' @param y numeric response vector of length nobs.
#' @param penalty Specification of penalty type in lowercase letters. Choices include "lasso", 
#' "ols" (Ordinary least squares, no penaly), "elastic.net", "scad", "mcp", "grp.lasso"
#' @param weights observation weights. defaults to 1 for each observation (setting weight vector to 
#' length 0 will default all weights to 1)
#' @param lambda A user supplied lambda sequence. By default, the program computes
#' its own lambda sequence based on nlambda and lambda.min.ratio. Supplying
#' a value of lambda overrides this.
#' @param type.measure measure to evaluate for cross-validation. The default is type.measure="deviance", 
#' which uses squared-error for gaussian models (a.k.a type.measure="mse" there), deviance for logistic
#' regression. type.measure="class" applies to binomial only. type.measure="auc" is for two-class logistic 
#' regression only. type.measure="mse" or type.measure="mae" (mean absolute error) can be used by all models;
#' they measure the deviation from the fitted mean to the response.
#' @param nfolds number of folds for cross-validation. default is 10. 3 is smallest value allowed. 
#' @param foldid an optional vector of values between 1 and nfold specifying which fold each observation belongs to.
#' @param grouped Like in glmnet, this is an experimental argument, with default TRUE, and can be ignored by most users. 
#' For all models, this refers to computing nfolds separate statistics, and then using their mean and estimated standard 
#' error to describe the CV curve. If grouped=FALSE, an error matrix is built up at the observation level from the 
#' predictions from the nfold fits, and then summarized (does not apply to type.measure="auc"). 
#' @param keep If keep=TRUE, a prevalidated list of arrasy is returned containing fitted values for each observation 
#' and each value of lambda for each model. This means these fits are computed with this observation and the rest of its
#' fold omitted. The folid vector is also returned. Default is keep=FALSE
#' @param parallel If TRUE, use parallel foreach to fit each fold. Must register parallel before hand, such as doMC.
#' @param ... other parameters to be passed to "oem" function
#' @return An object with S3 class "cv.oem" 
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
#' fit <- cv.oem(x = x, y = y, 
#'               penalty = c("lasso", "grp.lasso"), 
#'               groups = rep(1:20, each = 5))
#' 
#' layout(matrix(1:2, ncol = 2))
#' plot(fit)
#' plot(fit, which.model = 2)
cv.oem <- function (x, y, penalty = c("elastic.net", "lasso", "ols", "mcp", "scad", "grp.lasso"),
                    weights = numeric(0), lambda = NULL, 
                    type.measure = c("mse", "deviance", "class", "auc", "mae"), nfolds = 10, foldid = NULL, 
                    grouped = TRUE, keep = FALSE, parallel = FALSE, ...) 
{
    ## code modified from "glmnet" package
    penalty <- match.arg(penalty, several.ok = TRUE)
    if (missing(type.measure)) 
        type.measure = "default"
    else type.measure = match.arg(type.measure)
    if (!is.null(lambda) && length(lambda) < 2) 
        stop("Need more than one value of lambda for cv.oem")
    N = nrow(x)
    if (length(weights)) 
        weights = as.double(weights)
    y = drop(y)
    oem.call = match.call(expand.dots = TRUE)
    which = match(c("type.measure", "nfolds", "foldid", "grouped", 
                    "keep"), names(oem.call), FALSE)
    if (any(which)) 
        oem.call = oem.call[-which]
    oem.call[[1]] = as.name("oem")
    oem.object = oem(x, y, penalty = penalty, 
                     weights = weights, 
                     lambda = lambda, ...)
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
                    ...)
            } else 
            {
                oem(x[!which, , drop = FALSE], y_sub, 
                    penalty = penalty, 
                    lambda = lambda, 
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
                                   weights = weights[!which], ...)
            } else 
            {
                outlist[[i]] = oem(x[!which, , drop = FALSE], 
                                   y_sub, penalty = penalty, 
                                   lambda = lambda, ...)
            }
        }
    }
    fun = paste("cv", class(oem.object)[[1]], sep = ".")
    lambda = oem.object$lambda
    
    if (!length(weights))
    {
        weights <- rep(1, N)
    }
    
    cvstuff = do.call(fun, list(outlist, lambda, x, y, weights, 
                                foldid, type.measure, grouped, keep))
    cvm = cvstuff$cvm
    cvsd = cvstuff$cvsd
    nas.list <- vector(mode = "list", length = length(cvm))
    for (m in 1:length(cvm))
    {
        nas.list[[m]]=is.na(cvsd[[m]]) | is.nan(cvsd[[m]])
    }
    
    nas <- Reduce("+", nas.list) > 0
    if(any(nas)){
        lambda=lambda[!nas]
        for (m in 1:length(cvm))
        {
            cvm[[m]]=cvm[[m]][!nas]
            cvsd[[m]]=cvsd[[m]][!nas]
            nz[[m]]=nz[[m]][!nas]
        }
    }
    
    
    cvname = cvstuff$name
    out = list(lambda = lambda, cvm = cvm, cvsd = cvsd, 
               cvup = lapply(1:length(cvm), function(m) cvm[[m]] + cvsd[[m]]), 
               cvlo = lapply(1:length(cvm), function(m) cvm[[m]] - cvsd[[m]]), 
               nzero = nz, name = cvname, oem.fit = oem.object)
    if (keep) 
        out = c(out, list(fit.preval = cvstuff$fit.preval, foldid = foldid))
    lamin=if(cvname=="AUC")getmin(lambda,lapply(cvm, function(ccvvmm) -ccvvmm),cvsd)
    else getmin(lambda, cvm, cvsd)
    obj = c(out, as.list(lamin))
    class(obj) = "cv.oem"
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
                               "class"), FALSE)) {
        warning("Only 'deviance', 'class', 'auc', 'mse' or 'mae'  available for binomial models; 'deviance' used")
        type.measure = "deviance"
    }
    prob_min = 1e-05
    prob_max = 1 - prob_min
    nc = dim(y)
    if (is.null(nc)) {
        y = as.factor(y)
        ntab = table(y)
        nc = as.integer(length(ntab))
        y = diag(nc)[as.numeric(y), ]
    }
    N = nrow(y)
    nfolds = max(foldid)
    nmodels <- length(outlist[[1]]$beta)
    if ((N/nfolds < 10) && type.measure == "auc") {
        warning("Too few (< 10) observations per fold for type.measure='auc' in cv.lognet; changed to type.measure='deviance'. Alternatively, use smaller value for nfolds", 
                call. = FALSE)
        type.measure = "deviance"
    }
    if ((N/nfolds < 3) && grouped) {
        warning("Option grouped=FALSE enforced in cv.glmnet, since < 3 observations per fold", 
                call. = FALSE)
        grouped = FALSE
    }
    mlami=max(sapply(outlist, function(obj)min(obj$lambda)))
    which_lam=lambda >= mlami
    
    predmat = matrix(NA, nrow(y), length(lambda))
    predlist <- rep(list(predmat), nmodels)
    nlams = double(nfolds)
    for (i in seq(nfolds)) {
        which = foldid == i
        fitobj = outlist[[i]]
        for (m in 1:nmodels)
        {
            preds = predict(fitobj,x[which, , drop = FALSE], s=lambda[which_lam], 
                            which.model = m,
                            type = "response")
            nlami = sum(which_lam)
            predlist[[m]][which, seq(nlami)] = preds
        }
        nlams[i] = nlami
    }
    if (type.measure == "auc") {
        cvraw = rep(list(matrix(NA, nfolds, length(lambda))), nmodels)
        N = vector(mode = "list", length = nmodels)
        for (m in 1:nmodels)
        {
            good = matrix(0, nfolds, length(lambda))
            for (i in seq(nfolds)) {
                good[i, seq(nlams[i])] = 1
                which = foldid == i
                for (j in seq(nlams[i])) {
                    cvraw[[m]][i, j] = auc.mat(y[which, ], predlist[[m]][which, j], weights[which])
                }
            }
            N[[m]] = apply(good, 2, sum)
        }
        weights = tapply(weights, foldid, sum)
    }
    else {
        ywt = apply(y, 1, sum)
        y = y/ywt
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
            cvob = lapply(1:nmodels, function(xx) cvcompute(cvraw[[xx]], weights, foldid, nlams))
            cvraw = lapply(cvob, function(x) x$cvraw)
            weights = lapply(cvob, function(x) x$weights)
            N = lapply(cvob, function(x) x$N)
        }
    }
    cvm = lapply(1:length(cvraw), function(m) apply(cvraw[[m]], 2, weighted.mean, w = weights[[m]], na.rm = TRUE))
    cvsd = lapply(1:length(cvraw), function(m) sqrt(apply(scale(cvraw[[m]], cvm[[m]], FALSE)^2, 2, weighted.mean, 
                                                  w = weights[[m]], na.rm = TRUE)/(N[[m]] - 1)))
    out = list(cvm = cvm, cvsd = cvsd, name = typenames[type.measure])
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
    ##We dont want to extrapolate lambdas on the small side
    mlami=max(sapply(outlist,function(obj)min(obj$lambda)))
    which_lam=lambda >= mlami
    nmodels <- length(outlist[[1]]$beta)
    
    predmat = matrix(NA, length(y), length(lambda))
    predlist <- rep(list(predmat), nmodels)
    nfolds = max(foldid)
    nlams = double(nfolds)
    for (i in seq(nfolds)) 
    {
        which = foldid == i
        fitobj = outlist[[i]]
        x.tmp = x[which, , drop = FALSE]
        for (m in 1:nmodels)
        {
            preds = predict(fitobj, x.tmp, s=lambda[which_lam], which.model = m)
            
            
            nlami = sum(which_lam)
            predlist[[m]][which, seq(nlami)] = preds
        }
        nlams[i] = nlami
    }
    
    N <- lapply(1:nmodels, function(xx) length(y) - apply(is.na(predlist[[xx]]), 2, sum))
    cvraw = switch(type.measure, mse = (y - predmat)^2, deviance = (y - 
                                                                        predmat)^2, mae = abs(y - predmat))
    
    
    cvraw = lapply(1:nmodels, function(xx) 
        switch(type.measure, mse = (y - predlist[[xx]])^2, 
               deviance = (y - predlist[[xx]])^2, 
               mae = abs(y - predlist[[xx]]))
    )
    
    if ((length(y)/nfolds < 3) && grouped) {
        warning("Option grouped=FALSE enforced in cv.glmnet, since < 3 observations per fold", 
                call. = FALSE)
        grouped = FALSE
    }
    if (grouped) {
        cvob = lapply(1:nmodels, function(xx) cvcompute(cvraw[[xx]], weights, foldid, nlams))
        cvraw = lapply(cvob, function(x) x$cvraw)
        weights = lapply(cvob, function(x) x$weights)
        N = lapply(cvob, function(x) x$N)
    }
    cvm = lapply(1:length(cvraw), function(m) apply(cvraw[[m]], 2, weighted.mean, w = weights[[m]], na.rm = TRUE))
    cvsd = lapply(1:length(cvraw), function(m) sqrt(apply(scale(cvraw[[m]], cvm[[m]], FALSE)^2, 2, weighted.mean, 
                                                          w = weights[[m]], na.rm = TRUE)/(N[[m]] - 1)))
    out = list(cvm = cvm, cvsd = cvsd, name = typenames[type.measure])
    if (keep) 
        out$fit.preval = predlist
    out
}