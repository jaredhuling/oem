cv.oem <- function (x, y, weights, lambda = NULL, 
                    type.measure = c("mse", "deviance", "class", "auc", "mae"), nfolds = 10, foldid, 
                    grouped = TRUE, keep = FALSE, parallel = FALSE, ...) 
{
    if (missing(type.measure)) 
        type.measure = "default"
    else type.measure = match.arg(type.measure)
    if (!is.null(lambda) && length(lambda) < 2) 
        stop("Need more than one value of lambda for cv.oem")
    N = nrow(x)
    if (missing(weights)) 
        weights = rep(1, N)
    else weights = as.double(weights)
    y = drop(y)
    oem.call = match.call(expand.dots = TRUE)
    which = match(c("type.measure", "nfolds", "foldid", "grouped", 
                    "keep"), names(oem.call), F)
    if (any(which)) 
        oem.call = oem.call[-which]
    oem.call[[1]] = as.name("oem")
    oem.object = oem(x, y, weights = weights, 
                     lambda = lambda, ...)
    oem.object$call = oem.call
    
    ###Next line is commented out so each call generates its own lambda sequence
    # lambda=oem.object$lambda
    if (inherits(oem.object, "multnet") && !oem.object$grouped) {
        nz = predict(oem.object, type = "nonzero")
        nz = sapply(nz, function(x) sapply(x, length))
        nz = ceiling(apply(nz, 1, median))
    }
    else nz = sapply(predict(oem.object, type = "nonzero"), 
                     length)
    if (missing(foldid)) 
        foldid = sample(rep(seq(nfolds), length = N))
    else nfolds = max(foldid)
    if (nfolds < 3) 
        stop("nfolds must be bigger than 3; nfolds=10 recommended")
    outlist = as.list(seq(nfolds))
    if (parallel) {
        #  if (parallel && require(foreach)) {
        outlist = foreach(i = seq(nfolds), .packages = c("oem")) %dopar% 
        {
            which = foldid == i
            if (is.matrix(y)) 
                y_sub = y[!which, ]
            else y_sub = y[!which]
            oem(x[!which, , drop = FALSE], y_sub, lambda = lambda, 
                weights = weights[!which], 
                ...)
        }
    }
    else {
        for (i in seq(nfolds)) {
            which = foldid == i
            if (is.matrix(y)) 
                y_sub = y[!which, ]
            else y_sub = y[!which]
            outlist[[i]] = oem(x[!which, , drop = FALSE], 
                               y_sub, lambda = lambda, 
                               weights = weights[!which], ...)
        }
    }
    fun = paste("cv", class(oem.object)[[1]], sep = ".")
    lambda = oem.object$lambda
    cvstuff = do.call(fun, list(outlist, lambda, x, y, weights, 
                                foldid, type.measure, grouped, keep))
    cvm = cvstuff$cvm
    cvsd = cvstuff$cvsd
    nas=is.na(cvsd)
    if(any(nas)){
        lambda=lambda[!nas]
        cvm=cvm[!nas]
        cvsd=cvsd[!nas]
        nz=nz[!nas]
    }
    cvname = cvstuff$name
    out = list(lambda = lambda, cvm = cvm, cvsd = cvsd, cvup = cvm + 
                   cvsd, cvlo = cvm - cvsd, nzero = nz, name = cvname, oem.fit = oem.object)
    if (keep) 
        out = c(out, list(fit.preval = cvstuff$fit.preval, foldid = foldid))
    lamin=if(cvname=="AUC")getmin(lambda,-cvm,cvsd)
    else getmin(lambda, cvm, cvsd)
    obj = c(out, as.list(lamin))
    class(obj) = "cv.oem"
    obj
}


cv.oemfit_binomial <- function (outlist, lambda, x, y, weights, foldid, type.measure, grouped, keep = FALSE) 
{
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
    mlami=max(sapply(outlist,function(obj)min(obj$lambda)))
    which_lam=lambda >= mlami
    
    predmat = matrix(NA, nrow(y), length(lambda))
    nlams = double(nfolds)
    for (i in seq(nfolds)) {
        which = foldid == i
        fitobj = outlist[[i]]
        preds = predict(fitobj,x[which, , drop = FALSE], s=lambda[which_lam], 
                        type = "response")
        nlami = sum(which_lam)
        predmat[which, seq(nlami)] = preds
        nlams[i] = nlami
    }
    if (type.measure == "auc") {
        cvraw = matrix(NA, nfolds, length(lambda))
        good = matrix(0, nfolds, length(lambda))
        for (i in seq(nfolds)) {
            good[i, seq(nlams[i])] = 1
            which = foldid == i
            for (j in seq(nlams[i])) {
                cvraw[i, j] = auc.mat(y[which, ], predmat[which, 
                                                          j], weights[which])
            }
        }
        N = apply(good, 2, sum)
        weights = tapply(weights, foldid, sum)
    }
    else {
        ywt = apply(y, 1, sum)
        y = y/ywt
        weights = weights * ywt
        N = nrow(y) - apply(is.na(predmat), 2, sum)
        cvraw = switch(type.measure, mse = (y[, 1] - (1 - predmat))^2 + 
                           (y[, 2] - predmat)^2, mae = abs(y[, 1] - (1 - predmat)) + 
                           abs(y[, 2] - predmat), deviance = {
                               predmat = pmin(pmax(predmat, prob_min), prob_max)
                               lp = y[, 1] * log(1 - predmat) + y[, 2] * log(predmat)
                               ly = log(y)
                               ly[y == 0] = 0
                               ly = drop((y * ly) %*% c(1, 1))
                               2 * (ly - lp)
                           }, class = y[, 1] * (predmat > 0.5) + y[, 2] * (predmat <= 
                                                                               0.5))
        if (grouped) {
            cvob = cvcompute(cvraw, weights, foldid, nlams)
            cvraw = cvob$cvraw
            weights = cvob$weights
            N = cvob$N
        }
    }
    cvm = apply(cvraw, 2, weighted.mean, w = weights, na.rm = TRUE)
    cvsd = sqrt(apply(scale(cvraw, cvm, FALSE)^2, 2, weighted.mean, 
                      w = weights, na.rm = TRUE)/(N - 1))
    out = list(cvm = cvm, cvsd = cvsd, name = typenames[type.measure])
    if (keep) 
        out$fit.preval = predmat
    out
}


cv.oemfit_gaussian <- function (outlist, lambda, x, y, weights, foldid, type.measure, grouped, keep = FALSE) 
{
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
    
    predmat = matrix(NA, length(y), length(lambda))
    nfolds = max(foldid)
    nlams = double(nfolds)
    for (i in seq(nfolds)) {
        which = foldid == i
        fitobj = outlist[[i]]
        preds = predict(fitobj, x[which, , drop = FALSE], s=lambda[which_lam])
        nlami = sum(which_lam)
        predmat[which, seq(nlami)] = preds
        nlams[i] = nlami
    }
    N = length(y) - apply(is.na(predmat), 2, sum)
    cvraw = switch(type.measure, mse = (y - predmat)^2, deviance = (y - 
                                                                        predmat)^2, mae = abs(y - predmat))
    if ((length(y)/nfolds < 3) && grouped) {
        warning("Option grouped=FALSE enforced in cv.glmnet, since < 3 observations per fold", 
                call. = FALSE)
        grouped = FALSE
    }
    if (grouped) {
        cvob = cvcompute(cvraw, weights, foldid, nlams)
        cvraw = cvob$cvraw
        weights = cvob$weights
        N = cvob$N
    }
    cvm = apply(cvraw, 2, weighted.mean, w = weights, na.rm = TRUE)
    cvsd = sqrt(apply(scale(cvraw, cvm, FALSE)^2, 2, weighted.mean, 
                      w = weights, na.rm = TRUE)/(N - 1))
    out = list(cvm = cvm, cvsd = cvsd, name = typenames[type.measure])
    if (keep) 
        out$fit.preval = predmat
    out
}