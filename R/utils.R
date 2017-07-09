

getmin <- function(lambda, cvm, cvsd){
    #modified from glmnet package
    
    lambda.min.models <- lambda.1se.models <- numeric(length(cvm))
    
    cv.models <- numeric(length(cvm))
    for (m in 1:length(cvm))
    {
        cvmin <- min(cvm[[m]])
        idmin <- cvm[[m]] <= cvmin
        lambda.min.models[m] <- max(lambda[[m]][idmin])
        cv.models[m] <- min(cvm[[m]][idmin])
        idmin <- match(lambda.min.models[m], lambda[[m]])
        semin <- (cvm[[m]] + cvsd[[m]])[idmin]
        idmin <- cvm[[m]] < semin
        lambda.1se.models[m] <- max(lambda[[m]][idmin])
    }
    mmin <- which.min(cv.models)
    lambda.min <- lambda.min.models[mmin]
    lambda.1se <- lambda.1se.models[mmin]
    
    list(lambda.min = lambda.min, model.min = mmin, lambda.1se = lambda.1se,
         lambda.min.models = lambda.min.models, lambda.1se.models = lambda.1se.models)
}

# taken from caret
createFolds <- function (y, k = 10, list = TRUE, returnTrain = FALSE) {
    #copied from caret package
    if (is.numeric(y)) {
        cuts <- floor(length(y)/k)
        if (cuts < 2) 
            cuts <- 2
        if (cuts > 5) 
            cuts <- 5
        y <- cut(y, unique(quantile(y, probs = seq(0, 1, length = cuts))), 
                 include.lowest = TRUE)
    }
    if (k < length(y)) {
        y <- factor(as.character(y))
        numInClass <- table(y)
        foldVector <- vector(mode = "integer", length(y))
        for (i in 1:length(numInClass)) {
            seqVector <- rep(1:k, numInClass[i]%/%k)
            if (numInClass[i]%%k > 0) 
                seqVector <- c(seqVector, sample(1:k, numInClass[i]%%k))
            foldVector[which(y == dimnames(numInClass)$y[i])] <- sample(seqVector)
        }
    }
    else foldVector <- seq(along = y)
    if (list) {
        out <- split(seq(along = y), foldVector)
        names(out) <- paste("Fold", gsub(" ", "0", format(seq(along = out))), 
                            sep = "")
        if (returnTrain) 
            out <- lapply(out, function(data, y) y[-data], y = seq(along = y))
    }
    else out <- foldVector
    out
}

# taken from glmnet
lambda.interp <- function(lambda, s) {
    ## copied from glmnet package
    ## interpolation of lambda according to s
    ## take lambda = sfrac * left + (1 - sfrac * right)
    
    if(length(lambda) == 1){ #degenerated case of one lambda given
        lens <- length(s)
        left <- rep(1, lens)
        right <- left
        sfrac <- rep(1, lens)
    } else {
        s[s > max(lambda)] <- max(lambda)
        s[s < min(lambda)] <- min(lambda)
        lenLambda <- length(lambda)
        sfrac <- (lambda[1]-s)/(lambda[1] - lambda[lenLambda])
        lambda <- (lambda[1] - lambda)/(lambda[1] - lambda[lenLambda])
        coord <- approx(lambda, seq(lambda), sfrac)$y
        left <- floor(coord)
        right <- ceiling(coord)
        sfrac <- (sfrac - lambda[right]) / (lambda[left] - lambda[right])
        sfrac[left==right] <- 1
    }
    list(left = left, right = right, frac = sfrac)
}

# modified from glmnet
auc <- function(y,prob,w)
{
    if(missing(w))
    {
        rprob <- rank(prob)
        n1    <- sum(y)
        n0    <- length(y) - n1
        u     <- sum(rprob[y==1]) - n1 * (n1 + 1) / 2
        
        exp(log(u) - log(n1) - log(n0))
    }
    else
    {
        rprob <- runif(length(prob))
        op    <- order(prob,rprob)#randomize ties
        y     <- y[op]
        w     <- w[op]
        cw    <- cumsum(w)
        w1    <- w[y==1]
        cw1   <- cumsum(w1)
        wauc  <- log(sum(w1*(cw[y==1] - cw1)))
        sumw1 <- cw1[length(cw1)]
        sumw2 <- cw[length(cw)] - sumw1
        
        exp(wauc - log(sumw1) - log(sumw2))
    }
}

# taken from glmnet
auc.mat=function(y,prob,weights=rep(1,nrow(y))){
    Weights=as.vector(weights*y)
    ny=nrow(y)
    Y=rep(c(0,1),c(ny,ny))
    Prob=c(prob,prob)
    auc(Y,Prob,Weights)
}

# modified from glmnet
cvcompute=function(mat,weights,foldid,nlams)
{
    ###Computes the weighted mean and SD within folds, and hence the se of the mean
    wisum=tapply(weights,foldid,sum)
    nfolds=max(foldid)
    outmat=matrix(NA,nfolds,ncol(mat))
    good=matrix(0,nfolds,ncol(mat))
    mat[is.infinite(mat)]=NA#just in case some infinities crept in
    for(i in seq(nfolds)){
        mati=mat[foldid==i,,drop=FALSE]
        wi=weights[foldid==i]
        outmat[i,]=apply(mati,2,weighted.mean,w=wi,na.rm=TRUE)
        good[i,seq(nlams[i])]=1
    }
    N=apply(good,2,sum)
    list(cvraw=outmat,weights=wisum,N=N)
}

# taken from glmnet
error.bars <- function(x, upper, lower, width = 0.02, ...)
{
    xlim <- range(x)
    barw <- diff(xlim) * width
    segments(x, upper, x, lower, col = 8, lty = 5, lwd = 0.5, ...)
    segments(x - barw, upper, x + barw, upper, col = "grey50", lwd = 1, ...)
    segments(x - barw, lower, x + barw, lower, col = "grey50", lwd = 1, ...)
    range(upper, lower)
}