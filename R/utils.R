
# taken from glmnet
getmin <- function(lambda, cvm, cvsd){
    #copied from glmnet package
    cvmin <- min(cvm)
    idmin <- cvm <= cvmin
    lambda.min <- max(lambda[idmin])
    idmin <- match(lambda.min, lambda)
    semin <- (cvm + cvsd)[idmin]
    idmin <- cvm < semin
    lambda.1se <- max(lambda[idmin])
    list(lambda.min = lambda.min, lambda.1se = lambda.1se)
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

# taken from glmnet
auc=function(y,prob,w){
    if(missing(w)){
        rprob=rank(prob)
        n1=sum(y);n0=length(y)-n1
        u=sum(rprob[y==1])-n1*(n1+1)/2
        u/(n1*n0)
    }
    else{
        rprob=runif(length(prob))
        op=order(prob,rprob)#randomize ties
        y=y[op]
        w=w[op]
        cw=cumsum(w)
        w1=w[y==1]
        cw1=cumsum(w1)
        wauc=sum(w1*(cw[y==1]-cw1))
        sumw=cw1[length(cw1)]
        sumw=sumw*(cw[length(cw)]-sumw)
        wauc/sumw
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
