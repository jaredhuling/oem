penalized_nonzero <- function(fit, rows) {
    b <- fit$beta[[1]][-1, , drop = FALSE]
    colSums(abs(b[rows, , drop = FALSE]) > 1e-10)
}

test_that("the default path starts where every penalized coefficient is zero", {
    set.seed(123)
    n <- 200
    p <- 10
    x <- matrix(rnorm(n * p), n)
    eta <- drop(x[, 1:3] %*% c(1, -0.8, 0.5))
    responses <- list(gaussian = eta + rnorm(n), binomial = rbinom(n, 1, plogis(eta)))
    factors <- list(rep(1, p), rep(2, p), rep(0.5, p), c(0, rep(1, p - 1)))
    for (family in names(responses)) {
        for (sparse in c(FALSE, TRUE)) {
            xx <- if (sparse) as(x, "CsparseMatrix") else x
            for (penalty in c("lasso", "elastic.net")) {
                for (pf in factors) {
                    fit <- oem(xx, responses[[family]], family = family, penalty = penalty,
                               alpha = 0.5, penalty.factor = pf, nlambda = 30)
                    nz <- penalized_nonzero(fit, which(pf > 0))
                    info <- paste(family, sparse, penalty, paste(pf, collapse = ","))
                    expect_equal(nz[1], 0, ignore_attr = TRUE, info = info)
                    expect_gt(nz[2], 0, label = info)
                }
            }
            for (penalty in c("grp.lasso", "sparse.grp.lasso")) {
                for (scale in c(1, 2, 0.5)) {
                    fit <- oem(xx, responses[[family]], family = family, penalty = penalty,
                               groups = rep(1:5, each = 2), group.weights = rep(sqrt(2) * scale, 5),
                               nlambda = 30)
                    nz <- penalized_nonzero(fit, 1:p)
                    info <- paste(family, sparse, penalty, scale)
                    expect_equal(nz[1], 0, ignore_attr = TRUE, info = info)
                    expect_gt(nz[2], 0, label = info)
                }
            }
        }
    }
})

test_that("cross validation uses the lambda sequence of the full data", {
    set.seed(123)
    x <- matrix(rnorm(300 * 8), 300)
    y <- drop(x[, 1:2] %*% c(1, -1)) + rnorm(300)
    pf <- c(0, rep(2, 7))
    full <- oem(x, y, penalty = "lasso", penalty.factor = pf)
    cv <- cv.oem(x, y, penalty = "lasso", penalty.factor = pf, nfolds = 5)
    # cv.oem drops values below the smallest lambda reached in every fold
    k <- length(cv$lambda[[1]])
    expect_gt(k, 90)
    expect_equal(cv$lambda[[1]], full$lambda[[1]][seq_len(k)])
    xv <- xval.oem(x, y, penalty = "lasso", penalty.factor = pf, nfolds = 5)
    expect_equal(penalized_nonzero(xv, 2:8)[1], 0, ignore_attr = TRUE)
})

test_that("oem.xtx starts its path where every penalized coefficient is zero", {
    set.seed(123)
    x <- matrix(rnorm(300 * 8), 300)
    y <- drop(x[, 1:2] %*% c(1, -1)) + rnorm(300)
    pf <- c(0, rep(1, 7))
    fit <- oem.xtx(crossprod(x), drop(crossprod(x, y)), penalty = "lasso", penalty.factor = pf)
    # oem.xtx has no intercept row
    b <- fit$beta[[1]]
    nz <- colSums(abs(b[pf > 0, , drop = FALSE]) > 1e-10)
    expect_equal(nz[1], 0, ignore_attr = TRUE)
    expect_gt(nz[2], 0)
})
