test_that("big.oem and xval.oem agree with oem", {
    for (dims in list(c(300, 20), c(40, 60))) {
        set.seed(123)
        n <- dims[1]
        p <- dims[2]
        x <- matrix(rnorm(n * p), n)
        x[, 2] <- 4 * x[, 2] + 3
        x[, 3] <- x[, 3] + 2
        y <- drop(x[, 1:4] %*% c(0.8, -0.2, 0.5, 0.4)) + 1 + 2 * rnorm(n)
        lam <- if (p > n) c(0.4, 0.2) else c(0.1, 0.03)
        bx <- bigmemory::as.big.matrix(x)
        for (penalty in c("lasso", "elastic.net", "grp.lasso", "mcp")) {
            for (standardize in c(TRUE, FALSE)) {
                for (intercept in c(TRUE, FALSE)) {
                    args <- list(penalty = penalty, lambda = lam, alpha = 0.5,
                                 standardize = standardize, intercept = intercept,
                                 groups = rep(seq_len(p / 2), each = 2), tol = 1e-12, maxit = 1e6)
                    info <- paste(n, p, penalty, standardize, intercept)
                    ref <- suppressWarnings(do.call(oem, c(list(x, y), args)))$beta[[1]]
                    big <- suppressWarnings(do.call(big.oem, c(list(bx, y), args)))$beta[[1]]
                    expect_equal(big, ref, tolerance = 1e-7, info = info)
                    if (n > p) {
                        xv <- do.call(xval.oem, c(list(x, y, nfolds = 3), args))$beta[[1]]
                        expect_equal(xv, ref, tolerance = 1e-7, info = info)
                    }
                }
            }
        }
    }
})
