test_that("lasso and elastic net fits agree with glmnet", {
    skip_if_not_installed("glmnet")
    for (family in c("gaussian", "binomial")) {
        for (dims in list(c(300, 20), c(40, 60))) {
            set.seed(123)
            n <- dims[1]
            p <- dims[2]
            x <- matrix(rnorm(n * p), n)
            x[, 2] <- 4 * x[, 2] + 3
            x[, 3] <- x[, 3] + 2
            eta <- drop(x[, 1:4] %*% c(0.8, -0.2, 0.5, 0.4))
            y <- if (family == "gaussian") eta + 1 + rnorm(n) else rbinom(n, 1, plogis(eta - 1))
            lam <- if (p > n) c(0.2, 0.12, 0.08) else c(0.08, 0.03, 0.01)
            # glmnet rescales penalty factors to sum to the number of variables
            pf <- c(0, 2, rep(1, p - 2))
            pf <- pf * p / sum(pf)
            for (sparse in c(FALSE, TRUE)) {
                xx <- if (sparse) as(x, "CsparseMatrix") else x
                for (standardize in c(TRUE, FALSE)) {
                    for (intercept in c(TRUE, FALSE)) {
                        for (alpha in c(1, 0.5)) {
                            ref <- glmnet::glmnet(x, y, family = family, alpha = alpha, lambda = lam,
                                                  standardize = standardize, intercept = intercept,
                                                  penalty.factor = pf, thresh = 1e-16, maxit = 1e7)
                            fit <- suppressWarnings(
                                oem(xx, y, family = family, alpha = alpha, lambda = lam,
                                    penalty = if (alpha == 1) "lasso" else "elastic.net",
                                    standardize = standardize, intercept = intercept,
                                    penalty.factor = pf, hessian.type = "full",
                                    tol = 1e-11, irls.tol = 1e-11, maxit = 2e5))
                            b <- fit$beta[[1]]
                            expect_equal(unname(b), unname(as.matrix(coef(ref))), tolerance = 1e-5,
                                         info = paste(family, n, p, sparse, standardize, intercept, alpha))
                        }
                    }
                }
            }
        }
    }
})
