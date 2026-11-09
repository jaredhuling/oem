make_data <- function(n, p, family) {
    set.seed(123)
    x <- matrix(rnorm(n * p), n)
    x[, 2] <- 4 * x[, 2] + 3
    x[, 3] <- x[, 3] + 2
    eta <- drop(x[, 1:4] %*% c(0.8, -0.2, 0.5, 0.4))
    y <- if (family == "gaussian") eta + 1 + rnorm(n) else rbinom(n, 1, plogis(eta - 1))
    list(x = x, y = y)
}

fit_oem <- function(x, y, ...) {
    suppressWarnings(oem(x, y, tol = 1e-11, irls.tol = 1e-11, maxit = 1e5, ...))
}

test_that("sparse and dense designs give the same fits", {
    lam <- c(0.1, 0.05)
    for (family in c("gaussian", "binomial")) {
        for (dims in list(c(200, 10), c(40, 60))) {
            dat <- make_data(dims[1], dims[2], family)
            groups <- rep(seq_len(dims[2] / 2), each = 2)
            for (standardize in c(TRUE, FALSE)) {
                for (intercept in c(TRUE, FALSE)) {
                    for (penalty in c("lasso", "elastic.net", "mcp", "grp.lasso", "sparse.grp.lasso")) {
                        # MCP is nonconvex, and for logistic models the two solvers can
                        # reach different local solutions
                        if (family == "binomial" && penalty == "mcp") next
                        args <- list(family = family, penalty = penalty, lambda = lam, alpha = 0.5,
                                     standardize = standardize, intercept = intercept, groups = groups)
                        dense <- do.call(fit_oem, c(list(dat$x, dat$y), args))
                        sparse <- do.call(fit_oem, c(list(as(dat$x, "CsparseMatrix"), dat$y), args))
                        expect_equal(sparse$beta[[1]], dense$beta[[1]], tolerance = 1e-6,
                                     info = paste(family, dims[1], dims[2], standardize, intercept, penalty))
                    }
                }
            }
        }
    }
})

test_that("unpenalized coefficients and the intercept are not shrunk by the elastic net", {
    set.seed(123)
    n <- 200
    x <- matrix(rnorm(n * 6), n)
    eta <- 0.5 + 1.5 * x[, 1]
    pf <- c(0, rep(1, 5))
    responses <- list(gaussian = eta + rnorm(n), binomial = rbinom(n, 1, plogis(eta)))
    for (family in names(responses)) {
        y <- responses[[family]]
        ref <- coef(glm(y ~ x[, 1], family = family))
        for (sparse in c(FALSE, TRUE)) {
            xx <- if (sparse) as(x, "CsparseMatrix") else x
            # lambda is large enough that every penalized coefficient is zero
            fit <- fit_oem(xx, y, family = family, penalty = "elastic.net", alpha = 0.5,
                           penalty.factor = pf, lambda = 10)
            b <- fit$beta[[1]][, 1]
            expect_equal(unname(b[1:2]), unname(ref), tolerance = 1e-6, info = paste(family, sparse))
            expect_true(all(b[-(1:2)] == 0))
        }
    }
})
