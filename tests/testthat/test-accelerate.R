accel_data <- function(n, p, rho = 0.7) {
    set.seed(123)
    z <- matrix(rnorm(n * p), n)
    x <- z
    for (j in 2:p) x[, j] <- rho * x[, j - 1] + sqrt(1 - rho^2) * z[, j]
    x[, 2] <- 3 * x[, 2] + 2
    eta <- drop(x[, 1:4] %*% c(0.8, -0.4, 0.5, 0.4))
    list(x = x, yg = eta + 1 + rnorm(n), yb = rbinom(n, 1, plogis(eta - 0.5)))
}

test_that("accelerated fits agree with tightly converged plain fits", {
    methods <- c("fista", "squarem")
    for (dims in list(c(300, 20), c(60, 120))) {
        dat <- accel_data(dims[1], dims[2])
        lam <- if (dims[2] > dims[1]) c(0.3, 0.15) else c(0.1, 0.02)
        groups <- rep(seq_len(dims[2] / 2), each = 2)
        for (family in c("gaussian", "binomial")) {
            y <- if (family == "gaussian") dat$yg else dat$yb
            for (sparse in c(FALSE, TRUE)) {
                xx <- if (sparse) as(dat$x, "CsparseMatrix") else dat$x
                for (penalty in c("lasso", "elastic.net", "grp.lasso")) {
                    args <- list(family = family, penalty = penalty, lambda = lam, alpha = 0.5,
                                 groups = groups, irls.tol = 1e-10)
                    ref <- suppressWarnings(do.call(oem, c(list(xx, y, tol = 1e-12, maxit = 1e5, accelerate = "none"), args)))
                    for (m in methods) {
                        fit <- suppressWarnings(do.call(oem, c(list(xx, y, tol = 1e-10, maxit = 1e5,
                                                                    accelerate = m), args)))
                        expect_equal(fit$beta[[1]], ref$beta[[1]], tolerance = 1e-5,
                                     info = paste(dims[1], dims[2], family, sparse, penalty, m))
                    }
                }
            }
        }
    }
})

test_that("acceleration works in big.oem, xval.oem, and oem.xtx", {
    dat <- accel_data(300, 20)
    lam <- c(0.1, 0.02)
    ref <- oem(dat$x, dat$yg, penalty = "lasso", lambda = lam, tol = 1e-12, maxit = 1e5,
               accelerate = "none")
    bx <- bigmemory::as.big.matrix(dat$x)
    for (m in c("fista", "squarem")) {
        big <- big.oem(bx, dat$yg, penalty = "lasso", lambda = lam, tol = 1e-10, maxit = 1e5, accelerate = m)
        expect_equal(big$beta[[1]], ref$beta[[1]], tolerance = 1e-6, info = m)
        xv <- xval.oem(dat$x, dat$yg, penalty = "lasso", lambda = lam, tol = 1e-10, maxit = 1e5,
                       nfolds = 3, accelerate = m)
        expect_equal(xv$beta[[1]], ref$beta[[1]], tolerance = 1e-6, info = m)
        xtx <- crossprod(dat$x) / nrow(dat$x)
        xty <- drop(crossprod(dat$x, dat$yg)) / nrow(dat$x)
        plain <- oem.xtx(xtx, xty, penalty = "lasso", lambda = lam, tol = 1e-12, maxit = 1e5,
               accelerate = "none")
        acc <- oem.xtx(xtx, xty, penalty = "lasso", lambda = lam, tol = 1e-10, maxit = 1e5, accelerate = m)
        expect_equal(acc$beta[[1]], plain$beta[[1]], tolerance = 1e-6, info = m)
    }
    # big.oem with more variables than observations
    dat <- accel_data(60, 120)
    lam <- c(0.3, 0.15)
    ref <- suppressWarnings(oem(dat$x, dat$yg, penalty = "lasso", lambda = lam, tol = 1e-12,
                                maxit = 1e5, accelerate = "none"))
    bx <- bigmemory::as.big.matrix(dat$x)
    for (m in c("fista", "squarem")) {
        big <- suppressWarnings(big.oem(bx, dat$yg, penalty = "lasso", lambda = lam, tol = 1e-10,
                                        maxit = 1e5, accelerate = m))
        expect_equal(big$beta[[1]], ref$beta[[1]], tolerance = 1e-5, info = paste("p > n", m))
    }
})

test_that("accelerated coefficients are exactly sparse", {
    dat <- accel_data(300, 20)
    plain <- oem(dat$x, dat$yg, penalty = "lasso", nlambda = 20, tol = 1e-12, maxit = 1e5,
                 accelerate = "none")
    for (m in c("fista", "squarem")) {
        fit <- oem(dat$x, dat$yg, penalty = "lasso", nlambda = 20, tol = 1e-10, maxit = 1e5, accelerate = m)
        expect_identical(fit$beta[[1]] != 0, plain$beta[[1]] != 0, info = m)
    }
})

test_that("the accelerate argument accepts logical values and rejects unknown methods", {
    dat <- accel_data(100, 10)
    lam <- c(0.1, 0.02)
    fit_true <- oem(dat$x, dat$yg, penalty = "lasso", lambda = lam, accelerate = TRUE)
    fit_fista <- oem(dat$x, dat$yg, penalty = "lasso", lambda = lam, accelerate = "fista")
    expect_identical(fit_true$beta, fit_fista$beta)
    fit_false <- oem(dat$x, dat$yg, penalty = "lasso", lambda = lam, accelerate = FALSE, tol = 1e-12, maxit = 1e5)
    expect_equal(fit_false$beta[[1]], fit_fista$beta[[1]], tolerance = 1e-5)
    expect_error(oem(dat$x, dat$yg, penalty = "lasso", lambda = lam, accelerate = "anderson"))
})
