# oem 2.1.0

## Bug fixes

* The default `lambda` sequence now starts at the smallest value at which every
  penalized coefficient is zero. It accounts for `penalty.factor`,
  `group.weights`, `alpha`, unpenalized variables, and the intercept (#8).
  Previously the largest value was `max(abs(crossprod(x, y))) / n` for every
  penalty, so a path could begin with several fits with all coefficients zero,
  or with some coefficients already nonzero.
* Logistic regression with a sparse `x` returned wrong intercepts with
  `standardize = TRUE` and wrong coefficients with `standardize = FALSE`.
* Logistic regression with `standardize = FALSE` could use uninitialized column
  scales.
* Models with at least as many variables as observations now fit correctly for
  a sparse `x`, for logistic regression, and for `big.oem()` with an intercept
  or standardization. These cases previously returned wrong values or crashed R.
* `big.oem()` with a group penalty and an intercept left the last variable out
  of its group, so its coefficient was always zero.
* The ridge part of `elastic.net` and of the other `.net` penalties shrank
  unpenalized coefficients, including the intercept. For the elementwise
  penalties, the ridge part is now multiplied by `penalty.factor`, as in
  `glmnet`, and unpenalized coefficients are not shrunk by any penalty.
* C++ code no longer calls `Rf_error()`; errors are raised with Rcpp exception
  handling (#25).
* For logistic regression with `hessian.type = "full"`, the lower bound on the
  working weights was applied to only one observation.

## Changes in results

* With `standardize = TRUE`, every solver now scales each column by its standard
  deviation, computed with denominator n, as in `glmnet`. Fits with a sparse
  `x`, logistic regression, `big.oem()`, and `xval.oem()` previously used the
  uncentered root mean square with denominator n - 1, so their results with
  `standardize = TRUE` change.
* For Gaussian models, the ridge part of the `.net` penalties is divided by the
  standard deviation of `y` (its root mean square when `intercept = FALSE`), as
  in `glmnet`. This was already the case for a dense `x` with an intercept; it
  now holds for every solver except `oem.xtx()`, which does not see `y`.

## New features

* The OEM iterations are now accelerated by default. The new `accelerate`
  argument of `oem()`, `cv.oem()`, `xval.oem()`, `big.oem()`, and `oem.xtx()`
  selects FISTA with adaptive restarting (`"fista"`, the default), SQUAREM
  (`"squarem"`), or plain OEM (`"none"` or `FALSE`). For the nonconvex
  penalties, an accelerated step is accepted only if it does not increase the
  objective. On correlated designs, FISTA reduced computation time along the
  path by a factor of 4 to 10 and fitted the path to the requested tolerance
  in cases where plain OEM stopped at `maxit` before converging.

## Changes in defaults

* `accelerate = "fista"` is the default, so results change slightly for every
  model. Use `accelerate = "none"` to reproduce the plain OEM iterations of
  previous versions.
* The default `maxit` is now 5000 instead of 500. With 500 iterations, plain
  OEM often stopped before convergence for correlated variables, without a
  warning. Fits that converge sooner stop sooner, so the higher limit costs
  nothing for them.

## Performance

* Fits with a sparse `x`, logistic regression, `big.oem()`, and `xval.oem()`
  now center the columns implicitly and rescale the intercept column. When the
  columns have nonzero means, this reduces the number of OEM iterations along
  the path by about an order of magnitude.

## Other changes

* Sparse matrices are mapped with `Eigen::Map<Eigen::SparseMatrix>`, so the
  package builds with Eigen 5 (#26, from Dirk Eddelbuettel).
* Added a testthat suite, including comparisons with `glmnet`.
* `src/Makevars` assigned `PKG_CXXFLAGS` twice, so `-DNDEBUG` was dropped and
  Eigen's debugging assertions were compiled in. It is now set once.
