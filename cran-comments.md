## Update for 'oem' (2.1.0)

* Fixes incorrect results for models with at least as many variables as
  observations, for logistic regression with a sparse `x`, and for the ridge
  part of the elastic net penalties.
* The default `lambda` sequence now starts at the smallest value at which every
  penalized coefficient is zero.
* Accelerates the OEM iterations (FISTA by default) and raises the default
  `maxit` from 500 to 5000.
* Builds with Eigen 5 (`Eigen::Map<Eigen::SparseMatrix>`) and no longer calls
  `Rf_error()`.

See NEWS.md for details.

## Test environments

* local macOS (aarch64-apple-darwin20), R 4.5.1: R CMD check --as-cran
* (to fill in: R-hub and win-builder results)

## R CMD check results

0 errors | 0 warnings | 0 notes

## Reverse dependencies

* (to fill in: WpProj)
