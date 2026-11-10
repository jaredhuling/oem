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
* win-builder: R-devel (2026-11-07 r90850 ucrt) and R-release (4.6.2 ucrt)
* R-hub (R-devel): linux, macos-arm64, windows, m1-san, clang-asan,
  clang-ubsan, valgrind, gcc16, clang23, atlas, mkl, nosuggests

## R CMD check results

0 errors | 0 warnings | 0 notes

The sanitizer builds (m1-san, clang-asan, clang-ubsan) reported no runtime
errors, and valgrind reported 0 errors and no leaks. On the R-hub nosuggests
platform, the only problem was that the vignette could not be rebuilt because
rmarkdown, the vignette builder, is in Suggests and was not installed; the
tests and examples passed there.

## Reverse dependencies

We checked the one reverse dependency on CRAN, WpProj 0.3.1, with this version:
0 errors, 0 warnings, 0 notes, and all of its tests pass.
