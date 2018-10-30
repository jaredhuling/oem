
## Update for 'oem'

* Fixes error for datasets with p < 4
* Fixes makevars problem pointed out by Prof Ripley
* The clang-UBSAN gcc-UBSAN warnings appear to be not due to any issue with bigmemory, but with the tests not allowing inheritance of classes. See here https://github.com/kaneplusplus/bigmemory/issues/73#issuecomment-362330232

## Test environments

* Windows x64 install, (3.5.1, devel 2018-10-27 r75507, 3.5.1 Patched 2018-10-27 r75507)
* Ubuntu 14.04.5 LTS (on travis-ci), (R 3.5.1, R-patch)

## R CMD check results


-- R CMD check results ------------------------------------------ oem 2.0.9 ----
Duration: 12m 3.6s

0 errors v | 0 warnings v | 0 notes v

R CMD check succeeded