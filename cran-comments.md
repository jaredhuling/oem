
## Update for 'oem'

* Fixes error for datasets with p < 4
* Fixes makevars problem pointed out by Prof Ripley
* The clang-UBSAN gcc-UBSAN warnings appear to be not due to any issue with bigmemory, but with the tests not allowing inheritance of classes. See here https://github.com/kaneplusplus/bigmemory/issues/73#issuecomment-362330232

## Test environments

* Windows x64 install, (3.5.1, 2018-09-02 r75226, 3.5.1 Patched 2018-08-31 r75226)
* Ubuntu 14.04.5 LTS (on travis-ci), (R 3.5.0, R-patch)

## R CMD check results

Status: OK



R CMD check results
0 errors | 0 warnings | 0 notes

R CMD check succeeded