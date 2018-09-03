
## Update for 'oem'

* Vignette update
* The clang-UBSAN gcc-UBSAN warnings appear to be not due to any issue with bigmemory, but with the tests not allowing inheritance of classes. See here https://github.com/kaneplusplus/bigmemory/issues/73#issuecomment-362330232

## Test environments

* local Windows 7 x64 install, (R 3.4.2, r74446 R-devel)
* Ubuntu 14.04.5 LTS (on travis-ci), (R 3.4.4, R-patch)

## R CMD check results

Status: OK



R CMD check results
0 errors | 0 warnings | 0 notes

R CMD check succeeded