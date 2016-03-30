#ifndef _oem2_RUTILS_H
#define _oem2_RUTILS_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/SVD>
#include <vector> 
#include <functional> 
#include <algorithm> 
#include <iostream>
#include <cmath>

using namespace Rcpp;
using namespace RcppEigen;


RcppExport SEXP crossprodcpp(SEXP);

RcppExport SEXP xpwx(SEXP, SEXP);
                               
RcppExport SEXP subcpp(SEXP, SEXP);

RcppExport SEXP addcpp(SEXP, SEXP);

RcppExport SEXP subSparsecpp(SEXP, SEXP);

RcppExport SEXP addSparsecpp(SEXP, SEXP);

RcppExport SEXP LanczosBidiag(SEXP, SEXP, SEXP);

RcppExport SEXP LanczosBidiagSparse(SEXP, SEXP, SEXP);

RcppExport SEXP BidiagPoly(SEXP, SEXP, SEXP);

#endif