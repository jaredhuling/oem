#ifndef _oem_OEM_CALLS_H
#define _oem_OEM_CALLS_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector> 
#include <functional> 
#include <algorithm> 
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <float.h>

using namespace Rcpp;


RcppExport SEXP oem_fit_dense_tall(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                                   SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);


#endif
