
#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>


SEXP oemfit();
SEXP oem_fit_big();
SEXP oem_fit_dense();
SEXP oem_fit_logistic_dense();
SEXP oem_fit_logistic_sparse();
SEXP oem_fit_sparse();
SEXP oem_xtx();
SEXP oem_xval_dense();
SEXP crossprodcpp();
SEXP tcrossprodcpp_scaled();
SEXP crossprodcpp_scaled();
SEXP largestEig();
SEXP xpwx();
SEXP xxt();
SEXP subcpp();
SEXP addcpp();
SEXP subSparsecpp();
SEXP addSparsecpp();
SEXP LanczosBidiag();
SEXP LanczosBidiagSparse();
SEXP BidiagPoly();

static const
R_CallMethodDef callMethods[] = {
    {"Csetattrib", (DL_FUNC) &oemfit, -1},
    {"Cbmerge", (DL_FUNC) &oem_fit_big, -1},
    {"Cbmerge", (DL_FUNC) &oem_fit_dense, -1},
    {"Cbmerge", (DL_FUNC) &oem_fit_logistic_dense, -1},
    {"Cbmerge", (DL_FUNC) &oem_fit_logistic_sparse, -1},
    {"Cbmerge", (DL_FUNC) &oem_fit_sparse, -1},
    {"Cbmerge", (DL_FUNC) &oem_xtx, -1},
    {"Cbmerge", (DL_FUNC) &oem_xval_dense, -1},
    {"Cbmerge", (DL_FUNC) &crossprodcpp, -1},
    {"Cbmerge", (DL_FUNC) &tcrossprodcpp_scaled, -1},
    {"Cbmerge", (DL_FUNC) &crossprodcpp_scaled, -1},
    {"Cbmerge", (DL_FUNC) &largestEig, -1},
    {"Cbmerge", (DL_FUNC) &xpwx, -1},
    {"Cbmerge", (DL_FUNC) &xxt, -1},
    {"Cbmerge", (DL_FUNC) &subcpp, -1},
    {"Cbmerge", (DL_FUNC) &addcpp, -1},
    {"Cbmerge", (DL_FUNC) &subSparsecpp, -1},
    {"Cbmerge", (DL_FUNC) &addSparsecpp, -1},
    {"Cbmerge", (DL_FUNC) &LanczosBidiag, -1},
    {"Cbmerge", (DL_FUNC) &LanczosBidiagSparse, -1},
    {"Cbmerge", (DL_FUNC) &BidiagPoly, -1},
    {NULL, NULL, 0}
};

void attribute_visible R_init_oem(DllInfo *info)
    // relies on pkg/src/Makevars to mv data.table.so to datatable.so
{
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
