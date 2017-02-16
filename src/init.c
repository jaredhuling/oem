
#include <oem_construction.h>
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
    {NULL, NULL, 0}
};

void attribute_visible R_init_oem(DllInfo *info)
    // relies on pkg/src/Makevars to mv data.table.so to datatable.so
{
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
