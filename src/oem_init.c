#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/*
SEXP oemfit(SEXP,
            SEXP,
            SEXP,
            SEXP,
            SEXP,
            SEXP,
            SEXP,
            SEXP,
            SEXP,
            SEXP);

SEXP oem_fit_big(SEXP, 
                 SEXP, 
                 SEXP,
                 SEXP,
                 SEXP,
                 SEXP,
                 SEXP,
                 SEXP,
                 SEXP,
                 SEXP, 
                 SEXP,
                 SEXP,
                 SEXP,
                 SEXP,
                 SEXP, 
                 SEXP,
                 SEXP,
                 SEXP);

SEXP oem_fit_dense(SEXP, 
                   SEXP, 
                   SEXP,
                   SEXP,
                   SEXP,
                   SEXP,
                   SEXP,
                   SEXP,
                   SEXP,
                   SEXP, 
                   SEXP,
                   SEXP,
                   SEXP,
                   SEXP,
                   SEXP, 
                   SEXP,
                   SEXP,
                   SEXP);

SEXP oem_fit_logistic_dense(SEXP, 
                            SEXP, 
                            SEXP,
                            SEXP,
                            SEXP,
                            SEXP,
                            SEXP,
                            SEXP,
                            SEXP,
                            SEXP, 
                            SEXP,
                            SEXP,
                            SEXP,
                            SEXP,
                            SEXP, 
                            SEXP,
                            SEXP,
                            SEXP);

SEXP oem_fit_logistic_sparse(SEXP, 
                             SEXP, 
                             SEXP,
                             SEXP,
                             SEXP,
                             SEXP,
                             SEXP,
                             SEXP,
                             SEXP,
                             SEXP, 
                             SEXP,
                             SEXP,
                             SEXP,
                             SEXP,
                             SEXP, 
                             SEXP,
                             SEXP,
                             SEXP);

SEXP oem_fit_sparse(SEXP, 
                    SEXP, 
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP, 
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP, 
                    SEXP,
                    SEXP,
                    SEXP);

SEXP oem_xtx(SEXP, 
             SEXP, 
             SEXP,
             SEXP,
             SEXP,
             SEXP,
             SEXP,
             SEXP,
             SEXP, 
             SEXP,
             SEXP,
             SEXP,
             SEXP,
             SEXP,
             SEXP);

SEXP oem_xval_dense(SEXP, 
                    SEXP, 
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP, 
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP, 
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP,
                    SEXP);


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
*/


void R_init_oem(DllInfo *info)
{
    R_registerRoutines(info, NULL, NULL, NULL, NULL);
    R_useDynamicSymbols(info, TRUE);
}
