
#include "oem_dense_tall.h"
#include "DataStd.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXf;
using Eigen::ArrayXd;
using Eigen::ArrayXXf;
using Eigen::Map;

using Rcpp::wrap;
using Rcpp::as;
using Rcpp::List;
using Rcpp::Named;
using Rcpp::IntegerVector;
using Rcpp::CharacterVector;

typedef Map<VectorXd> MapVecd;
typedef Map<Eigen::MatrixXd> MapMatd;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;


RcppExport SEXP oem_fit_dense_tall(SEXP x_, 
                                   SEXP y_, 
                                   SEXP family_,
                                   SEXP penalty_,
                                   SEXP lambda_,
                                   SEXP nlambda_, 
                                   SEXP lmin_ratio_,
                                   SEXP alpha_,
                                   SEXP gamma_,
                                   SEXP penalty_factor_,
                                   SEXP standardize_, 
                                   SEXP intercept_,
                                   SEXP opts_)
{
    BEGIN_RCPP

    //Rcpp::NumericMatrix xx(x_);
    //Rcpp::NumericVector yy(y_);
    
    
    Rcpp::NumericMatrix xx(x_);
    Rcpp::NumericVector yy(y_);
    
    const int n = xx.rows();
    const int p = xx.cols();
    
    MatrixXd X(n, p);
    VectorXd Y(n);
    
    // Copy data 
    std::copy(xx.begin(), xx.end(), X.data());
    std::copy(yy.begin(), yy.end(), Y.data());
    

    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    ArrayXd lambda(as<ArrayXd>(lambda_));
    int nlambda = lambda.size();
    

    List opts(opts_);
    const int maxit        = as<int>(opts["maxit"]);
    const int irls_maxit   = as<int>(opts["irls_maxit"]);
    const double irls_tol  = as<double>(opts["irls_tol"]);
    const double tol       = as<double>(opts["tol"]);
    const double alpha     = as<double>(alpha_);
    const double gamma     = as<double>(gamma_);
    bool standardize       = as<bool>(standardize_);
    bool intercept         = as<bool>(intercept_);
    bool intercept_bin     = intercept;
    
    CharacterVector family(as<CharacterVector>(family_));
    std::vector<std::string> penalty(as< std::vector<std::string> >(penalty_));
    VectorXd penalty_factor(as<VectorXd>(penalty_factor_));
    
    // don't standardize if not linear model. 
    // fit intercept the dumb way if it is wanted
    bool fullbetamat = false;
    int add = 0;
    if (family(0) != "gaussian")
    {
        standardize = false;
        intercept = false;
        
        if (intercept_bin)
        {
            fullbetamat = true;
            add = 1;
            // dont penalize the intercept
            VectorXd penalty_factor_tmp(p+1);
            
            penalty_factor_tmp << 0, penalty_factor;
            penalty_factor.swap(penalty_factor_tmp);
            
            VectorXd v(n);
            v.fill(1);
            MatrixXd X_tmp(n, p+1);
            
            X_tmp << v, X;
            X.swap(X_tmp);
            
            X_tmp.resize(0,0);
        }
    }
    
    DataStd<double> datstd(n, p + add, standardize, intercept);
    datstd.standardize(X, Y);
    
    // initialize pointers 
    oemBase<Eigen::VectorXd> *solver = NULL; // obj doesn't point to anything yet
    
    
    // initialize classes
    if(n > 2 * p)
    {
        if (family(0) == "gaussian")
        {
            solver = new oemDenseTall(X, Y, penalty_factor, alpha, gamma, tol);
        } else if (family(0) == "binomial")
        {
            //solver = new oem(X, Y, penalty_factor, irls_tol, irls_maxit, eps_abs, eps_rel);
        }
    } else
    {
        if (family(0) == "gaussian")
        {
            //solver = new oemWide(datX, datY, penalty_factor, eps_abs, eps_rel);
        } else if (family(0) == "binomial")
        {
            //solver_wide = new ADMMLassoLogisticWide(datX, datY, penalty_factor, irls_tol, irls_maxit, eps_abs, eps_rel);
            //solver = new oemWide(datX, datY, penalty_factor, eps_abs, eps_rel);
            std::cout << "Warning: binomial not implemented for wide case yet \n"  << std::endl;
        }
    }

    
    if (nlambda < 1) {
        
        double lmax = 0.0;
        
        lmax = solver->get_lambda_zero() / n * datstd.get_scaleY();
        double lmin = as<double>(lmin_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }

    List beta_list(penalty.size());
    List iter_list(penalty.size());
    
    MatrixXd beta(p + 1, nlambda);

    IntegerVector niter(nlambda);
    double ilambda = 0.0;

    for (unsigned int p = 0; p < penalty.size(); p++)
    {
        for(int i = 0; i < nlambda; i++)
        {
            ilambda = lambda[i] * n / datstd.get_scaleY();
            if(i == 0)
                solver->init(ilambda, penalty[p]);
            else
                solver->init_warm(ilambda);
    
            niter[i] = solver->solve(maxit);
            VectorXd res = solver->get_beta();
            
            double beta0 = 0.0;
            
            // if the design matrix includes the intercept
            // then don't back into the intercept with
            // datastd and include it to beta directly.
            if (fullbetamat)
            {
                beta.block(0, i, p+1, 1) = res;
                datstd.recover(beta0, res);
            } else 
            {
                datstd.recover(beta0, res);
                beta(0,i) = beta0;
                beta.block(1, i, p, 1) = res;
            }
            
        } //end loop over lambda values
        
        beta_list(p) = beta;
        iter_list(p) = niter;
        
    } // end loop over penalties

    delete solver;

    return List::create(Named("beta")   = beta_list,
                        Named("lambda") = lambda,
                        Named("niter")  = iter_list);
    END_RCPP
}




