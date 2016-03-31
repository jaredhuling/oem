
#include "oem_dense.h"

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
    
    const MapMatd X(as<MapMatd>(x_));
    const MapVecd Y(as<MapVecd>(y_));
    
    const int n = X.rows();
    const int p = X.cols();
    

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
            
            //VectorXd v(n);
            //v.fill(1);
            //MatrixXd X_tmp(n, p+1);
            
            //X_tmp << v, X;
            //X.swap(X_tmp);
            
            //X_tmp.resize(0,0);
        }
    } else 
    {
        if (intercept & !standardize) 
        {
            VectorXd penalty_factor_tmp(p+1);
            
            penalty_factor_tmp << 0, penalty_factor;
            penalty_factor.swap(penalty_factor_tmp);
        }
    }
    
    //DataStd<double> datstd(n, p + add, standardize, intercept);
    //datstd.standardize(X, Y);
    
    std::cout << "before solver" << std::endl;
    
    // initialize pointers 
    oemBase<Eigen::VectorXd> *solver = NULL; // obj doesn't point to anything yet
    
    
    // initialize classes

    if (family(0) == "gaussian")
    {
        solver = new oemDense(X, Y, penalty_factor, alpha, gamma, intercept, standardize, tol);
    } else if (family(0) == "binomial")
    {
        //solver = new oem(X, Y, penalty_factor, irls_tol, irls_maxit, eps_abs, eps_rel);
    }
    
    std::cout << "after new solver " << std::endl;
    
    if (nlambda < 1) {
        
        double lmax = 0.0;
        
        lmax = solver->compute_lambda_zero() / n; // * datstd.get_scaleY();
        double lmin = as<double>(lmin_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
        std::cout << "lambda" << lmax << std::endl;
    }
    
    
    MatrixXd beta(p + 1, nlambda);
    List beta_list(penalty.size());
    List iter_list(penalty.size());
    
    IntegerVector niter(nlambda);
    int nlambda_store = nlambda;
    double ilambda = 0.0;

    for (unsigned int pp = 0; pp < penalty.size(); pp++)
    {
        if (penalty[pp] == "ols")
        {
            nlambda = 1L;
        } 
        
        for(int i = 0; i < nlambda; i++)
        {
            ilambda = lambda[i] * n; //     / datstd.get_scaleY();
            if(i == 0)
                solver->init(ilambda, penalty[pp]);
            else
                solver->init_warm(ilambda);
            
            std::cout << "initialized" << std::endl;
            
            niter[i] = solver->solve(maxit);
            VectorXd res = solver->get_beta();
            
            double beta0 = 0.0;
            
            // if the design matrix includes the intercept
            // then don't back into the intercept with
            // datastd and include it to beta directly.
            /*
            if (fullbetamat)
            {
                beta.block(0, i, p+1, 1) = res;
                //datstd.recover(beta0, res);
            } else 
            {
                //datstd.recover(beta0, res);
                //beta(0,i) = beta0;
                //beta.block(1, i, p, 1) = res;
            }
            */
            beta.block(0, i, p+1, 1) = res;
            
        } //end loop over lambda values
        
        if (penalty[pp] == "ols")
        {
            // reset to old nlambda
            nlambda = nlambda_store;
            beta_list(pp) = beta.col(0);
        } else 
        {
            beta_list(pp) = beta;
        }
        
        
        iter_list(pp) = niter;
        
    } // end loop over penalties
    
    double d = solver->get_d();

    delete solver;

    return List::create(Named("beta")   = beta_list,
                        Named("lambda") = lambda,
                        Named("niter")  = iter_list,
                        Named("d")      = d);
    END_RCPP
}




