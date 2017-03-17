
#include "oem_xtx.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
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
typedef Map<VectorXi> MapVeci;
typedef Map<Eigen::MatrixXd> MapMatd;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;


RcppExport SEXP oem_xtx(SEXP xtx_, 
                        SEXP xty_, 
                        SEXP family_,
                        SEXP penalty_,
                        SEXP groups_,
                        SEXP unique_groups_,
                        SEXP group_weights_,
                        SEXP lambda_,
                        SEXP nlambda_, 
                        SEXP lmin_ratio_,
                        SEXP alpha_,
                        SEXP gamma_,
                        SEXP tau_,
                        SEXP scale_factor_,
                        SEXP penalty_factor_,
                        SEXP opts_)
{
    BEGIN_RCPP
    
    const MapMatd xtx(as<MapMatd >(xtx_));
    const MapVecd xty(as<MapVecd >(xty_));
    
    const int p = xtx.cols();
    
    const VectorXd scale_factor(as<VectorXd>(scale_factor_));
    const VectorXi groups(as<VectorXi>(groups_));
    const VectorXi unique_groups(as<VectorXi>(unique_groups_));
    

    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    ArrayXd lambda(as<ArrayXd>(lambda_));
    VectorXd group_weights(as<VectorXd>(group_weights_));
    int nlambda = lambda.size();
    
    
    List opts(opts_);
    const int maxit        = as<int>(opts["maxit"]);
    const double tol       = as<double>(opts["tol"]);
    const double alpha     = as<double>(alpha_);
    const double gamma     = as<double>(gamma_);
    const double tau       = as<double>(tau_);
    
    CharacterVector family(as<CharacterVector>(family_));
    std::vector<std::string> penalty(as< std::vector<std::string> >(penalty_));
    VectorXd penalty_factor(as<VectorXd>(penalty_factor_));
    

    // initialize pointers 
    oemBase<Eigen::VectorXd> *solver = NULL; // solver doesn't point to anything yet
    
    
    // initialize classes
    
    if (family(0) == "gaussian")
    {
        solver = new oemXTX(xtx, xty, groups, unique_groups, 
                            group_weights, penalty_factor, 
                            scale_factor, tol);
    } else if (family(0) == "binomial")
    {
        throw std::invalid_argument("binomial not available for oem_fit_dense, use oem_fit_logistic_dense");
        //solver = new oem(X, Y, penalty_factor, irls_tol, irls_maxit, eps_abs, eps_rel);
    }
    
    // initialize oem
    solver->init_oem();
    
    double lmax = 0.0;
    lmax = solver->compute_lambda_zero(); // 
    
    
    if (nlambda < 1) {
        
        double lmin = as<double>(lmin_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
        
    }
    
    
    MatrixXd beta(p, nlambda);
    List beta_list(penalty.size());
    List iter_list(penalty.size());
    List loss_list(penalty.size());
    
    IntegerVector niter(nlambda);
    int nlambda_store = nlambda;
    double ilambda = 0.0;
    
    for (unsigned int pp = 0; pp < penalty.size(); pp++)
    {
        if (penalty[pp] == "ols")
        {
            nlambda = 1L;
        }
        
        VectorXd loss(nlambda);
        loss.fill(1e99);
        
        for(int i = 0; i < nlambda; i++)
        {
            if (i % 3 == 0)
            {
                Rcpp::checkUserInterrupt();
            }
            ilambda = lambda[i]; // * n; //     
            if(i == 0)
                solver->init(ilambda, penalty[pp],
                             alpha, gamma, tau);
            else
                solver->init_warm(ilambda);
            
            niter[i]     = solver->solve(maxit);
            
            VectorXd res = solver->get_beta();
            
            beta.col(i)  = res;
            
            
        } //end loop over lambda values
        
        
        
        if (penalty[pp] == "ols")
        {
            // reset to old nlambda
            nlambda = nlambda_store;
            beta_list(pp) = beta.col(0);
            iter_list(pp) = niter(0);
            loss_list(pp) = loss(0);
        } else 
        {
            beta_list(pp) = beta;
            iter_list(pp) = niter;
            loss_list(pp) = loss;
        }
        
        
    } // end loop over penalties
    
    double d = solver->get_d();
    
    delete solver;
    
    return List::create(Named("beta")   = beta_list,
                        Named("lambda") = lambda,
                        Named("niter")  = iter_list,
                        Named("loss")   = loss_list,
                        Named("d")      = d);
    END_RCPP
}

