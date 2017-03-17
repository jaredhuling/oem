
#include "oem_sparse.h"

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
using Eigen::MappedSparseMatrix;
using Eigen::SparseMatrix;

typedef Map<VectorXd> MapVecd;
typedef Map<VectorXi> MapVeci;
typedef Map<Eigen::MatrixXd> MapMatd;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::MappedSparseMatrix<double> MSpMat;
typedef Eigen::SparseMatrix<double> SpMat;



RcppExport SEXP oem_fit_sparse(SEXP x_, 
                               SEXP y_, 
                               SEXP family_,
                               SEXP penalty_,
                               SEXP weights_,
                               SEXP groups_,
                               SEXP unique_groups_,
                               SEXP group_weights_,
                               SEXP lambda_,
                               SEXP nlambda_, 
                               SEXP lmin_ratio_,
                               SEXP alpha_,
                               SEXP gamma_,
                               SEXP tau_,
                               SEXP penalty_factor_,
                               SEXP standardize_, 
                               SEXP intercept_,
                               SEXP compute_loss_,
                               SEXP opts_)
{
    BEGIN_RCPP
    
    const MSpMat X(as<MSpMat>(x_));
    Rcpp::NumericVector yy(y_);
    
    const int n = X.rows();
    const int p = X.cols();
    
    const VectorXi groups(as<VectorXi>(groups_));
    const VectorXi unique_groups(as<VectorXi>(unique_groups_));
    
    VectorXd Y(n);
    
    // Copy data 
    std::copy(yy.begin(), yy.end(), Y.data());
    

    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    ArrayXd lambda(as<ArrayXd>(lambda_));
    VectorXd weights(as<VectorXd>(weights_));
    VectorXd group_weights(as<VectorXd>(group_weights_));
    int nlambda = lambda.size();
    
    
    List opts(opts_);
    const int maxit        = as<int>(opts["maxit"]);
    int ncores             = as<int>(opts["ncores"]);
    const double tol       = as<double>(opts["tol"]);
    const double alpha     = as<double>(alpha_);
    const double gamma     = as<double>(gamma_);
    const double tau       = as<double>(tau_);
    bool standardize       = as<bool>(standardize_);
    bool intercept         = as<bool>(intercept_);
    bool compute_loss      = as<bool>(compute_loss_);
    
    CharacterVector family(as<CharacterVector>(family_));
    std::vector<std::string> penalty(as< std::vector<std::string> >(penalty_));
    VectorXd penalty_factor(as<VectorXd>(penalty_factor_));
    
    // take all threads but one
    if (ncores < 1)
    {
        ncores = std::max(omp_get_num_threads() - 1, 1);
    }
    
    omp_set_num_threads(ncores);
    
    
    if (intercept)
    {
        // dont penalize the intercept
        VectorXd penalty_factor_tmp(p+1);
        
        penalty_factor_tmp << 0, penalty_factor;
        penalty_factor.swap(penalty_factor_tmp);
    }
    
    // initialize pointers 
    oemBase<Eigen::VectorXd> *solver = NULL; // solver doesn't point to anything yet
    
    // initialize class
    if (family(0) == "gaussian")
    {
        solver = new oemSparse(X, Y, weights, groups, unique_groups, 
                               group_weights, penalty_factor, 
                               intercept, standardize, ncores, tol);
        
    } else if (family(0) == "binomial")
    {
        throw std::invalid_argument("binomial not available for oem_fit_sparse, use oem_fit_logistic_sparse");
    }
    
    // compute initial pieces of oem
    solver->init_oem();
    
    double lmax = 0.0;
    lmax = solver->compute_lambda_zero(); // 
    
    if (nlambda < 1) 
    {
        double lmin = as<double>(lmin_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }
    
    
    MatrixXd beta(p + 1, nlambda);
    beta.setZero();
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
            
            niter[i] = solver->solve(maxit);
            VectorXd res = solver->get_beta();
            
            // store beta estimates
            if (intercept)
            {
                beta.block(0, i, p + 1, 1) = res;
            } else 
            {
                beta.block(1, i, p, 1) = res;
            }
            
            if (compute_loss)
            {
                // get associated loss
                loss(i) = solver->get_loss();
            }
            
            
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




