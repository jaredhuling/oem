
#include "oem_dense.h"
#include "DataStd.h"

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


RcppExport SEXP oem_fit_dense(SEXP x_, 
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
    
    Rcpp::NumericMatrix xx(x_);
    Rcpp::NumericVector yy(y_);
    
    const int n = xx.rows();
    const int p = xx.cols();
    
    const VectorXi groups(as<VectorXi>(groups_));
    const VectorXi unique_groups(as<VectorXi>(unique_groups_));
    
    MatrixXd X(n, p);
    VectorXd Y(n);
    
    
    // Copy data 
    std::copy(xx.begin(), xx.end(), X.data());
    std::copy(yy.begin(), yy.end(), Y.data());
    

    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    //ArrayXd lambda(as<ArrayXd>(lambda_)); // old lambda code
    VectorXd weights(as<VectorXd>(weights_));
    VectorXd group_weights(as<VectorXd>(group_weights_));
    
    
    std::vector<VectorXd> lambda(as< std::vector<VectorXd> >(lambda_));
    
    VectorXd lambda_tmp;
    lambda_tmp = lambda[0];
    
    int nl = as<int>(nlambda_);
    VectorXd lambda_base(nl);
    
    
    int nlambda = lambda_tmp.size();
    
    
    List opts(opts_);
    const int maxit        = as<int>(opts["maxit"]);
    int ncores             = as<int>(opts["ncores"]);
    const double tol       = as<double>(opts["tol"]);
    const double alpha     = as<double>(alpha_);
    const double gamma     = as<double>(gamma_);
    const double tau       = as<double>(tau_);
    bool standardize       = as<bool>(standardize_);
    bool intercept         = as<bool>(intercept_);
    bool intercept_bin     = intercept;
    bool compute_loss      = as<bool>(compute_loss_);
    const bool accelerate  = as<double>(opts["accelerate"]);
    
    
    CharacterVector family(as<CharacterVector>(family_));
    std::vector<std::string> penalty(as< std::vector<std::string> >(penalty_));
    VectorXd penalty_factor(as<VectorXd>(penalty_factor_));
    
    
    // take all threads but one
    if (ncores < 1)
    {
        ncores = std::max(omp_get_num_threads() - 1, 1);
    }
    
    omp_set_num_threads(ncores);
    
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    
    // don't standardize if not linear model. 
    // fit intercept the dumb way if it is wanted
    // bool fullbetamat = false;
    int add = 0;
    if (family(0) != "gaussian")
    {
        standardize = false;
        intercept = false;
        
        if (intercept_bin)
        {
            // fullbetamat = true;
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
    }
    
    DataStd<double> datstd(n, p + add, standardize, intercept);
    datstd.standardize(X, Y, weights);
    
    
    // initialize pointers 
    oemBase<Eigen::VectorXd> *solver = NULL; // solver doesn't point to anything yet
    
    
    // initialize classes
    if (family(0) == "gaussian")
    {
        solver = new oemDense(X, Y, weights, groups, unique_groups, 
                              group_weights, penalty_factor, 
                              intercept, standardize, 
                              ncores, tol, accelerate);
    } else if (family(0) == "binomial")
    {
        throw std::invalid_argument("binomial not available for oem_fit_dense, use oem_fit_logistic_dense");
        //solver = new oem(X, Y, penalty_factor, irls_tol, irls_maxit, eps_abs, eps_rel);
    }
    
    
    solver->init_oem();
    
    double lmax = 0.0;
    lmax = solver->compute_lambda_zero() * datstd.get_scaleY(); // 
    
    
    bool provided_lambda = false;
    if (nlambda < 1) 
    {
        double lmin = as<double>(lmin_ratio_) * lmax;
        
        lambda_base.setLinSpaced(nl, std::log(lmax), std::log(lmin));
        lambda_base = lambda_base.array().exp();
        nlambda = lambda_base.size();
        
        lambda_tmp.resize(nlambda);
    } else
    {
        provided_lambda = true;
    }
    
    
    MatrixXd beta(p + 1, nlambda);
    List beta_list(penalty.size());
    List iter_list(penalty.size());
    List loss_list(penalty.size());
    
    IntegerVector niter(nlambda);
    int nlambda_store = nlambda;
    double ilambda = 0.0;
    
    std::string elasticnettxt(".net");
    
    for (unsigned int pp = 0; pp < penalty.size(); pp++)
    {
        if (penalty[pp] == "ols")
        {
            nlambda = 1L;
        }
        
        bool is_net_pen = penalty[pp].find(elasticnettxt) != std::string::npos;
        
        if (provided_lambda)
        {
            lambda_tmp = lambda[pp];
        } else 
        {
            if (is_net_pen)
            {
                lambda_tmp = (lambda_base.array() / alpha).matrix(); // * n; // 
            } else
            {
                lambda_tmp = lambda_base; // * n; // 
            }
        }
        
        VectorXd loss(nlambda);
        loss.fill(1e99);
        
        for(int i = 0; i < nlambda; i++)
        {
            
            if (i % 3 == 0)
            {
                Rcpp::checkUserInterrupt();
            }
            
            
            ilambda = lambda_tmp(i) / datstd.get_scaleY();
                
            if(i == 0)
                solver->init(ilambda, penalty[pp], alpha, gamma, tau);
            else
                solver->init_warm(ilambda);
            
            niter[i] = solver->solve(maxit);
            VectorXd res = solver->get_beta();
            
            double beta0 = 0.0;
            datstd.recover(beta0, res);
            beta(0,i) = beta0;
            beta.block(1, i, p, 1) = res;
            
            if (compute_loss)
            {
                // get associated loss
                loss(i) = solver->get_loss();
            }
            
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
            
        } //end loop over lambda values
        
        lambda[pp] = lambda_tmp;
        
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




