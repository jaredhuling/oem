
#include "oem_xval_dense.h"

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
typedef Eigen::Map<const MatrixXd> MapMat;
typedef Map<Eigen::MatrixXd> MapMatd;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRXd;
typedef Map<MatrixRXd> MapMatRd;

RcppExport SEXP oem_xval_dense(SEXP x_, 
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
                               SEXP nfolds_,
                               SEXP foldid_,
                               SEXP compute_loss_,
                               SEXP type_measure_,
                               SEXP opts_)
{
    BEGIN_RCPP
    
    
    Rcpp::NumericMatrix xx(x_);
    Rcpp::NumericVector yy(y_);
    
    //const Eigen::Map<MatrixXd> xx(as<Map<MatrixXd> >(x_));
    
    
    const int n = xx.rows();
    const int p = xx.cols();
    
    const VectorXi foldid(as<VectorXi>(foldid_));
    const VectorXi groups(as<VectorXi>(groups_));
    const VectorXi unique_groups(as<VectorXi>(unique_groups_));
    
    
    VectorXd Y(n);
    
    // Copy data 
    const MapMatd XX(as<MapMatd >(xx));
    const MatrixRXd X(XX);
    
    // std::copy(xx.begin(), xx.end(), XX.data());
    std::copy(yy.begin(), yy.end(), Y.data());
    
    
    // X = XX;
    // XX.resize(0,0);

    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    ArrayXd lambda(as<ArrayXd>(lambda_));
    VectorXd weights(as<VectorXd>(weights_));
    VectorXd group_weights(as<VectorXd>(group_weights_));
    int nlambda = lambda.size();
    
    
    List opts(opts_);
    const int nfolds       = as<int>(nfolds_);
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
        std::vector<std::string> type_measure(as< std::vector<std::string> >(type_measure_));
    VectorXd penalty_factor(as<VectorXd>(penalty_factor_));
    
    
    // take all threads but one
    if (ncores < 1)
    {
        ncores = std::max(omp_get_num_threads() - 1, 1);
    }
    
    omp_set_num_threads(ncores);
    
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    
    
    // don't standardize.
    // fit intercept the dumb way if it is wanted
    // bool fullbetamat = false;
    //int add = 0;
    
    if (intercept)
    {
        // dont penalize the intercept
        VectorXd penalty_factor_tmp(p+1);
        
        penalty_factor_tmp << 0, penalty_factor;
        penalty_factor.swap(penalty_factor_tmp);
    }
    
    // initialize pointers 
    oemBase<Eigen::VectorXd> *solver = NULL; // solver doesn't point to anything yet
    
    
    // initialize classes
    if (family(0) == "gaussian")
    {
        solver = new oemXvalDense(X, Y, weights, nfolds, foldid,
                                  groups, unique_groups, 
                                  group_weights, penalty_factor, 
                                  intercept, standardize, tol);
    } else if (family(0) == "binomial")
    {
        throw std::invalid_argument("binomial not available for oem_xval_dense, use oem_xval_logistic_dense");
        //solver = new oem(X, Y, penalty_factor, irls_tol, irls_maxit, eps_abs, eps_rel);
    }
    
    
    // only compute X'X parts once
    solver->init_xtx(intercept);
    
    // get eigenvalue
    double d = solver->get_d();
    
    
    double lmax = 0.0;
    lmax = solver->compute_lambda_zero(); // 
    
    
    if (nlambda < 1) {
        
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
    //std::vector<Eigen::MatrixXd> out_of_fold_predictions_list(penalty.size());
    std::vector<Eigen::VectorXd> xval_mean(penalty.size());
    std::vector<Eigen::VectorXd> xval_sd(penalty.size());
    std::vector<int> nlam_list(penalty.size());
    
    // vector of vectors of MatrixXd's. confusing
    std::vector<std::vector<Eigen::MatrixXd> > beta_folds(penalty.size(), std::vector<Eigen::MatrixXd>(nfolds));
    //List beta_folds(nfolds);
    
    IntegerVector niter(nlambda);
    int nlambda_store = nlambda;
    double ilambda = 0.0;
    
    for (int ff = 0; ff < nfolds + 1; ++ff)
    {
        // ff == 0 will fit the models
        // on the entire dataset
        // ff = 1, ..., nfolds will fit the models
        // for each cross validation fold
        
        if (ff > 0)
        {
            // update X'X and X'Y on this fold's 
            // subset of data
            solver->update_xtx(ff);
            
        }
        
        for (unsigned int pp = 0; pp < penalty.size(); pp++)
        {
            if (penalty[pp] == "ols")
            {
                nlambda = 1L;
            }
            
            
            if (ff == 0)
            {
                //out_of_fold_predictions_list[pp] = MatrixXd(n, nlambda);
                nlam_list[pp] = nlambda;
            }
            
            VectorXd loss(nlambda);
            loss.fill(1e99);
            
            for(int i = 0; i < nlambda; i++)
            {
                
                if (i % 10 == 0)
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
                
                
                //double beta0 = 0.0;
                //beta(0,i) = beta0;
                //beta.block(1, i, p, 1) = res;
                if (intercept)
                {
                    beta.block(0, i, p + 1, 1) = res;
                } else 
                {
                    beta.block(1, i, p, 1) = res;
                }
                
                // only compute loss if asked for 
                // and not for any cross validation folds
                if (compute_loss && ff == 0)
                {
                    // get associated loss
                    loss(i) = solver->get_loss();
                }
                
                
            } //end loop over lambda values
            
            if (ff == 0)
            {
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
            } else 
            {
                if (penalty[pp] == "ols")
                {
                    // reset to old nlambda
                    nlambda = nlambda_store;
                    //beta_folds[ff-1][pp] = beta.col(0);
                    beta_folds[pp][ff-1] = beta.col(0);
                    //iter_list(pp) = niter(0);
                } else 
                {
                    beta_folds[pp][ff-1] = beta;
                    //beta_folds[ff-1][pp] = beta;
                    //iter_list(pp) = niter;
                }
            }
            
            
        } // end loop over penalties
    } // end loop over cross validation folds
    
    bool use_weights = bool(weights.size() > 0);
    
    // compute cross validation scores for each model
    for (unsigned int pp = 0; pp < penalty.size(); pp++)
    {
        /*
        if (intercept)
        {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                out_of_fold_predictions_list[pp].row(i) = X.row(i) * beta_folds[pp][foldid(i)-1].bottomRows(p);
            }
        } else 
        {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                out_of_fold_predictions_list[pp].row(i) = X.row(i) * beta_folds[pp][foldid(i)-1];
            }
        }*/
        
        //int nlam = out_of_fold_predictions_list[pp].cols();
        int nlam = nlam_list[pp];
        VectorXd tempres(nlam);
        VectorXd tempsdres(nlam);
        
        // static enforces l = i comes before l = i + 1
        VectorXd tmpcv(nlam);
        VectorXd tmpss(nlam);
        tmpcv.setZero();
        tmpss.setZero();
        
        if (intercept)
        {
            
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                VectorXd cur_preds = ((X.row(i) * beta_folds[pp][foldid(i)-1].bottomRows(p))).array()
                                        + beta_folds[pp][foldid(i)-1].row(0).array();
                
                VectorXd resid = Y(i) - cur_preds.array();
                VectorXd tmp_cv;
                
                if (type_measure[0] == "mse")
                {
                    if (use_weights)
                    {
                        tmp_cv = resid.array().square().array() * weights(i);
                    } else 
                    {
                        tmp_cv = resid.array().square();
                    }
                } else if (type_measure[0] == "mae")
                {
                    if (use_weights)
                    {
                        tmp_cv = resid.array().abs().array() * weights(i);
                    } else 
                    {
                        tmp_cv = resid.array().abs();
                    }
                } else 
                {
                    std::invalid_argument("type.measure provided not availabe");
                }
                //  delta = x - mean_current
                VectorXd delta = tmp_cv.array() - tmpcv.array();
                tmpcv.array() += delta.array() / (i + 1);
                tmpss.array() += delta.array() * (tmp_cv.array() - tmpcv.array()).array();
            }
        } else 
        {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                VectorXd cur_preds = X.row(i) * beta_folds[pp][foldid(i)-1].bottomRows(p);
                
                VectorXd resid = Y(i) - cur_preds.array();
                VectorXd tmp_cv;
                
                if (type_measure[0] == "mse")
                {
                    if (use_weights)
                    {
                        tmp_cv = resid.array().square().array() * weights(i);
                    } else 
                    {
                        tmp_cv = resid.array().square();
                    }
                } else if (type_measure[0] == "mae")
                {
                    if (use_weights)
                    {
                        tmp_cv = resid.array().abs().array() * weights(i);
                    } else 
                    {
                        tmp_cv = resid.array().abs();
                    }
                } else 
                {
                    std::invalid_argument("type.measure provided not availabe");
                }
                //  delta = x - mean_current
                VectorXd delta = tmp_cv.array() - tmpcv.array();
                tmpcv.array() += delta.array() / (i + 1);
                tmpss.array() += delta.array() * (tmp_cv.array() - tmpcv.array()).array();
            }
        }
        
        tempres = tmpcv.array();
        //tmpcv.array() -= tempres.array();
        
        //tempsdres = ((tmpss.array() - ((tmpcv.array() - 5.0).array().square().array() / n) ) / (n * (n - 1))).array().sqrt();
        
        tempsdres = (tmpss.array() / (double(n - 1))).array().sqrt();
        
        //tempsdres = sqrt(tmpcv.array().square().mean() / (n - 1));
        
        xval_mean[pp] = tempres;
        xval_sd[pp] = tempsdres / sqrt(double(n));
    }


    delete solver;

    return List::create(Named("beta")   = beta_list,
                        Named("lambda") = lambda,
                        Named("niter")  = iter_list,
                        Named("loss")   = loss_list,
                        Named("cvm")    = xval_mean,
                        Named("cvsd")   = xval_sd,
                        //Named("pred")   = out_of_fold_predictions_list,
                        Named("d")      = d);
    END_RCPP
}




