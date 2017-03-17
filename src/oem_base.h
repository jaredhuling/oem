#ifndef OEM_BASE_H
#define OEM_BASE_H

#include <RcppEigen.h>
#include "utils.h"


template<typename VecTypeBeta>
class oemBase
{
protected:
    
    const int nvars;                  // dimension of beta
    int nobs;                         // number of rows
    const int ngroups;                // number of groups for group lasso
    
    bool intercept;                   //
    bool standardize;                 //
    
    double meanY;
    double scaleY;
    
    VectorXd u;                       // u vector
    
    VecTypeBeta beta;                 // parameters to be optimized
    VecTypeBeta beta_prev;            // parameters from previous iteration
    VecTypeBeta beta_prev_irls;       // parameters from previous irls iteration
    
    Eigen::RowVectorXd colmeans;      // column means of X
    Eigen::RowVectorXd colstd;        // column std devs of X
    
    double tol;                       // tolerance for convergence
    
    virtual void next_beta(VecTypeBeta &res) = 0;
    
    virtual void next_u(VectorXd &res) = 0;
    
    virtual bool converged()
    {
        return (stopRule(beta, beta_prev, tol));
    }
    
    
    void print_row(int iter)
    {
        const char sep = ' ';
        
        Rcpp::Rcout << std::left << std::setw(7)  << std::setfill(sep) << iter;
        Rcpp::Rcout << std::endl;
    }
    
    
public:
    oemBase(int n_, 
            int p_,
            int ngroups_,
            bool intercept_,
            bool standardize_,
            double tol_ = 1e-6) :
    nvars(p_), 
    nobs(n_),
    ngroups(ngroups_),
    intercept(intercept_),
    standardize(standardize_),
    u(p_),               // allocate space but do not set values
    beta(p_),            // allocate space but do not set values
    beta_prev(p_),       // allocate space but do not set values
    beta_prev_irls(p_),
    colmeans(p_),
    colstd(p_),
    tol(tol_)
    {}
    
    virtual ~oemBase() {}
    
    void update_u()
    {
        //VecTypeBeta newbeta(nvars);
        next_u(u);
        //beta.swap(newbeta);
    }
    
    void update_beta()
    {
        //VecTypeBeta newbeta(nvars);
        next_beta(beta);
        //beta.swap(newbeta);
    }
    
    virtual int solve(int maxit)
    {
        int i;
        
        for(i = 0; i < maxit; ++i)
        {
            
            beta_prev = beta;
            
            update_u();
            
            update_beta();
            
            if(converged())
                break;
            
        }
        
        
        return i + 1;
    }
    
    virtual void init_oem() {}
    
    virtual void init_xtx(bool add_int_) {}
    virtual void update_xtx(int fold_) {}
    virtual double compute_lambda_zero() { return 0; }
    virtual VecTypeBeta get_beta() { return beta; }
    virtual double get_d() { return 0; }
    
    Eigen::RowVectorXd get_X_colmeans() {return colmeans;}
    Eigen::RowVectorXd get_X_colstd() {return colstd;}
    
    virtual double get_loss() { return 1e99; }
    
    virtual void init(double lambda_, std::string penalty_,
                      double alpha_, double gamma_, double tau_) {}
    virtual void init_warm(double lambda_) {}
};



#endif // OEM_BASE_H