#ifndef OEM_BASE_H
#define OEM_BASE_H

#include <RcppEigen.h>
#include "utils.h"


template<typename VecTypeBeta>
class oemBase
{
protected:
    
    const int nvars;                  // dimension of beta
    const int nobs;                   // number of rows
    
    VectorXd u;                       // u vector
    
    VecTypeBeta beta;                 // parameters to be optimized
    VecTypeBeta beta_prev;            // parameters from previous iteration
    
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
            double tol_ = 1e-6) :
    nvars(p_), 
    nobs(n_),
    beta(p_),            // allocate space but do not set values
    beta_prev(p_),       // allocate space but do not set values
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
    
    virtual double get_lambda_zero() const { return 0; }
    virtual VecTypeBeta get_beta() { return beta; }
    
    virtual void init(double lambda_, std::string penalty_) {}
    virtual void init_warm(double lambda_) {}
};



#endif // OEM_BASE_H