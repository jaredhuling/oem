#ifndef OEMBASE_H
#define OEMBASE_H

#include <RcppEigen.h>
#include "utils.h"


template<typename VecTypeBeta, typename MatTypeX>
class oemBase
{
protected:
    
    const int nvars;         // dimension of beta
    const int nobs;          // number of rows
    
    MatTypeX X;              // design matrix
    MatTypeX A;              // A matrix
    VectorXd u;              // u
    VecTypeBeta beta;        // parameters to be optimized
    VecTypeBeta beta_prev;   // parameters from previous iteration
    
    double tol;              // tolerance for convergence
    
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
    
    int solve(int maxit)
    {
        int i;
        
        for(i = 0; i < maxit; ++i)
        {
            beta_prev = beta;
            
            update_u();
            update_beta();
            
            // print_row(i);
            
            if(converged())
                break;
            
        }
        
        
        return i + 1;
    }
    
    virtual VecTypeBeta get_beta() { return beta; }
};



#endif // OEMBASE_H