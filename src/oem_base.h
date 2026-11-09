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

    // Smallest lambda at which the current fit is a fixed point of the
    // thresholding step, so that no coefficient that is zero in the current
    // fit becomes nonzero. When the current fit is the solution for a very
    // large lambda, every penalized coefficient is zero and the unpenalized
    // ones are fitted, so the result is the smallest lambda whose solution
    // has all penalized coefficients equal to zero. The search uses each
    // class's own thresholding step, so it accounts for penalty factors,
    // group weights, the elastic net mixing parameter, and the intercept.
    // Returns a negative value if no finite lambda keeps the fit fixed,
    // as for a pure ridge penalty.
    double lambda_max_at_current_fit()
    {
        const VecTypeBeta beta_null = beta;
        beta_prev = beta;
        update_u();

        const double scale = u.cwiseAbs().maxCoeff();
        const double zero_tol = 1e-12 * (1.0 + beta_null.cwiseAbs().maxCoeff());

        // true if thresholding u at lam leaves every zero coefficient at zero
        auto stays_null = [&](double lam) {
            init_warm(lam);
            beta = beta_null;
            update_beta();
            for (int j = 0; j < beta.size(); ++j)
            {
                if (beta_null(j) == 0.0 && std::abs(beta(j)) > zero_tol) return false;
            }
            return true;
        };

        double result = -1.0;
        if (!(scale > 0.0))
        {
            result = 0.0;
        } else
        {
            double hi = scale;
            int k = 0;
            while (!stays_null(hi) && k < 60)
            {
                hi *= 2.0;
                ++k;
            }
            if (stays_null(hi))
            {
                double lo = 0.0;
                for (int it = 0; it < 100 && (hi - lo) > 1e-10 * hi; ++it)
                {
                    double mid = 0.5 * (lo + hi);
                    if (stays_null(mid)) hi = mid;
                    else lo = mid;
                }
                result = hi;
            }
        }
        beta = beta_null;
        beta_prev = beta_null;
        return result;
    }
    virtual VecTypeBeta get_beta() { return beta; }
    virtual double get_d() { return 0; }
    
    Eigen::RowVectorXd get_X_colmeans() {return colmeans;}
    Eigen::RowVectorXd get_X_colstd() {return colstd;}
    
    virtual double get_loss() { return 1e99; }
    
    virtual void init(double lambda_, std::string penalty_,
                      double alpha_, double gamma_, double tau_) {}
    virtual void init_warm(double lambda_) {}
};


// Largest lambda of the default sequence for one penalty: the smallest lambda
// at which every penalized coefficient is zero. The solver first fits the model
// in which only the unpenalized coefficients (and the intercept) are free, by
// fitting at a very large lambda with the L1 form of the penalty family. Lasso,
// MCP, and SCAD share the zero region |u_j| <= lambda * pf_j, and the group
// penalties share ||u_g|| <= lambda * w_g, so the L1 form gives the same value.
// The ".net" penalties apply alpha * lambda to the L1 part, so their value is
// divided by alpha. Returns a negative value when there is no such lambda (for
// "ols", or alpha = 0), in which case the caller keeps its previous sequence.
template<typename SolverType>
inline double oem_lambda_max(SolverType *solver, const std::string &penalty,
                             double alpha, double gamma, double tau, int maxit)
{
    if (penalty == "ols") return -1.0;

    std::string base_penalty("lasso");
    if (penalty == "sparse.grp.lasso")
    {
        base_penalty = penalty;
    } else if (penalty.compare(0, 4, "grp.") == 0)
    {
        base_penalty = "grp.lasso";
    }

    solver->init(1e100, base_penalty, alpha, gamma, tau);
    solver->solve(maxit);
    double lmax = solver->lambda_max_at_current_fit();

    if (lmax > 0.0 && penalty.find(".net") != std::string::npos)
    {
        lmax = (alpha > 0.0) ? lmax / alpha : -1.0;
    }
    return lmax;
}

// Log-spaced sequence from lmax down to lmin_ratio * lmax.
inline Eigen::VectorXd oem_lambda_sequence(double lmax, double lmin_ratio, int nlambda)
{
    Eigen::VectorXd out(nlambda);
    out.setLinSpaced(nlambda, std::log(lmax), std::log(lmin_ratio * lmax));
    return out.array().exp().matrix();
}


#endif // OEM_BASE_H