#ifndef OEM_DENSE_H
#define OEM_DENSE_H

#include "oem_base.h"
#include "Spectra/SymEigsSolver.h"
#include "utils.h"



// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. x - z = 0
//
// x => beta
// z => -X * beta
// A => X
// b => y
// f(x) => 1/2 * ||Ax - b||^2
// g(z) => lambda * ||z||_1
class oemDense: public oemBase<Eigen::VectorXd> //Eigen::SparseVector<double>
{
protected:
    typedef float Scalar;
    typedef double Double;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Map<const Matrix> MapMat;
    typedef Map<const Vector> MapVec;
    typedef Map<VectorXd> MapVecd;
    typedef Map<Eigen::MatrixXd> MapMatd;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    
    const MapMatd X;           // data matrix
    const MapVecd Y;           // response vector
    VectorXd penalty_factor;   // penalty multiplication factors 
    int penalty_factor_size;   // size of penalty_factor vector
    int XXdim;                 // dimension of XX (different if n > p and p >= n)
    Vector XY;                 // X'Y
    MatrixXd XX;               // X'X
    MatrixXd A;                // A = d * I - X'X
    double d;                  // d value (largest eigenvalue of X'X)
    double alpha;              // alpha = mixing parameter for elastic net
    double gamma;              // extra tuning parameter for mcp/scad
    
    std::string penalty;       // penalty specified
    
    double lambda;             // L1 penalty
    double lambda0;            // minimum lambda to make coefficients all zero
    
    double threshval;
    
    
    void compute_XtX_d_update_A()
    {
        
        // compute X'X
        
        if (standardize) 
        {
            if (nobs > nvars) {
                XX = XtX_scaled(X, colmeans, colstd);
            } else 
            {
                XX = XXt_scaled(X, colmeans, colstd);
            }
            
        } else if (intercept && !standardize) 
        {
            Eigen::RowVectorXd colsums = colmeans * nobs;
            if (nobs > nvars) 
            {
                XX.bottomRightCorner(nvars, nvars) = XtX(X);
                XX.block(0,1,1,nvars) = colsums;
                XX.block(1,0,nvars,1) = colsums.transpose();
                XX(0,0) = nobs;
            } else 
            {
                XX = XXt(X);
                XX.array() += 1; // adding 1 to all of XX' for the intercept
            }
            
        } else 
        {
            if (nobs > nvars) 
            {
                XX = XtX(X);
            } else 
            {
                XX = XXt(X);
            }
        }
        
        
        Spectra::DenseSymMatProd<double> op(XX);
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, 1, 4);
        
        eigs.init();
        eigs.compute(1000, 0.0001);
        Vector eigenvals = eigs.eigenvalues();
        d = eigenvals[0];
        
        if (nobs > nvars)
        {
            A = -XX;
            A.diagonal().array() += d;
        }
    }
    
    void next_u(Vector &res)
    {
        if (nvars < nobs)
        {
            res = A * beta_prev + XY;
        } else 
        {
            if (standardize) 
            {
                res = X.adjoint() * (Y - X * beta_prev) + d * beta_prev;
                
            } else if (intercept && !standardize) 
            {
                // need to handle differently with intercept
                VectorXd resid  = Y - X * beta_prev.tail(nvars).matrix();
                resid.array() -= beta_prev(0);
                res.tail(nvars) = X.adjoint() * (resid) + d * beta_prev.tail(nvars);
                res(0) = resid.sum() + d * beta_prev(0);
                
            } else 
            {
                res = X.adjoint() * (Y - X * beta_prev) + d * beta_prev;
            }
        }
    }
    
    void next_beta(Vector &res)
    {
        if (penalty == "lasso")
        {
            soft_threshold(beta, u, lambda, penalty_factor, d);
        } else if (penalty == "ols")
        {
            beta = u / d;
        } else if (penalty == "elastic.net")
        {
            double denom = d + alpha;
            soft_threshold(beta, u, lambda, penalty_factor, denom);
        } else if (penalty == "scad") 
        {
            
        } else if (penalty == "mcp") 
        {
            soft_threshold_mcp(beta, u, lambda, penalty_factor, d, gamma);
        }
    }
    
    
public:
    oemDense(const MapMatd &X_, 
             const MapVecd &Y_,
             VectorXd &penalty_factor_,
             const double &alpha_,
             const double &gamma_,
             bool &intercept_,
             bool &standardize_,
             const double tol_ = 1e-6) :
    oemBase<Eigen::VectorXd>(X_.rows(), 
                             X_.cols(),
                             intercept_, 
                             standardize_,
                             tol_),
              X( Map<MatrixXd>(X_) ),
              Y( Map<VectorXd>(Y_) ),
              penalty_factor(penalty_factor_),
              penalty_factor_size(penalty_factor_.size()),
              XXdim(std::min(X_.cols(), X_.rows()) + intercept_ * (1 - standardize_) * (X_.rows() > X_.cols()) ),
              // only add extra row/column to XX if  intercept  and no standardize  AND nobs > nvars
              XY(X_.cols() + intercept_ * (1 - standardize_)), // add extra space if intercept but no standardize
              XX(XXdim, XXdim),                                // add extra space if intercept but no standardize
              alpha(alpha_),
              gamma(gamma_)
    {}
    
    
    double compute_lambda_zero() 
    { 
        
        
        meanY = Y.mean();
        colmeans = X.colwise().mean();
        
        
        if (standardize)
        {
            colstd = ((X.rowwise() - colmeans).array().square().colwise().sum() / (nobs - 1)).sqrt();
        }
        
        if (intercept && standardize) 
        {
            scaleY = (Y.array() - meanY).matrix().norm() * (1.0 / std::sqrt(double(nobs)));
            
            XY = ((X.rowwise() - colmeans).array().rowwise() / 
                      colstd.array()).array().matrix().adjoint() * 
                      ((Y.array() - meanY) / scaleY).matrix();
        } else if (intercept && !standardize) 
        {
            
            XY.tail(nvars) = X.transpose() * Y;
            XY(0) = Y.sum();
            
        } else if (!intercept && standardize)
        {
            scaleY = std::sqrt((Y.array() - meanY).square().sum() / (double(nobs - 1) ));
            
            // maybe don't center X?
            XY = ((X.rowwise() - colmeans).array().rowwise() / 
                      colstd.array()).array().matrix().adjoint() * 
                      ((Y.array() - meanY) / scaleY).matrix();
        } else 
        {
            XY = X.transpose() * Y;
        }
        
        
        compute_XtX_d_update_A();
        
        lambda0 = XY.cwiseAbs().maxCoeff();
        return lambda0; 
    }
    double get_d() { return d; }
    
    // init() is a cold start for the first lambda
    void init(double lambda_, std::string penalty_)
    {
        beta.setZero();
        
        lambda = lambda_;
        penalty = penalty_;
        
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    void init_warm(double lambda_)
    {
        lambda = lambda_;
        
    }
    
    VectorXd get_beta() 
    { 
        
        if (standardize) 
        {
            VectorXd betaret(nvars+1);
            betaret.setZero();
            
            betaret.tail(nvars) = beta.array() / colstd.transpose().array();
            
            betaret *= scaleY;
            
            if (intercept) 
            {
                betaret(0) = meanY - (beta.array() * colmeans.array()).array().sum();
            }
            return betaret;
            
        } else if (intercept && !standardize) 
        {
            return beta;
        } 
        
        VectorXd betaret(nvars+1);
        betaret(0) = 0;
        betaret.tail(nvars) = beta;
        return betaret;
    }
};



#endif // OEM_DENSE_TALL_H
