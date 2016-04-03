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
    typedef Map<const MatrixXd> MapMatd;
    typedef Map<const VectorXd> MapVecd;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    
    const MapMatd X;                  // data matrix
    MapVec Y;                  // response vector
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
    
    
    MatrixXd XtX() const {
        return MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
        rankUpdate(X.adjoint());
    }
    
    MatrixXd XXt() const {
        return MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
        rankUpdate(X);
    }
    
    void compute_XtX_d_update_A()
    {
        
        // compute X'X
        
        if (nobs > nvars) 
        {
            XX = XtX();
        } else 
        {
            XX = XXt();
        }
        
        
        XX /= nobs;
        
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
        if (nobs > nvars)
        {
            res = A * beta_prev + XY;
        } else 
        {
            res = X.adjoint() * (Y - X * beta_prev) / double(nobs) + d * beta_prev;
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
            double denom = d + (1 - alpha) * lambda;
            double lam = alpha * lambda;
            soft_threshold(beta, u, lam, penalty_factor, denom);
        } else if (penalty == "scad") 
        {
            
        } else if (penalty == "mcp") 
        {
            soft_threshold_mcp(beta, u, lambda, penalty_factor, d, gamma);
        }
    }
    
    
public:
    oemDense(const Eigen::Ref<const MatrixXd>  &X_, 
             ConstGenericVector &Y_,
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
                             X(X_.data(), X_.rows(), X_.cols()),
                             Y(Y_.data(), Y_.size()),
                             penalty_factor(penalty_factor_),
                             penalty_factor_size(penalty_factor_.size()),
                             XXdim( std::min(X_.cols(), X_.rows()) ),
                             XY(X_.cols()), // add extra space if intercept but no standardize
                             XX(XXdim, XXdim),                                // add extra space if intercept but no standardize
                             alpha(alpha_),
                             gamma(gamma_)
    {}
    
    
    double compute_lambda_zero() 
    { 
        
        XY = X.transpose() * Y;
        XY /= nobs;
        
        
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
        return beta;
    }
};



#endif // OEM_DENSE_TALL_H
