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
    int penalty_factor_size;
    Vector XY;                 // X'Y
    MatrixXd XX;               // X'X
    MatrixXd A;                // A = d * I - X'X
    double d;                  // d value (largest eigenvalue of X'X)
    double alpha;              // alpha = mixing parameter for elastic net
    double gamma;              // extra tuning parameter for mcp/scad
    std::string penalty;       // penalty specified
    
    Scalar lambda;             // L1 penalty
    Scalar lambda0;            // minimum lambda to make coefficients all zero
    
    double threshval;
    
    
    void compute_XtX_d_update_A()
    {
        
        // compute X'X
        XX = XtX(X);
        
        Spectra::DenseSymMatProd<double> op(XX);
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, 1, 4);
        
        eigs.init();
        eigs.compute(1000, 0.0001);
        Vector eigenvals = eigs.eigenvalues();
        d = eigenvals[0];
        
        A = -XX;
        A.diagonal().array() += d;
    }
    
    void next_u(Vector &res)
    {
        res = A * beta_prev + XY;
    }
    
    void next_beta(Vector &res)
    {
        if (penalty == "lasso")
        {
            soft_threshold(beta, u, lambda, penalty_factor, d);
        } else if (penalty == "ols")
        {
            beta = u / d;
        }
    }
    
    
public:
    oemDense(const MapMatd &X_, 
             const MapVecd &Y_,
             VectorXd &penalty_factor_,
             const double &alpha_,
             const double &gamma_,
             const double tol_ = 1e-6) :
    oemBase<Eigen::VectorXd>(X_.rows(), X_.cols(),
              tol_),
              X( Map<MatrixXd>(X_) ),
              Y( Map<VectorXd>(Y_) ),
              penalty_factor(penalty_factor_),
              penalty_factor_size(penalty_factor_.size()),
              XY(X.transpose() * Y),
              XX(X_.cols(), X_.cols()),
              alpha(alpha_),
              gamma(gamma_),
              lambda0(XY.cwiseAbs().maxCoeff())
    {}
    
    
    double get_lambda_zero() const { return lambda0; }
    double get_d() { return d; }
    
    // init() is a cold start for the first lambda
    void init(double lambda_, std::string penalty_)
    {
        beta.setZero();
        compute_XtX_d_update_A();
        
        lambda = lambda_;
        penalty = penalty_;
        
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    void init_warm(double lambda_)
    {
        lambda = lambda_;
        
    }
};



#endif // OEM_DENSE_TALL_H
