#ifndef OEM_H
#define OEM_H

#include "oem_base.h"
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
class oem: public oemBase<Eigen::VectorXd, Eigen::MatrixXd> //Eigen::SparseVector<double>
{
protected:
    typedef float Scalar;
    typedef double Double;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    
    MapMat X;                  // data matrix
    MapVec Y;                  // response vector
    Vector XY;                 // X'Y
    MatrixXd XX;               // X'X
    MatrixXd A;                // A 
    
    Scalar lambda;             // L1 penalty
    Scalar lambda0;            // minimum lambda to make coefficients all zero
    
    double threshval;
    
    ArrayXd penalty_factor;       // penalty multiplication factors 
    int penalty_factor_size;
    
    void next_u(Vector &res)
    {
        
        
        
    }
    
    void next_beta(Vector &res)
    {
        
        
        
    }
    
    
    
public:
    oem(ConstGenericMatrix &X_, 
               ConstGenericVector &Y_,
               ArrayXd &penalty_factor_,
               double tol_ = 1e-6) :
    oemBase<Eigen::VectorXd, Eigen::MatrixXd>(X_.rows(), X_.cols(),
              tol_),
              X(X_.data(), X_.rows(), X_.cols()),
              Y(Y_.data(), Y_.size()),
              penalty_factor(penalty_factor_),
              penalty_factor_size(penalty_factor_.size()),
              XY(X.transpose() * Y),
              XX(XtX(X)),
              lambda0(XY.cwiseAbs().maxCoeff())
    {}
    
    double get_lambda_zero() const { return lambda0; }
    
    // init() is a cold start for the first lambda
    void init(double lambda_)
    {
        beta.setZero();
        
        lambda = lambda_;
        
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    void init_warm(double lambda_)
    {
        lambda = lambda_;
        
    }
};



#endif // OEM_H
