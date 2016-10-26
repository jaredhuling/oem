#ifndef OEM_BIG_H
#define OEM_BIG_H



#include "oem_base.h"
#include "Spectra/SymEigsSolver.h"
#include "utils.h"
#include <bigmemory/MatrixAccessor.hpp>
#include <bigmemory/BigMatrix.h>


// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
class oemBig: public oemBase<Eigen::VectorXd> 
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
    typedef Map<VectorXi> MapVeci;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    
    const MapMatd X;            // data matrix
    MapVec Y;                   // response vector
    VectorXd weights;
    VectorXi groups;            // vector of group membersihp indexes 
    VectorXi unique_groups;     // vector of all unique groups
    VectorXd penalty_factor;    // penalty multiplication factors 
    VectorXd group_weights;     // group lasso penalty multiplication factors 
    int penalty_factor_size;    // size of penalty_factor vector
    int XXdim;                  // dimension of XX (different if n > p and p >= n)
    Vector XY;                  // X'Y
    MatrixXd XX;                // X'X
    MatrixXd A;                 // A = d * I - X'X
    double d;                   // d value (largest eigenvalue of X'X)
    double alpha;               // alpha = mixing parameter for elastic net
    double gamma;               // extra tuning parameter for mcp/scad
    bool default_group_weights; // do we need to compute default group weights?
    
    
    std::vector<std::vector<int> > grp_idx; // vector of vectors of the indexes for all members of each group
    std::string penalty;       // penalty specified
    
    double lambda;             // L1 penalty
    double lambda0;            // minimum lambda to make coefficients all zero
    
    double threshval;
    int wt_len;
    
    Eigen::RowVectorXd colsums;
    Eigen::RowVectorXd colsq;
    Eigen::VectorXd colsq_inv;
    
    static void soft_threshold(VectorXd &res, const VectorXd &vec, const double &penalty, 
                               VectorXd &pen_fact, double &d)
    {
        int v_size = vec.size();
        res.setZero();
        
        const double *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            double total_pen = pen_fact(i) * penalty;
            
            if(ptr[i] > total_pen)
                res(i) = (ptr[i] - total_pen)/d;
            else if(ptr[i] < -total_pen)
                res(i) = (ptr[i] + total_pen)/d;
        }
    }
    
    static void soft_threshold_mcp(VectorXd &res, const VectorXd &vec, const double &penalty, 
                                   VectorXd &pen_fact, double &d, double &gamma)
    {
        int v_size = vec.size();
        res.setZero();
        double gammad = gamma * d;
        double d_minus_gammainv = d - 1 / gamma;
        
        
        const double *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            double total_pen = pen_fact(i) * penalty;
            
            if (std::abs(ptr[i]) > gammad * total_pen)
                res(i) = ptr[i]/d;
            else if(ptr[i] > total_pen)
                res(i) = (ptr[i] - total_pen)/(d_minus_gammainv);
            else if(ptr[i] < -total_pen)
                res(i) = (ptr[i] + total_pen)/(d_minus_gammainv);
            
        }
        
    }
    
    static void soft_threshold_scad(VectorXd &res, const VectorXd &vec, const double &penalty, 
                                    VectorXd &pen_fact, double &d, double &gamma)
    {
        int v_size = vec.size();
        res.setZero();
        double gammad = gamma * d;
        double gamma_minus1_d = (gamma - 1) * d;
        
        const double *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            double total_pen = pen_fact(i) * penalty;
            
            if (std::abs(ptr[i]) > gammad * total_pen)
                res(i) = ptr[i]/d;
            else if (std::abs(ptr[i]) > (d + 1) * total_pen)
            {
                double gam_ptr = (gamma - 1) * ptr[i];
                double gam_pen = gamma * total_pen;
                if(gam_ptr > gam_pen)
                    res(i) = (gam_ptr - gam_pen)/(gamma_minus1_d - 1);
                else if(gam_ptr < -gam_pen)
                    res(i) = (gam_ptr + gam_pen)/(gamma_minus1_d - 1);
            }
            else if(ptr[i] > total_pen)
                res(i) = (ptr[i] - total_pen)/d;
            else if(ptr[i] < -total_pen)
                res(i) = (ptr[i] + total_pen)/d;
            
        }
    }
    
    static void block_soft_threshold(VectorXd &res, const VectorXd &vec, const double &penalty,
                                     VectorXd &pen_fact, double &d,
                                     std::vector<std::vector<int> > &grp_idx, 
                                     const int &ngroups, VectorXi &unique_grps, VectorXi &grps)
    {
        //int v_size = vec.size();
        res.setZero();
        
        for (int g = 0; g < ngroups; ++g) 
        {
            double thresh_factor;
            std::vector<int> gr_idx = grp_idx[g];
            /*
            for (int v = 0; v < v_size; ++v) 
            {
            if (grps(v) == unique_grps(g)) 
            {
            gr_idx.push_back(v);
            }
            }
            */
            if (unique_grps(g) == 0) 
            {
                thresh_factor = 1;
            } else 
            {
                double ds_norm = 0;
                for (std::vector<int>::size_type v = 0; v < gr_idx.size(); ++v)
                {
                    int c_idx = gr_idx[v];
                    ds_norm += std::pow(vec(c_idx), 2);
                }
                ds_norm = std::sqrt(ds_norm);
                // double grp_wts = sqrt(gr_idx.size());
                double grp_wts = pen_fact(g);
                thresh_factor = std::max(0.0, 1 - penalty * grp_wts / (ds_norm) );
            }
            if (thresh_factor != 0.0)
            {
                for (std::vector<int>::size_type v = 0; v < gr_idx.size(); ++v)
                {
                    int c_idx = gr_idx[v];
                    res(c_idx) = vec(c_idx) * thresh_factor / d;
                }
            }
        }
    }
    
    
    MatrixXd XtX() const {
        return MatrixXd(nvars, nvars).setZero().selfadjointView<Lower>().
        rankUpdate(X.adjoint());
    }
    
    MatrixXd XXt() const {
        return MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().
        rankUpdate(X);
    }
    
    MatrixXd XtWX() const {
        return MatrixXd(nvars, nvars).setZero().selfadjointView<Lower>().
        rankUpdate(X.adjoint() * (weights.array().sqrt().matrix()).asDiagonal() );
    }
    
    MatrixXd XWXt() const {
        return MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().
        rankUpdate( (weights.array().sqrt().matrix()).asDiagonal() * X );
    }
    
    void get_group_indexes()
    {
        if (penalty == "grp.lasso") 
        {
            
            grp_idx.reserve(ngroups);
            for (int g = 0; g < ngroups; ++g) 
            {
                // find all variables in group number g
                std::vector<int> idx_tmp;
                for (int v = 0; v < nvars; ++v) 
                {
                    if (groups(v) == unique_groups(g)) 
                    {
                        idx_tmp.push_back(v);
                    }
                }
                grp_idx[g] = idx_tmp;
            }
            // if group weights were not specified,
            // then set the group weight for each
            // group to be the sqrt of the size of the
            // group
            if (default_group_weights)
            {
                group_weights.resize(ngroups);
                for (int g = 0; g < ngroups; ++g) 
                {
                    group_weights(g) = std::sqrt(double(grp_idx[g].size()));
                }
            }
        }
    }
    
    void compute_XtX_d_update_A()
    {
        
        // compute X'X
        // if weights specified, compute X'WX instead
        if (wt_len)
        {
            if (nobs > nvars + int(intercept)) 
            {
                if (intercept)
                {
                    if (standardize)
                    {
                        colsums.array() *= colsq_inv.array();
                        XX.bottomRightCorner(nvars, nvars) = colsq_inv.asDiagonal() * XtWX() * colsq_inv.asDiagonal();
                    } else 
                    {
                        XX.bottomRightCorner(nvars, nvars) = XtWX();
                    }
                    // colsums should already be standardized if standardize = TRUE
                    XX.block(0,1,1,nvars) = colsums;
                    XX.block(1,0,nvars,1) = colsums.transpose();
                    XX(0,0) = weights.array().sum();
                } else 
                {
                    if (standardize)
                    {
                        XX = colsq_inv.asDiagonal() * XtWX() * colsq_inv.asDiagonal();
                    } else 
                    {
                        XX = XtWX();
                    }
                }
            } else 
            {
                XX = XWXt();
                if (intercept)
                    XX.array() += 1; // adding 1 to all of XX' for the intercept
            }
        } else 
        {
            if (nobs > nvars + int(intercept)) 
            {
                if (intercept)
                {
                    if (standardize)
                    {
                        colsums.array() *= colsq_inv.array();
                        XX.bottomRightCorner(nvars, nvars) = colsq_inv.asDiagonal() * XtX() * colsq_inv.asDiagonal();
                    } else 
                    {
                        XX.bottomRightCorner(nvars, nvars) = XtX();
                    }
                    // colsums should already be standardized if standardize = TRUE
                    XX.block(0,1,1,nvars) = colsums;
                    XX.block(1,0,nvars,1) = colsums.transpose();
                    XX(0,0) = nobs;
                } else 
                {
                    if (standardize)
                    {
                        XX = colsq_inv.asDiagonal() * XtX() * colsq_inv.asDiagonal();
                    } else 
                    {
                        XX = XtX();
                    }
                }
            } else 
            {
                XX = XXt();
                if (intercept)
                    XX.array() += 1; // adding 1 to all of XX' for the intercept
            }
        }
        
        XX /= nobs;
        
        Spectra::DenseSymMatProd<double> op(XX);
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, 1, 4);
        
        eigs.init();
        eigs.compute(10000, 1e-10);
        Vector eigenvals = eigs.eigenvalues();
        d = eigenvals[0] * 1.005; // multiply by an increasing factor to be safe
        
        if (nobs > nvars)
        {
            A = -XX;
            A.diagonal().array() += d;
        }
    }
    
    void next_u(Vector &res)
    {
        if (nobs > nvars + int(intercept))
        {
            res.noalias() = A * beta_prev + XY;
        } else 
        {
            if (wt_len)
            {
                res.noalias() = X.adjoint() * ((Y - X * beta_prev).array() * weights.array().square()).matrix() / double(nobs) + d * beta_prev;
            } else
            {
                res.noalias() = X.adjoint() * (Y - X * beta_prev) / double(nobs) + d * beta_prev;
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
            double denom = d + (1 - alpha) * lambda;
            double lam = alpha * lambda;
            soft_threshold(beta, u, lam, penalty_factor, denom);
        } else if (penalty == "scad") 
        {
            soft_threshold_scad(beta, u, lambda, penalty_factor, d, gamma);
        } else if (penalty == "mcp") 
        {
            soft_threshold_mcp(beta, u, lambda, penalty_factor, d, gamma);
        } else if (penalty == "grp.lasso")
        {
            block_soft_threshold(beta, u, lambda, group_weights,
                                 d, grp_idx, ngroups, 
                                 unique_groups, groups);
        }
        
    }
    
    
    public:
        oemBig(const Map<MatrixXd> &X_, 
               ConstGenericVector &Y_,
               const VectorXd &weights_,
               const VectorXi &groups_,
               const VectorXi &unique_groups_,
               VectorXd &group_weights_,
               VectorXd &penalty_factor_,
               const double &alpha_,
               const double &gamma_,
               bool &intercept_,
               bool &standardize_,
               const double tol_ = 1e-6) :
        oemBase<Eigen::VectorXd>(X_.rows(), 
                                 X_.cols(),
                                 unique_groups_.size(),
                                 intercept_, 
                                 standardize_,
                                 tol_),
                                 X(X_.data(), X_.rows(), X_.cols()),
                                 Y(Y_.data(), Y_.size()),
                                 weights(weights_),
                                 groups(groups_),
                                 unique_groups(unique_groups_),
                                 penalty_factor(penalty_factor_),
                                 group_weights(group_weights_),
                                 penalty_factor_size(penalty_factor_.size()),
                                 XXdim( std::min(X_.cols() + int(intercept_) , X_.rows()) ),
                                 XY(X_.cols() + int(intercept_) ), // add extra space if intercept
                                 XX(XXdim, XXdim),                 // add extra space if intercept
                                 alpha(alpha_),
                                 gamma(gamma_),
                                 default_group_weights(bool(group_weights_.size() < 1)), // compute default weights if none given
                                                                                    grp_idx(unique_groups_.size()),
                                 colsums(X_.cols()),
                                 colsq(X_.cols()),
                                 colsq_inv(X_.cols())
        
                                                                                    {}
        
        
        double compute_lambda_zero() 
        { 
            
            wt_len = weights.size();
            
            if (standardize)
            {
                if (wt_len)
                {
                    colsq = ( weights.asDiagonal() * (X.array().square().matrix()) ).colwise().sum() / (nobs - 1);
                } else 
                {
                    colsq = X.array().square().matrix().colwise().sum() / (nobs - 1);
                }
                    colsq_inv = 1 / colsq.array().sqrt();
            }
            
            if (wt_len)
            {
                if (intercept)
                {
                    XY.tail(nvars) = X.transpose() * (Y.array() * weights.array()).matrix();
                    XY(0) = (weights.array().sqrt() * Y.array()).sum();
                } else
                {
                    XY.noalias() = X.transpose() * (Y.array() * weights.array()).matrix();
                }
            } else
            {
                if (intercept)
                {
                    XY.tail(nvars) = X.transpose() * Y;
                    XY(0) = Y.sum();
                } else 
                {
                    XY.noalias() = X.transpose() * Y;
                }
            }
            
            if (standardize)
            {
                if (intercept)
                {
                    XY.tail(nvars).array() *= colsq_inv.array();
                } else 
                {
                    XY.array() *= colsq_inv.array();
                }
            }
            
            XY /= nobs;
            
            if (intercept) 
            {
                u.resize(nvars + 1);
                beta.resize(nvars + 1);
                beta_prev.resize(nvars + 1);
                colsums = X.colwise().sum();
            }
            
            // compute XtX or XXt (depending on if n > p or not)
            // and compute A = dI - XtX (if n > p)
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
            
            // get indexes of members of each group.
            // best to do just once in the beginning
            get_group_indexes();
            
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
                if (intercept)
                {
                    VectorXd beta_ret = beta;
                    beta_ret.tail(nvars).array() *= colsq_inv.array();
                    return(beta_ret);
                } else 
                {
                    return (beta.array() * colsq_inv.array()).matrix();
                }
            } else 
            {
                return beta;
            }
        }
        
        virtual double get_loss()
        {
            double loss;
            if (wt_len)
            {
                loss = ((Y - X * beta).array().square() * weights.array()).sum();
            } else 
            {
                loss = (Y - X * beta).array().square().sum();
            }
            return loss;
        }
        };



#endif // OEM_BIG_H
