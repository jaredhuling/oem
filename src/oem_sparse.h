#ifndef OEM_SPARSE_H
#define OEM_SPARSE_H

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
class oemSparse: public oemBase<Eigen::VectorXd> //Eigen::SparseVector<double>
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
    typedef Eigen::MappedSparseMatrix<double> MSpMat;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    
    const MSpMat X;             // sparse data matrix
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
    double xxdiag;
    double intval;
    
    
    std::vector<std::vector<int> > grp_idx; // vector of vectors of the indexes for all members of each group
    std::string penalty;        // penalty specified
    
    double lambda;              // L1 penalty
    double lambda0;             // minimum lambda to make coefficients all zero
    
    double threshval;
    int wt_len;
    
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
                    ds_norm += pow(vec(c_idx), 2);
                }
                ds_norm = sqrt(ds_norm);
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
    
    
    SpMat XtX() const {
        return SpMat(XXdim, XXdim).selfadjointView<Upper>().
        rankUpdate(X.adjoint());
    }
    
    SpMat XXt() const {
        return SpMat(XXdim, XXdim).selfadjointView<Upper>().
        rankUpdate(X);
    }
    
    SpMat XtWX() const {
        return SpMat(nvars, nvars).selfadjointView<Upper>().
        rankUpdate(X.adjoint() * (weights.array().sqrt().matrix()).asDiagonal() );
    }
    
    SpMat XWXt() const {
        return SpMat(nobs, nobs).selfadjointView<Upper>().
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
                for (int v = 0; v < groups.size(); ++v) 
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
                    group_weights(g) = sqrt(double(grp_idx[g].size()));
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
            if (nobs > nvars) 
            {
                if (intercept)
                {
                    // compute X'X with intercept
                    XX.bottomRightCorner(nvars, nvars) = XtWX();
                    xxdiag = XX.diagonal().tail(nvars).mean();
                    intval = sqrt(xxdiag / double(nobs));
                    
                    Eigen::RowVectorXd colsums = X.adjoint() * weights; 
                    colsums.array() *= intval;
                    
                    XX.block(0,1,1,nvars) = colsums;
                    XX.block(1,0,nvars,1) = colsums.transpose();
                    XX(0,0) = weights.sum() * xxdiag;
                } else
                {
                    XX = XtWX();
                }
            } else 
            {
                XX = XWXt();
                if (intercept)
                {
                    XX += MatrixXd(XXdim, XXdim).selfadjointView<Upper>().rankUpdate(weights);
                }
            }
        } else 
        {
            if (nobs > nvars) 
            {
                
                if (intercept)
                {
                    // compute X'X with intercept
                    XX.bottomRightCorner(nvars, nvars) = XtX();
                    
                    xxdiag = XX.diagonal().tail(nvars).mean();
                    intval = sqrt(xxdiag / nobs);
                    
                    Eigen::RowVectorXd colsums = X.adjoint() * VectorXd::Ones( nobs );
                    colsums.array() *= intval;
                    
                    
                    XX.block(0,1,1,nvars) = colsums;
                    XX.block(1,0,nvars,1) = colsums.transpose();
                    XX(0,0) = xxdiag;
                    
                } else 
                {
                    XX = XtX();
                }
            } else 
            {
                XX = XXt();
                if (intercept)
                {
                    XX.array() += 1;
                }
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
        if (nobs > nvars)
        {
            res.noalias() = A * beta_prev + XY;
        } else 
        {
            res.noalias() = X.adjoint() * (Y - X * beta_prev) / double(nobs) + d * beta_prev;
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
    oemSparse(const MSpMat &X_, 
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
                             X(X_),
                             Y(Y_.data(), Y_.size()),
                             weights(weights_),
                             groups(groups_),
                             unique_groups(unique_groups_),
                             penalty_factor(penalty_factor_),
                             group_weights(group_weights_),
                             penalty_factor_size(penalty_factor_.size()),
                             XXdim( std::min(X_.cols(), X_.rows()) + intercept_ * (X_.rows() > X_.cols()) ),
                             XY(XXdim),            // add extra space if intercept and n > p
                             XX(XXdim, XXdim),     // add extra space if intercept and n > p
                             alpha(alpha_),
                             gamma(gamma_),
                             default_group_weights(bool(group_weights_.size() < 1)), // compute default weights if none given
                             grp_idx(unique_groups_.size())
    
    {}
    
    
    double compute_lambda_zero() 
    { 
        
        if (intercept)
        {
            u.resize(nvars + 1);
            beta.resize(nvars + 1);
            beta_prev.resize(nvars + 1);
        }
        
        wt_len = weights.size();
        
        // compute XtX or XXt (depending on if n > p or not)
        // and compute A = dI - XtX (if n > p)
        compute_XtX_d_update_A();
        
        if (wt_len)
        {
            if (intercept)
            {
                XY.tail(nvars) = X.transpose() * (Y.array() * weights.array()).matrix();
                if (nobs > nvars)
                {
                    XY(0) = (Y.array() * weights.array()).sum() * intval;
                } else 
                {
                    XY(0) = (Y.array() * weights.array()).sum();
                }
            } else 
            {
                XY.noalias() = X.transpose() * (Y.array() * weights.array()).matrix();
            }
        } else
        {
            if (intercept)
            {
                XY.tail(nvars) = X.transpose() * Y;
                if (nobs > nvars)
                {
                    XY(0) = Y.sum() * intval;
                } else 
                {
                    XY(0) = Y.sum();
                }
                
            } else 
            {
                XY.noalias() = X.transpose() * Y;
            }
        }
        
        XY /= nobs;
        
        
        if (intercept)
        {
            lambda0 = XY.tail(nvars).cwiseAbs().maxCoeff();
        } else 
        {
            lambda0 = XY.cwiseAbs().maxCoeff();
        }
        
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
        if (intercept && nobs > nvars)
        {
            beta(0) *= (intval);
        }
        return beta;
    }
    
    virtual double get_loss()
    {
        double loss;
        loss = (Y - X * beta).array().square().sum();
        return loss;
    }
};



#endif // OEM_SPARSE_H
