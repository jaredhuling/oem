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
class oemLogisticDense: public oemBase<Eigen::VectorXd> //Eigen::SparseVector<double>
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
    VectorXd W;                 // weight vector for IRLS
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
    int irls_maxit;
    double irls_tol;
    
    
    std::vector<std::vector<int> > grp_idx; // vector of vectors of the indexes for all members of each group
    std::string penalty;       // penalty specified
    
    double lambda;             // L1 penalty
    double lambda0;            // minimum lambda to make coefficients all zero
    
    double threshval;
    
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
    
    
    MatrixXd XtWX() const {
        return MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
        rankUpdate(X.adjoint() * (W.array().sqrt().matrix()).asDiagonal() );
    }
    
    MatrixXd XWXt() const {
        return MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
        rankUpdate( (W.array().sqrt().matrix()).asDiagonal() * X );
    }
    
    // function to be called once in the beginning
    // to get the locations of all members of each group
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
                    group_weights(g) = sqrt(double(grp_idx[g].size()));
                }
            }
        }
    }
    
    double sum_dev_resid(MapVec &y, VectorXd &prob)
    {
        double dev;
        for (int i = 0; i < nobs; ++i)
        {
            if (y(i) == 1)
            {
                dev += std::sqrt(2 * std::log(1/prob(i)));
            } else 
            {
                dev += std::sqrt(2 * std::log(1/(1 - prob(i))));
            }
        }
        return dev;
    }
    
    void compute_XtX_d_update_A()
    {
        
        // compute X'X
        
        if (nobs > nvars) 
        {
            XX = XtWX();
        } else 
        {
            XX = XWXt();
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
            res = X.adjoint() * ((W.array() * (Y - X * beta_prev).array() ) / double(nobs)).matrix() + d * beta_prev;
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
    oemLogisticDense(const Eigen::Ref<const MatrixXd>  &X_, 
                     ConstGenericVector &Y_,
                     const VectorXi &groups_,
                     const VectorXi &unique_groups_,
                     VectorXd &group_weights_,
                     VectorXd &penalty_factor_,
                     const double &alpha_,
                     const double &gamma_,
                     bool &intercept_,
                     bool &standardize_,
                     const int &irls_maxit_ = 100,
                     const double &irls_tol = 1e-6,
                     const double tol_ = 1e-6) :
    oemBase<Eigen::VectorXd>(X_.rows(), 
                             X_.cols(),
                             unique_groups_.size(),
                             intercept_, 
                             standardize_,
                             tol_),
                             X(X_.data(), X_.rows(), X_.cols()),
                             Y(Y_.data(), Y_.size()),
                             W(X_.rows()),
                             groups(groups_),
                             unique_groups(unique_groups_),
                             penalty_factor(penalty_factor_),
                             group_weights(group_weights_),
                             penalty_factor_size(penalty_factor_.size()),
                             XXdim( std::min(X_.cols(), X_.rows()) ),
                             XY(X_.cols()), // add extra space if intercept but no standardize
                             XX(XXdim, XXdim),                                // add extra space if intercept but no standardize
                             alpha(alpha_),
                             gamma(gamma_),
                             default_group_weights(bool(group_weights_.size() < 1)), // compute default weights if none given
                             grp_idx(unique_groups_.size())
    
    {}
    
    
    double compute_lambda_zero() 
    { 
        
        XY = X.transpose() * Y;
        XY /= nobs;
        
        
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
    
    // re-define solve to do IRLS
    // iterations
    virtual int solve(int maxit)
    {
        VectorXd beta_prev_irls;
        double dev = 1e6;
        double dev0;
        
        int i;
        int j;
        for (int i = 0; i < irls_maxit; ++i)
        {
            
            dev0 = dev;
            VectorXd prob;
            VectorXd grad;
            
            beta_prev = beta;
            
            // calculate mu hat
            prob = 1 / (1 + (-1 * (X * beta).array()).exp().array());
            
            
            // calculate Jacobian
            W = prob.array() * (1 - prob.array());
            
            
            // make sure no weights are too small
            for (int kk = 0; kk < nobs; ++kk)
            {
                if (W(i) < 1e-5) 
                {
                    W(i) = 1e-5;
                }
            }
            
            // compute XtX or XXt (depending on if n > p or not)
            // and compute A = dI - XtX (if n > p)
            compute_XtX_d_update_A();
            
            
            // compute X'Wz
            // only for p < n case
            if (nobs > nvars)
            {
                grad = X.adjoint() * (Y.array() - prob.array()).matrix() / double(nobs);
                // not sure why the following doesn't work but the above, which seems
                // wrong does work
                //grad = X.adjoint() * ( W.array() * (Y.array() - prob.array()).array()).matrix();
                XY = XX * beta + grad;
            } 
            
            for(j = 0; j < maxit; ++j)
            {
                    
                update_u();
                
                update_beta();
                
                if(converged())
                    break;
                
            }
            
            dev = sum_dev_resid(Y, prob);
            std::cout << "dev resid" << dev << std::endl;
            if (std::abs(dev - dev0) / (0.1 + std::abs(dev) ) < irls_tol)
                break;
        }
        
        
        return i + 1;
    }
    
    VectorXd get_beta() 
    { 
        return beta;
    }
};



#endif // OEM_DENSE_TALL_H
