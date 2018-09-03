#ifndef OEM_DENSE_H
#define OEM_DENSE_H

#ifdef _OPENMP
    #define has_openmp 1
    #include <omp.h>
#else 
    #define has_openmp 0
    #define omp_get_num_threads() 1
    #define omp_set_num_threads(x) 1
    #define omp_get_max_threads() 1
    #define omp_get_num_threads() 1
    #define omp_get_num_procs() 1
    #define omp_get_thread_limit() 1
    #define omp_set_dynamic(x) 1
    #define omp_get_thread_num() 0
#endif

#include "oem_base.h"
#include "Spectra/SymEigsSolver.h"
#include "utils.h"



// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
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
    bool default_group_weights; // do we need to compute default group weights?
    int ncores;
    
    
    std::vector<std::vector<int> > grp_idx; // vector of vectors of the indexes for all members of each group
    bool accelerate;
    std::string penalty;        // penalty specified
    
    double lambda;              // L1 penalty
    double lambda0;             // minimum lambda to make coefficients all zero
    double alpha;               // alpha = mixing parameter for elastic net
    double gamma;               // extra tuning parameter for mcp/scad
    double tau;                 // mixing parameter for group sparse penalties
    
    double threshval;
    int wt_len;
    double ak, ak_prev;
    bool found_grp_idx;
    
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
        double d_minus_gammainv = d - 1.0 / gamma;
        
        
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
        double gamma_minus1_d = (gamma - 1.0) * d;
        
        const double *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            double total_pen = pen_fact(i) * penalty;
            
            if (std::abs(ptr[i]) > gammad * total_pen)
                res(i) = ptr[i]/d;
            else if (std::abs(ptr[i]) > (d + 1.0) * total_pen)
            {
                double gam_ptr = (gamma - 1.0) * ptr[i];
                double gam_pen = gamma * total_pen;
                if(gam_ptr > gam_pen)
                    res(i) = (gam_ptr - gam_pen)/(gamma_minus1_d - 1.0);
                else if(gam_ptr < -gam_pen)
                    res(i) = (gam_ptr + gam_pen)/(gamma_minus1_d - 1.0);
            }
            else if(ptr[i] > total_pen)
                res(i) = (ptr[i] - total_pen)/d;
            else if(ptr[i] < -total_pen)
                res(i) = (ptr[i] + total_pen)/d;
            
        }
    }
    
    static double soft_threshold_scad_norm(double &b, const double &pen, double &d, double &gamma)
    {
        double retval = 0.0;
        
        double gammad = gamma * d;
        double gamma_minus1_d = (gamma - 1.0) * d;
        
        if (std::abs(b) > gammad * pen)
            retval = 1;
        else if (std::abs(b) > (d + 1.0) * pen)
        {
            double gam_ptr = (gamma - 1.0);
            double gam_pen = gamma * pen / b;
            if(gam_ptr > gam_pen)
                retval = d * (gam_ptr - gam_pen)/(gamma_minus1_d - 1.0);
            else if(gam_ptr < -gam_pen)
                retval = d * (gam_ptr + gam_pen)/(gamma_minus1_d - 1.0);
        }
        else if(b > pen)
            retval = (1.0 - pen / b);
        else if(b < -pen)
            retval = (1.0 + pen / b);
        return retval;
    }
    
    static double soft_threshold_mcp_norm(double &b, const double &pen, double &d, double &gamma)
    {
        double retval = 0.0;
        
        double gammad = gamma * d;
        double d_minus_gammainv = d - 1.0 / gamma;
        
        if (std::abs(b) > gammad * pen)
            retval = 1;
        else if(b > pen)
            retval = d * (1.0 - pen / b)/(d_minus_gammainv);
        else if(b < -pen)
            retval = d * (1.0 + pen / b)/(d_minus_gammainv);
        
        return retval;
    }
    
    static void block_soft_threshold_scad(VectorXd &res, const VectorXd &vec, const double &penalty,
                                          VectorXd &pen_fact, double &d,
                                          std::vector<std::vector<int> > &grp_idx, 
                                          const int &ngroups, VectorXi &unique_grps, VectorXi &grps,
                                          double & gamma)
    {
        //int v_size = vec.size();
        res.setZero();
        
        for (int g = 0; g < ngroups; ++g) 
        {
            double thresh_factor;
            std::vector<int> gr_idx = grp_idx[g];
            
            if (unique_grps(g) == 0) // the 0 group represents unpenalized variables
            {
                thresh_factor = 1.0;
            } else 
            {
                double ds_norm = 0.0;
                for (std::vector<int>::size_type v = 0; v < gr_idx.size(); ++v)
                {
                    int c_idx = gr_idx[v];
                    ds_norm += std::pow(vec(c_idx), 2);
                }
                ds_norm = std::sqrt(ds_norm);
                // double grp_wts = sqrt(gr_idx.size());
                double grp_wts = pen_fact(g);
                //thresh_factor = std::max(0.0, 1.0 - penalty * grp_wts / (ds_norm) );
                thresh_factor = soft_threshold_scad_norm(ds_norm, penalty * grp_wts, d, gamma);
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
    
    static void block_soft_threshold_mcp(VectorXd &res, const VectorXd &vec, const double &penalty,
                                         VectorXd &pen_fact, double &d,
                                         std::vector<std::vector<int> > &grp_idx, 
                                         const int &ngroups, VectorXi &unique_grps, VectorXi &grps,
                                         double & gamma)
    {
        //int v_size = vec.size();
        res.setZero();
        
        for (int g = 0; g < ngroups; ++g) 
        {
            double thresh_factor;
            std::vector<int> gr_idx = grp_idx[g];
            
            if (unique_grps(g) == 0) // the 0 group represents unpenalized variables
            {
                thresh_factor = 1.0;
            } else 
            {
                double ds_norm = 0.0;
                for (std::vector<int>::size_type v = 0; v < gr_idx.size(); ++v)
                {
                    int c_idx = gr_idx[v];
                    ds_norm += std::pow(vec(c_idx), 2);
                }
                ds_norm = std::sqrt(ds_norm);
                // double grp_wts = sqrt(gr_idx.size());
                double grp_wts = pen_fact(g);
                //thresh_factor = std::max(0.0, 1.0 - penalty * grp_wts / (ds_norm) );
                thresh_factor = soft_threshold_mcp_norm(ds_norm, penalty * grp_wts, d, gamma);
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
            
            if (unique_grps(g) == 0) // the 0 group represents unpenalized variables
            {
                thresh_factor = 1.0;
            } else 
            {
                double ds_norm = 0.0;
                for (std::vector<int>::size_type v = 0; v < gr_idx.size(); ++v)
                {
                    int c_idx = gr_idx[v];
                    ds_norm += std::pow(vec(c_idx), 2);
                }
                ds_norm = std::sqrt(ds_norm);
                // double grp_wts = sqrt(gr_idx.size());
                double grp_wts = pen_fact(g);
                thresh_factor = std::max(0.0, 1.0 - penalty * grp_wts / (ds_norm) );
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
        if (ncores <= 1)
        {
            return MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
            rankUpdate(X.adjoint());
        } else 
        {
            MatrixXd XXtmp(XXdim, XXdim);
            XXtmp.setZero();
            
            int numrowscurfirst = std::floor(double(nobs) / double(ncores) );
            
            #pragma omp parallel
            {
                MatrixXd XXtmp_private(XXdim, XXdim);
                XXtmp_private.setZero();
                
                // break up computation of X'X into 
                // X'X = X_1'X_1 + ... + X_ncores'X_ncores
                #pragma omp for schedule(static) nowait
                for (int ff = 0; ff < ncores; ++ff)
                {
                    
                    if (ff + 1 == ncores)
                    {
                        int numrowscur = nobs - (ncores - 1) * std::floor(double(nobs) / double(ncores));
                        XXtmp_private += MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
                                         rankUpdate(X.bottomRows(numrowscur).adjoint());
                    } else 
                    {
                        XXtmp_private += MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
                                         rankUpdate(X.middleRows(ff * numrowscurfirst, numrowscurfirst).adjoint());
                    }
                }
                #pragma omp critical
                {
                    XXtmp += XXtmp_private; 
                }
                
                
            }
            return XXtmp;
        }
    }
    
    MatrixXd XXt() const {
        return MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
        rankUpdate(X);
    }
    
    MatrixXd XtWX() const {
        
        if (ncores <= 1)
        {
            return MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
            rankUpdate(X.adjoint() * (weights.array().sqrt().matrix()).asDiagonal() );
        } else 
        {
            MatrixXd XXtmp(XXdim, XXdim);
            XXtmp.setZero();
            
            int numrowscurfirst = std::floor(double(nobs) / double(ncores));
            
            #pragma omp parallel
            {
                MatrixXd XXtmp_private(XXdim, XXdim);
                XXtmp_private.setZero();
                
                // break up computation of X'X into 
                // X'X = X_1'X_1 + ... + X_ncores'X_ncores
                
                #pragma omp for schedule(static) nowait
                for (int ff = 0; ff < ncores; ++ff)
                {
                    
                    if (ff + 1 == ncores)
                    {
                        int numrowscur = nobs - (ncores - 1) * std::floor(double(nobs) / double(ncores));
                        XXtmp_private += MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
                                         rankUpdate(X.bottomRows(numrowscur).adjoint() * 
                                         (weights.tail(numrowscur).array().sqrt().matrix()).asDiagonal());
                    } else 
                    {
                        XXtmp_private += MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
                                         rankUpdate(X.middleRows(ff * numrowscurfirst, numrowscurfirst).adjoint() * 
                                         (weights.segment(ff * numrowscurfirst, numrowscurfirst).array().sqrt().matrix()).asDiagonal());
                    }
                }
                #pragma omp critical
                {
                    XXtmp += XXtmp_private; 
                }
                
            }
            return XXtmp;
        }
    }
    
    MatrixXd XWXt() const {
        return MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().
        rankUpdate( (weights.array().sqrt().matrix()).asDiagonal() * X );
    }
    
    void get_group_indexes()
    {
        // if the group is any group penalty
        std::string grptxt("grp");
        
        if (penalty.find(grptxt) != std::string::npos)
        {
            found_grp_idx = true;
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
            if (nobs > nvars) 
            {
                XX = XtWX();
            } else 
            {
                XX = XWXt();
            }
        } else 
        {
            if (nobs > nvars) 
            {
                XX = XtX();
            } else 
            {
                XX = XXt();
            }
        }
        
        XX /= nobs;
        
        Spectra::DenseSymMatProd<double> op(XX);
        
        int ncv = 4;
        if (XX.cols() < 4)
        {
            ncv = XX.cols();
        }
        
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, 1, ncv);
        
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
        VectorXd beta_last = beta;
        if (penalty == "lasso")
        {
            soft_threshold(beta, u, lambda, penalty_factor, d);
        } else if (penalty == "ols")
        {
            beta = u / d;
        } else if (penalty == "elastic.net")
        {
            double denom = d + (1.0 - alpha) * lambda;
            double lam = lambda * alpha;
            
            soft_threshold(beta, u, lam, penalty_factor, denom);
        } else if (penalty == "scad") 
        {
            soft_threshold_scad(beta, u, lambda, penalty_factor, d, gamma);
            
        } else if (penalty == "scad.net") 
        {
            double denom = d + (1.0 - alpha) * lambda;
            double lam = lambda * alpha;
            
            if (alpha == 0)
            {
                lam   = 0;
                denom = d + lambda;
            }
            
            soft_threshold_scad(beta, u, lam, penalty_factor, denom, gamma);
        
        } else if (penalty == "mcp") 
        {
            soft_threshold_mcp(beta, u, lambda, penalty_factor, d, gamma);
        } else if (penalty == "mcp.net") 
        {
            double denom = d + (1.0 - alpha) * lambda;
            double lam = lambda * alpha;

            soft_threshold_mcp(beta, u, lam, penalty_factor, denom, gamma);
            
        } else if (penalty == "grp.lasso")
        {
            block_soft_threshold(beta, u, lambda, group_weights,
                                 d, grp_idx, ngroups, 
                                 unique_groups, groups);
        } else if (penalty == "grp.lasso.net")
        {
            double denom = d + (1.0 - alpha) * lambda;
            double lam = lambda * alpha;
            
            block_soft_threshold(beta, u, lam, group_weights,
                                 denom, grp_idx, ngroups, 
                                 unique_groups, groups);
            
        } else if (penalty == "grp.mcp")
        {
            block_soft_threshold_mcp(beta, u, lambda, group_weights,
                                     d, grp_idx, ngroups, 
                                     unique_groups, groups, gamma);
        } else if (penalty == "grp.scad")
        {
            block_soft_threshold_scad(beta, u, lambda, group_weights,
                                      d, grp_idx, ngroups, 
                                      unique_groups, groups, gamma);
        } else if (penalty == "grp.mcp.net")
        {
            double denom = d + (1.0 - alpha) * lambda;
            double lam = lambda * alpha;
            
            
            block_soft_threshold_mcp(beta, u, lam, group_weights,
                                     denom, grp_idx, ngroups, 
                                     unique_groups, groups, gamma);
        } else if (penalty == "grp.scad.net")
        {
            double denom = d + (1.0 - alpha) * lambda;
            double lam = lambda * alpha;
            
            block_soft_threshold_scad(beta, u, lam, group_weights,
                                      denom, grp_idx, ngroups, 
                                      unique_groups, groups, gamma);
        } else if (penalty == "sparse.grp.lasso")
        {
            double lam_grp = (1.0 - tau) * lambda;
            double lam_l1  = tau * lambda;
            
            double fact = 1.0;
            
            // first apply soft thresholding
            // but don't divide by d
            soft_threshold(beta, u, lam_l1, penalty_factor, fact);
            
            VectorXd beta_tmp = beta;
            
            // then apply block soft thresholding
            block_soft_threshold(beta, beta_tmp, lam_grp, 
                                 group_weights,
                                 d, grp_idx, ngroups, 
                                 unique_groups, groups);
        }
        
        
        
        
        if (accelerate)
        {
            ak_prev = ak;
            ak      = 0.5 * (1 + std::sqrt(1.0 + 4.0 * std::pow(ak, 2)));
            double ratio_k = (ak_prev - 1.0) / ak;
            
            VectorXd beta_update = beta;
            VectorXd beta_diff = beta - beta_last;
            beta.array() += ratio_k * beta_diff.array();
            
            double adaptive_val = ((beta.array() - beta_update.array()).array() * 
                                   beta_diff.array()).sum();
                
            if (adaptive_val > 0)
            {
                ak = 1;
                //beta = beta_last.array() * 0.90 + beta.array() * 0.1;
            }
        }
        
    }
    
    
public:
    oemDense(const Eigen::Ref<const MatrixXd>  &X_, 
             ConstGenericVector &Y_,
             const VectorXd &weights_,
             const VectorXi &groups_,
             const VectorXi &unique_groups_,
             VectorXd &group_weights_,
             VectorXd &penalty_factor_,
             bool &intercept_,
             bool &standardize_,
             int &ncores_,
             const double tol_ = 1e-6,
             const bool accelerate_ = false) :
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
                             XXdim( std::min(X_.cols(), X_.rows()) ),
                             XY(X_.cols()), // add extra space if intercept but no standardize
                             XX(XXdim, XXdim),                                // add extra space if intercept but no standardize
                             default_group_weights(bool(group_weights_.size() < 1)), // compute default weights if none given
                             ncores(ncores_),
                             grp_idx(unique_groups_.size()),
                             accelerate(accelerate_)
    
    {}
    
    void init_oem()
    {
        found_grp_idx = false;
        
        wt_len = weights.size();
        
        if (wt_len)
        {
            XY.noalias() = X.transpose() * (Y.array() * weights.array()).matrix();
        } else
        {
            XY.noalias() = X.transpose() * Y;
        }
        
        XY /= nobs;
        
        // compute XtX or XXt (depending on if n > p or not)
        // and compute A = dI - XtX (if n > p)
        compute_XtX_d_update_A();
    }
    
    double compute_lambda_zero() 
    { 
        lambda0 = XY.cwiseAbs().maxCoeff();
        return lambda0; 
    }
    double get_d() { return d; }
    
    // init() is a cold start for the first lambda.
    // init() called before each penalty 
    void init(double lambda_, std::string penalty_,
              double alpha_, double gamma_, double tau_)
    {
        beta.setZero();
        
        lambda = lambda_;
        penalty = penalty_;
        
        alpha = alpha_;
        gamma = gamma_;
        tau   = tau_;
        
        
        
        // get indexes of members of each group.
        // best to do just once in the beginning
        if (!found_grp_idx)
        {
            get_group_indexes();
        }
        
        ak = 1;
        
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



#endif // OEM_DENSE_TALL_H
