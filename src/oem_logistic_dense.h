#ifndef OEM_LOGISTIC_DENSE_H
#define OEM_LOGISTIC_DENSE_H

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




// minimize  1/2 * ||y - X * beta||^2 + P_\lambda(beta)
//
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
    VectorXd prob;              // 1 / (1 + exp(-x * beta))
    VectorXd grad;
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
    std::string hessian_type;
    int irls_maxit;
    double irls_tol;
    double dev, dev0;
    Eigen::RowVectorXd colsums;
    Eigen::RowVectorXd colsq;
    Eigen::VectorXd colsq_inv;
    
    std::vector<std::vector<int> > grp_idx; // vector of vectors of the indexes for all members of each group
    std::string penalty;        // penalty specified
    
    double lambda;              // L1 penalty
    double lambda0;             // minimum lambda to make coefficients all zero
    double alpha;               // alpha = mixing parameter for elastic net
    double gamma;               // extra tuning parameter for mcp/scad
    double tau;                 // mixing parameter for group sparse penalties
    
    double threshval;
    int wt_len;
    bool on_lam_1;
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
        double gamma_minus1_d = (gamma - 1) * d;
        
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
            retval = 1.0;
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
            retval = 1.0;
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
            
            if (unique_grps(g) == 0) 
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
    
    /*
    MatrixXd XtWX() const {
        return MatrixXd(nvars, nvars).setZero().selfadjointView<Lower>().
        rankUpdate(X.adjoint() * (W.array().sqrt().matrix()).asDiagonal() );
    }*/
    
    MatrixXd XtWX() const {
        
        if (ncores <= 1)
        {
            return MatrixXd(nvars, nvars).setZero().selfadjointView<Lower>().
            rankUpdate(X.adjoint() * (W.array().sqrt().matrix()).asDiagonal() );
        } else 
        {
            MatrixXd XXtmp(nvars, nvars);
            XXtmp.setZero();
            
            int numrowscurfirst = std::floor(double(nobs) / double(ncores));
            
            #pragma omp parallel
            {
                MatrixXd XXtmp_private(nvars, nvars);
                XXtmp_private.setZero();
                
                // break up computation of X'X into 
                // X'X = X_1'X_1 + ... + X_ncores'X_ncores
                
                #pragma omp for schedule(static) nowait
                for (int ff = 0; ff < ncores; ++ff)
                {
                    
                    if (ff + 1 == ncores)
                    {
                        int numrowscur = nobs - (ncores - 1) * std::floor(double(nobs) / double(ncores));
                        
                        XXtmp_private += MatrixXd(nvars, nvars).setZero().selfadjointView<Lower>().
                        rankUpdate(X.bottomRows(numrowscur).adjoint() * 
                        (W.tail(numrowscur).array().sqrt().matrix()).asDiagonal());
                    } else 
                    {
                        XXtmp_private += MatrixXd(nvars, nvars).setZero().selfadjointView<Lower>().
                        rankUpdate(X.middleRows(ff * numrowscurfirst, numrowscurfirst).adjoint() * 
                        (W.segment(ff * numrowscurfirst, numrowscurfirst).array().sqrt().matrix()).asDiagonal());
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
        //if (standardize)
        //{
        //    return MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().
        //    rankUpdate( (W.array().sqrt().matrix()).asDiagonal() * X * colsq_inv.asDiagonal() );
        //} else 
        //{
        return MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().
        rankUpdate( (W.array().sqrt().matrix()).asDiagonal() * X );
        //}
    }
    
    // function to be called once in the beginning
    // to get the locations of all members of each group
    void get_group_indexes()
    {
        // if the group is group lasso
        std::string grptxt("grp");
        if (penalty.find(grptxt) != std::string::npos) 
        {
            found_grp_idx = true;
            grp_idx.reserve(ngroups);
            for (int g = 0; g < ngroups; ++g) 
            {
                // find all variables in group number g
                std::vector<int> idx_tmp;
                for (int v = 0; v < nvars + int(intercept); ++v) 
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
                    if (unique_groups(g) == 0)
                    {
                        // don't apply group lasso
                        // penalty for group 0
                        group_weights(g) = 0;
                    } else {
                        group_weights(g) = std::sqrt(double(grp_idx[g].size()));
                    }
                }
            }
        }
    }
    
    // deviance residuals for logistic glm
    double sum_dev_resid(MapVec &y, VectorXd &prob)
    {
        double dev = 0.0;
        for (int ii = 0; ii < nobs; ++ii)
        {
            if (y(ii) == 1)
            {
                dev += std::sqrt(2.0 * std::log(1.0/prob(ii)));
            } else 
            {
                dev += std::sqrt(2.0 * std::log(1.0/(1.0 - prob(ii))));
            }
        }
        return dev;
    }
    
    void compute_XtX_d_update_A()
    {
        
        // compute X'X or XX'
        // must handle differently if p > n
        if (nobs > nvars + int(intercept)) 
        {
            if (intercept)
            {
                colsums.noalias() = (((W.array().matrix()).asDiagonal() * X).colwise().sum()).matrix(); 
                
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
                XX(0,0) = W.array().sum();
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
                XX.array() += 1.0; // adding 1 to all of XX' for the intercept
        }
        
        // scale by sample size. needed for SCAD/MCP
        XX /= nobs;
        
        Spectra::DenseSymMatProd<double> op(XX);
        
        int ncv = 4;
        if (XX.cols() < 4)
        {
            ncv = XX.cols();
        }
        
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, 1, ncv);
        
        eigs.init();
        eigs.compute(1000, 1e-6);
        Vector eigenvals = eigs.eigenvalues();
        d = eigenvals[0] * 1.0005; // multiply by an increasing factor to be safe
        
        if (nobs > nvars + int(intercept))
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
            if (intercept)
            {
                if (standardize)
                {
                    // need to handle differently with intercept
                    VectorXd resid  = Y - X * (beta_prev.tail(nvars).array() * colsq_inv.array()).matrix();
                    resid.array() -= beta_prev(0);
                    //resid.array() *= W.array();
                    
                    resid /=  double(nobs);
                    res.tail(nvars) = (colsq_inv.asDiagonal() * X.adjoint()) * (resid) + d * beta_prev.tail(nvars);
                    res(0) = resid.sum() + d * beta_prev(0);
                } else 
                {
                    // need to handle differently with intercept
                    VectorXd resid  = Y - X * beta_prev.tail(nvars).matrix();
                    resid.array() -= beta_prev(0);
                    //resid.array() *= W.array();
                    
                    resid /=  double(nobs);
                    res.tail(nvars) = X.adjoint() * (resid) + d * beta_prev.tail(nvars);
                    res(0) = resid.sum() + d * beta_prev(0);
                }
            } else 
            {
                if (standardize)
                {
                    res.noalias() = (colsq_inv.array() * (X.adjoint() * ((W.array() * (Y - X * ( beta_prev.array() * colsq_inv.array() ).matrix() ).array() ) / double(nobs)).matrix()).array()).matrix();
                } else 
                {
                    res.noalias() = X.adjoint() * ((W.array() * (Y - X * beta_prev).array() ) / double(nobs)).matrix();
                }
                res += d * beta_prev;
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
        
        
    }
    
    
public:
    oemLogisticDense(const Eigen::Ref<const MatrixXd>  &X_, 
                     ConstGenericVector &Y_,
                     const VectorXd &weights_,
                     const VectorXi &groups_,
                     const VectorXi &unique_groups_,
                     VectorXd &group_weights_,
                     VectorXd &penalty_factor_,
                     bool &intercept_,
                     bool &standardize_,
                     int &ncores_,
                     std::string &hessian_type_,
                     const int &irls_maxit_ = 100,
                     const double &irls_tol_ = 1e-6,
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
                             prob(X_.rows()),
                             grad(X_.cols() + int(intercept_)),
                             weights(weights_),
                             groups(groups_),
                             unique_groups(unique_groups_),
                             penalty_factor(penalty_factor_),
                             group_weights(group_weights_),
                             penalty_factor_size(penalty_factor_.size()),
                             XXdim( std::min(X_.cols() + int(intercept_) , X_.rows())),
                             XY(X_.cols() + int(intercept)), // add extra space if intercept but no standardize
                             XX(XXdim, XXdim),                                // add extra space if intercept but no standardize
                             default_group_weights( bool(group_weights_.size() < 1) ),  // compute default weights if none given
                             ncores(ncores_),
                             hessian_type(hessian_type_),
                             irls_maxit(irls_maxit_),
                             irls_tol(irls_tol_),
                             colsums(X_.cols()),
                             colsq(X_.cols()),
                             colsq_inv(X_.cols()),
                             grp_idx(unique_groups_.size())
    {}
    
    void init_oem()
    {
        wt_len = weights.size();
        
        found_grp_idx = false;
        
        if (standardize)
        {
            //if (wt_len)
            //{
            //    colsq = ((weights.array().matrix()).asDiagonal() * X).array().square().matrix().colwise().sum() / (nobs - 1);
            //} else 
            //{
            colsq = X.array().square().matrix().colwise().sum() / (double(nobs) - 1.0);
            //}
            colsq_inv = 1.0 / colsq.array().sqrt();
        }
        
        //if (wt_len)
        //{
        //    ((weights.array().matrix()).asDiagonal() * X).colwise().sum();
        //} else 
        //{
        colsums = X.colwise().sum();
        //}
        
        if (intercept)
        {
            // these need to be one element
            // larger for model with intercept
            u.resize(nvars + 1);
            beta.resize(nvars + 1);
            beta_prev.resize(nvars + 1);
            
            //if (wt_len)
            //{
            //    XY.tail(nvars) = X.transpose() * (Y.array() * weights.array()).matrix();
            //    XY(0) = (Y.array() * weights.array()).sum();
            //} else 
            //{
            XY.tail(nvars) = X.transpose() * Y;
            XY(0) = Y.sum();
            //}
            
            
            if (standardize)
            {
                XY.tail(nvars).array() *= colsq_inv.array();
                colsums.array() *= colsq_inv.array();
            } 
            
            
        } else 
        {
            //if (wt_len)
            //{
            //    XY.noalias() = X.transpose() * (Y.array() * weights.array()).matrix();
            //} else
            //{
            XY.noalias() = X.transpose() * Y;
            //}
            
            
            if (standardize)
            {
                XY.array() *= colsq_inv.array();
                colsums.array() *= colsq_inv.array();
            } 
        }
        
        XY /= nobs;
    }
    
    double compute_lambda_zero() 
    { 
        
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
    void init(double lambda_, std::string penalty_,
              double alpha_, double gamma_, double tau_)
    {
        beta.setZero();
        
        on_lam_1 = true;
        if (intercept)
        {
            //double ymean = Y.mean();
            //beta(0) = std::log(ymean / (1 - ymean));
        }
        
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
        
    }
    
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    void init_warm(double lambda_)
    {
        on_lam_1 = false;
        lambda = lambda_;
    }
    
    // re-define solve to do IRLS
    // iterations
    virtual int solve(int maxit)
    {
        
        dev = 1e30;
        
        int i;
        int j;
        for (i = 0; i < irls_maxit; ++i)
        {
            
            dev0 = dev;
            beta_prev_irls = beta;
            
            if (!(i == 0 && !on_lam_1))
            {
                
                if (ncores <= 1)
                {
                    // calculate mu hat
                    if (intercept)
                    {
                        if (standardize)
                        {
                            prob = 1 / (1 + (-1 * ((X * ( beta.tail(nvars).array() * colsq_inv.array() ).matrix()   ).array() + 
                                beta(0)).array()).exp().array());
                        } else 
                        {
                            prob = 1 / (1 + (-1 * ((X * beta.tail(nvars)).array() + beta(0)).array()).exp().array());
                        }
                    } else
                    {
                        if (standardize){
                            prob.noalias() = (1 / (1 + (-1 * (X * ( beta.array() * colsq_inv.array() ).matrix()  ).array()).exp().array())).matrix();
                        } else 
                        {
                            prob.noalias() = (1 / (1 + (-1 * (X * beta).array()).exp().array())).matrix();
                        }
                    }
                } else 
                {
                    int numrowscur = std::floor(double(nobs) / double(ncores));
                    int numrowscurfirst = numrowscur;
                    
                    #pragma omp parallel for schedule(static)
                    for (int ff = 0; ff < ncores; ++ff)
                    {
                        
                        if (ff + 1 == ncores)
                        {
                            numrowscur = nobs - (ncores - 1) * std::floor(double(nobs) / double(ncores));
                            if (intercept)
                            {
                                if (standardize)
                                {
                                    prob.tail(numrowscur) = 1 / (1 + (-1 * ((X.bottomRows(numrowscur) * (beta.tail(nvars).array() * colsq_inv.array()).matrix()    ).array() + beta(0)).array()).exp().array());
                                } else 
                                {
                                    prob.tail(numrowscur) = 1 / (1 + (-1 * ((X.bottomRows(numrowscur) * beta.tail(nvars)).array() + beta(0)).array()).exp().array());
                                }
                            } else
                            {
                                if (standardize)
                                {
                                    prob.tail(numrowscur) = (1 / (1 + (-1 * (X.bottomRows(numrowscur) * (beta.array() * colsq_inv.array()).matrix()   ).array()).exp().array())).matrix();
                                } else 
                                {
                                    prob.tail(numrowscur) = (1 / (1 + (-1 * (X.bottomRows(numrowscur) * beta).array()).exp().array())).matrix();
                                }
                            }
                        } else 
                        {
                            if (intercept)
                            {
                                if (standardize)
                                {
                                    prob.segment(ff * numrowscurfirst, numrowscurfirst) = 1 / (1 + (-1 * ((X.middleRows(ff * numrowscurfirst, numrowscurfirst) * (beta.tail(nvars).array() * colsq_inv.array()).matrix()   ).array() + beta(0)).array()).exp().array());
                                } else 
                                {
                                    prob.segment(ff * numrowscurfirst, numrowscurfirst) = 1 / (1 + (-1 * ((X.middleRows(ff * numrowscurfirst, numrowscurfirst) * beta.tail(nvars)).array() + beta(0)).array()).exp().array());
                                }
                            } else
                            {
                                if (standardize)
                                {
                                    prob.segment(ff * numrowscurfirst, numrowscurfirst) = (1 / (1 + (-1 * (X.middleRows(ff * numrowscurfirst, numrowscurfirst) * (beta.array() * colsq_inv.array()).matrix()).array()).exp().array())).matrix();
                                } else 
                                {
                                    prob.segment(ff * numrowscurfirst, numrowscurfirst) = (1 / (1 + (-1 * (X.middleRows(ff * numrowscurfirst, numrowscurfirst) * beta).array()).exp().array())).matrix();
                                }
                            }
                        }
                    }
                } 
                
                // calculate Jacobian (or weight vector)
                W = prob.array() * (1 - prob.array());
                
                // if observation weights specified, use them
                if (wt_len)
                {
                    W.array() *= weights.array();
                }
                
                
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
                if ((i == 0 && on_lam_1) || hessian_type == "full")
                    compute_XtX_d_update_A();
                
                
                // compute X'Wz
                // only for p < n case
                if (nobs > nvars + int(intercept))
                {
                    
                    if (intercept)
                    {
                        VectorXd presid = Y.array() - prob.array();
                        grad.tail(nvars) = (X.adjoint() * presid).array() / double(nobs);
                        grad(0) = presid.sum() / double(nobs);
                        
                        if (standardize)
                        {
                            grad.tail(nvars).array() *= colsq_inv.array();
                        }
                        
                    } else 
                    {
                        grad.noalias() = X.adjoint() * (Y.array() - prob.array() ).matrix() / double(nobs);
                        
                        if (standardize)
                        {
                            grad.array() *= colsq_inv.array();
                        }
                    }
                    
                    
                    // not sure why the following doesn't 
                    // work but the above, which seems
                    // wrong does work
                    //grad = X.adjoint() * ( W.array() * (Y.array() - prob.array()).array()).matrix();
                    XY.noalias() = XX * beta + grad;
                }
            } //else 
            //{
            //    W.fill(0.25);
            //    if (wt_len) W.array() *= weights.array();
            //    XY.array() -= 0.5 / nobs;
            //    prob.fill(0.5);
            //}
            
            // oem iterations
            for(j = 0; j < maxit; ++j)
            {
                
                beta_prev = beta;
                
                update_u();
                
                update_beta();
                
                if(converged())
                    break;
                
            }
            
            // update deviance residual
            dev = sum_dev_resid(Y, prob);
            
            //if (std::abs(dev - dev0) / (0.1 + std::abs(dev) ) < irls_tol)
            if (stopRule(beta, beta_prev_irls, irls_tol))
            {
                break;
            }
            
        }
        
        return i + 1;
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
        double loss = 0.0;
        
        // compute logistic loss
        for (int ii = 0; ii < nobs; ++ii)
        {
            if (Y(ii) == 1)
            {
                if (prob(ii) > 1e-5)
                {
                    loss += std::log(1 / prob(ii));
                } else
                {
                    // don't divide by zero
                    loss += std::log(1 / 1e-5);
                }
                
            } else
            {
                if (prob(ii) <= 1 - 1e-5)
                {
                    loss += std::log(1 / (1 - prob(ii)));
                } else
                {
                    // don't divide by zero
                    loss += std::log(1 / 1e-5);
                }
                
            }
        }
        return loss;
    }
    
};


#endif // OEM_LOGISTIC_DENSE_H
