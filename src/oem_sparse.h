#ifndef OEM_SPARSE_H
#define OEM_SPARSE_H

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
    typedef MSpMat::InnerIterator InIterMat;
    
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
    bool default_group_weights; // do we need to compute default group weights?
    int ncores;
    double xxdiag;
    double intval;
    
    
    std::vector<std::vector<int> > grp_idx; // vector of vectors of the indexes for all members of each group
    std::string penalty;        // penalty specified
    
    double lambda;              // L1 penalty
    double lambda0;             // minimum lambda to make coefficients all zero
    double alpha;               // alpha = mixing parameter for elastic net
    double gamma;               // extra tuning parameter for mcp/scad
    double tau;                 // mixing parameter for group sparse penalties
    
    double threshval;
    int wt_len;
    
    VectorXd colsq_inv;
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
        double retval = 0;
        
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
    
    
    
    SpMat XtX() const {
        return SpMat(XXdim, XXdim).selfadjointView<Upper>().
        rankUpdate(X.adjoint());
    }
    
    /*
    SpMat XtX() const {
        
        if (ncores <= 1)
        {
            return SpMat(XXdim, XXdim).selfadjointView<Lower>().
            rankUpdate(X.adjoint()  );
        } else 
        {
            SpMat XXtmp(XXdim, XXdim);
            
            int numrowscurfirst = floor(nobs / ncores);
            
            #pragma omp parallel
            {
                SpMat XXtmp_private(XXdim, XXdim);
                
                // break up computation of X'X into 
                // X'X = X_1'X_1 + ... + X_ncores'X_ncores
                
                #pragma omp for schedule(static) nowait
                for (int ff = 0; ff < ncores; ++ff)
                {
                    
                    if (ff + 1 == ncores)
                    {
                        int numrowscur = nobs - (ncores - 1) * floor(nobs / ncores);
                        
                        XXtmp_private += SpMat(XXdim, XXdim).selfadjointView<Upper>().rankUpdate( X.bottomRows(numrowscur).transpose() );
                    } else 
                    {
                        XXtmp_private += SpMat(XXdim, XXdim).selfadjointView<Upper>().rankUpdate( X.middleRows(ff * numrowscurfirst, numrowscurfirst).transpose() );
                    }
                }
                #pragma omp critical
                {
                    XXtmp += XXtmp_private; 
                }
                
            }
            return XXtmp;
        }
    }*/
    
    SpMat XXt() const {
        return SpMat(XXdim, XXdim).selfadjointView<Upper>().
        rankUpdate(X);
    }
    
    
    SpMat XtWX() const {
        return SpMat(nvars, nvars).selfadjointView<Upper>().
        rankUpdate(X.adjoint() * (weights.array().sqrt().matrix()).asDiagonal() );
    }
    
    /*
    SpMat XtWX() const {
        
        if (ncores <= 1)
        {
            return SpMat(XXdim, XXdim).selfadjointView<Upper>().
            rankUpdate(X.adjoint() * (weights.array().sqrt().matrix()).asDiagonal() );
        } else 
        {
            SpMat XXtmp(XXdim, XXdim);
            
            int numrowscurfirst = floor(nobs / ncores);
            
        #pragma omp parallel
        {
            SpMat XXtmp_private(XXdim, XXdim);
            
            // break up computation of X'X into 
            // X'X = X_1'X_1 + ... + X_ncores'X_ncores
            
            #pragma omp for schedule(static) nowait
            for (int ff = 0; ff < ncores; ++ff)
            {
                
                if (ff + 1 == ncores)
                {
                    int numrowscur = nobs - (ncores - 1) * floor(nobs / ncores);
                    XXtmp_private += SpMat(XXdim, XXdim).selfadjointView<Upper>().
                    rankUpdate(X.bottomRows(numrowscur).adjoint() * 
                    (weights.tail(numrowscur).array().sqrt().matrix()).asDiagonal());
                } else 
                {
                    XXtmp_private += SpMat(XXdim, XXdim).selfadjointView<Upper>().
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
    }*/
    
    SpMat XWXt() const {
        return SpMat(nobs, nobs).selfadjointView<Upper>().
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
                    group_weights(g) = std::sqrt(double(grp_idx[g].size()));
                }
            }
        }
    }
    
    void compute_XtX_d_update_A()
    {
        
        if (standardize)
        {
            VectorXd colsq(nvars);
            colsq.setZero();
            
            for (int j = 0; j < nvars; ++j)
            {
                for (InIterMat i_(X, j); i_; ++i_)
                {
                    colsq(j) += std::pow(i_.value(), 2);
                }
            }
            colsq /= (double(nobs) - 1.0);
            colsq_inv = 1.0 / colsq.array().sqrt();
        }
        
        // compute X'X
        // if weights specified, compute X'WX instead
        if (wt_len)
        {
            if (nobs > nvars) 
            {
                if (intercept)
                {
                    // compute X'X with intercept
                    if (standardize)
                    {
                        XX.bottomRightCorner(nvars, nvars) = colsq_inv.asDiagonal() * XtWX() * colsq_inv.asDiagonal();
                    } else
                    {
                        XX.bottomRightCorner(nvars, nvars) = XtWX();
                    }
                        
                    xxdiag = XX.diagonal().tail(nvars).mean();
                    intval = std::sqrt(xxdiag / double(nobs));
                    
                    Eigen::RowVectorXd colsums = X.adjoint() * weights; 
                    colsums.array() *= intval;
                    
                    if (standardize)
                    {
                        XX.block(0,1,1,nvars) = colsums.array() * colsq_inv.array();
                        XX.block(1,0,nvars,1) = (colsums.array() * colsq_inv.array()).transpose();
                    } else 
                    {
                        XX.block(0,1,1,nvars) = colsums;
                        XX.block(1,0,nvars,1) = colsums.transpose();
                    }
                    XX(0,0) = weights.sum() * xxdiag;
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
                    if (standardize)
                    {
                        XX.bottomRightCorner(nvars, nvars) = colsq_inv.asDiagonal() * XtX() * colsq_inv.asDiagonal();
                    } else
                    {
                        XX.bottomRightCorner(nvars, nvars) = XtX();
                    }
                    
                    xxdiag = XX.diagonal().tail(nvars).mean();
                    intval = std::sqrt(xxdiag / nobs);
                    
                    Eigen::RowVectorXd colsums = X.adjoint() * VectorXd::Ones( nobs );
                    colsums.array() *= intval;
                    
                    
                    XX.block(0,1,1,nvars) = colsums;
                    XX.block(1,0,nvars,1) = colsums.transpose();
                    
                    if (standardize)
                    {
                        XX.row(0).tail(nvars).array() *= colsq_inv.array();
                        XX.col(0).tail(nvars).array() *= colsq_inv.array();
                    }
                    
                    XX(0,0) = xxdiag;
                    
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
                {
                    XX.array() += 1.0;
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
            double denom = d + (1.0 - alpha) * lambda / alpha;
            double lam = lambda;
            soft_threshold(beta, u, lam, penalty_factor, denom);
        } else if (penalty == "scad") 
        {
            soft_threshold_scad(beta, u, lambda, penalty_factor, d, gamma);
            
        } else if (penalty == "scad.net") 
        {
            double denom = d + (1.0 - alpha) * lambda / alpha;
            double lam = lambda;
            soft_threshold_scad(beta, u, lam, penalty_factor, denom, gamma);
            
        } else if (penalty == "mcp") 
        {
            soft_threshold_mcp(beta, u, lambda, penalty_factor, d, gamma);
        } else if (penalty == "mcp.net") 
        {
            double denom = d + (1.0 - alpha) * lambda / alpha;
            double lam = lambda;
            soft_threshold_mcp(beta, u, lam, penalty_factor, denom, gamma);
            
        } else if (penalty == "grp.lasso")
        {
            block_soft_threshold(beta, u, lambda, group_weights,
                                 d, grp_idx, ngroups, 
                                 unique_groups, groups);
        } else if (penalty == "grp.lasso.net")
        {
            double denom = d + (1.0 - alpha) * lambda / alpha;
            double lam = lambda;
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
            double denom = d + (1.0 - alpha) * lambda / alpha;
            double lam = lambda;
            block_soft_threshold_mcp(beta, u, lam, group_weights,
                                     denom, grp_idx, ngroups, 
                                     unique_groups, groups, gamma);
        } else if (penalty == "grp.scad.net")
        {
            double denom = d + (1.0 - alpha) * lambda / alpha;
            double lam = lambda;
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
    oemSparse(const MSpMat &X_, 
              ConstGenericVector &Y_,
              const VectorXd &weights_,
              const VectorXi &groups_,
              const VectorXi &unique_groups_,
              VectorXd &group_weights_,
              VectorXd &penalty_factor_,
              bool &intercept_,
              bool &standardize_,
              int &ncores_,
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
                             default_group_weights(bool(group_weights_.size() < 1)), // compute default weights if none given
                             ncores(ncores_),
                             grp_idx(unique_groups_.size()),
                             colsq_inv(X_.cols())
    
    {}
    
    void init_oem()
    {
        if (intercept)
        {
            u.resize(nvars + 1);
            beta.resize(nvars + 1);
            beta_prev.resize(nvars + 1);
        }
        
        found_grp_idx = false;
        
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
                if (standardize) XY.tail(nvars).array() *= colsq_inv.array();
            } else 
            {
                XY.noalias() = X.transpose() * (Y.array() * weights.array()).matrix();
                if (standardize) XY.array() *= colsq_inv.array();
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
                if (standardize) XY.tail(nvars).array() *= colsq_inv.array();
            } else 
            {
                XY.noalias() = X.transpose() * Y;
                if (standardize) XY.array() *= colsq_inv.array();
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
        lambda = lambda_;
        
    }
    
    VectorXd get_beta() 
    { 
        if (intercept && nobs > nvars)
        {
            beta(0) *= (intval);
        }
        if (standardize)
        {
            if (intercept)
            {
                VectorXd beta_tmp = beta;
                beta_tmp.tail(nvars).array() *= colsq_inv.array();
                return beta_tmp;
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
        if (intercept)
        {
            if (standardize)
            {
                loss = ((Y - X * (beta.tail(nvars).array() * colsq_inv.array()).matrix()  ).array() - beta(0)).array().square().sum();
            } else 
            {
                loss = ((Y - X * beta.tail(nvars)).array() - beta(0)).array().square().sum();
            }
        } else 
        {
            if (standardize)
            {
                loss = (Y - X * (beta.array() * colsq_inv.array()).matrix() ).array().square().sum();
            } else 
            {
                loss = (Y - X * beta).array().square().sum();
            }
        }
        return loss;
    }
};



#endif // OEM_SPARSE_H
