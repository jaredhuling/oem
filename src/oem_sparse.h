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
    typedef Eigen::Map<Eigen::SparseMatrix<double>> MSpMat;
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
    double intval;              // scale c of the intercept column
    
    
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
    VectorXd colmeans_x;        // column means of X, for implicit centering
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
    
    static double largest_eigenvalue(const MatrixXd &M)
    {
        Spectra::DenseSymMatProd<double> op(M);
        int ncv = 4;
        if (M.cols() < 4)
        {
            ncv = M.cols();
        }
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, 1, ncv);
        eigs.init();
        eigs.compute(10000, 1e-10);
        return eigs.eigenvalues()[0];
    }
    
    void compute_XtX_d_update_A()
    {
        // The solver works with Z = [c 1, (X - 1 m') S^{-1}] when there is an
        // intercept and with Z = X S^{-1} otherwise. S holds the column scales:
        // the standard deviations (denominator n) when standardizing, as in
        // glmnet, and 1 otherwise. Centering by the column means m (taken
        // implicitly, so X stays sparse) makes the intercept column orthogonal
        // to the others, which greatly improves the conditioning of Z'Z. The
        // intercept column is scaled by c so that its curvature equals the
        // largest eigenvalue of the rest, so it converges in about one step.
        const double wsum = wt_len ? weights.sum() : double(nobs);
        colsq_inv.setOnes();
        colmeans_x.setZero();
        if (standardize || intercept)
        {
            for (int j = 0; j < nvars; ++j)
            {
                double s1 = 0.0, s2 = 0.0;
                for (InIterMat i_(X, j); i_; ++i_)
                {
                    double wi = wt_len ? weights(i_.row()) : 1.0;
                    s1 += wi * i_.value();
                    s2 += wi * i_.value() * i_.value();
                }
                double mean = s1 / wsum;
                if (intercept) colmeans_x(j) = mean;
                double var = s2 / wsum - mean * mean;
                if (standardize && var > 0.0) colsq_inv(j) = 1.0 / std::sqrt(var);
            }
        }
        const VectorXd v = colmeans_x.cwiseProduct(colsq_inv);   // S^{-1} m
        SpMat Xs = X * colsq_inv.asDiagonal();
        double drest;
        
        if (nobs > nvars)
        {
            MatrixXd G;
            VectorXd cw;
            if (wt_len)
            {
                G = MatrixXd(SpMat(Xs.adjoint() * weights.asDiagonal() * Xs));
                cw = Xs.adjoint() * weights;
            } else
            {
                G = MatrixXd(SpMat(Xs.adjoint() * Xs));
                cw = Xs.adjoint() * VectorXd::Ones(nobs);
            }
            if (intercept)
            {
                // centered: (X - 1 m')' W (X - 1 m'), scaled
                G -= v * cw.transpose() + cw * v.transpose();
                G += wsum * v * v.transpose();
            }
            G /= double(nobs);
            drest = largest_eigenvalue(G);
            if (!(drest > 0.0)) drest = wsum / double(nobs);
            if (intercept)
            {
                intval = std::sqrt(drest / (wsum / double(nobs)));
                XX.setZero();
                XX(0, 0) = drest;
                XX.bottomRightCorner(nvars, nvars) = G;
            } else
            {
                XX = G;
            }
            d = drest * 1.005; // multiply by an increasing factor to be safe
            A = -XX;
            A.diagonal().array() += d;
        } else 
        {
            // Z W Z' / n has the same nonzero eigenvalues as Z'WZ / n and is
            // n x n; it is used only to find d.
            VectorXd sw = wt_len ? VectorXd(weights.array().sqrt().matrix()) : VectorXd(VectorXd::Ones(nobs));
            SpMat B = sw.asDiagonal() * Xs;
            MatrixXd M = MatrixXd(SpMat(B * B.adjoint()));
            if (intercept)
            {
                VectorXd Bv = B * v;
                M -= Bv * sw.transpose() + sw * Bv.transpose();
                M += v.squaredNorm() * sw * sw.transpose();
            }
            M /= double(nobs);
            drest = largest_eigenvalue(M);
            if (!(drest > 0.0)) drest = wsum / double(nobs);
            if (intercept) intval = std::sqrt(drest / (wsum / double(nobs)));
            d = drest * 1.005;
        }
    }
    
    // linear predictor for coefficients theta in the solver's coordinates
    VectorXd linear_predictor(const VectorXd &theta) const
    {
        if (intercept)
        {
            VectorXd b = theta.tail(nvars).cwiseProduct(colsq_inv);
            VectorXd eta = X * b;
            eta.array() += intval * theta(0) - colmeans_x.dot(b);
            return eta;
        }
        return X * theta.cwiseProduct(colsq_inv);
    }
    
    // Hooks used by the accelerated iterations in oemBase::inner_solve().
    // Within one inner solve next_u() is affine in beta_prev, so the default
    // linear_term() (one extra evaluation of next_u) is valid.
    bool penalty_is_convex()
    {
        return !(penalty.find("mcp") != std::string::npos ||
                 penalty.find("scad") != std::string::npos);
    }
    
    double penalty_value(const VectorXd &b)
    {
        if (penalty.find("grp") != std::string::npos && !found_grp_idx)
            return std::numeric_limits<double>::quiet_NaN();
        return penalty_value_generic(b, penalty, lambda, alpha, gamma, tau,
                                     penalty_factor, group_weights, grp_idx, unique_groups, ridge_scale);
    }
    
    void next_u(Vector &res)
    {
        if (nobs > nvars)
        {
            res.noalias() = A * beta_prev + XY;
        } else 
        {
            // u = d * theta + Z' W (y - Z theta) / n, without forming Z'Z
            res.resize(beta_prev.size());
            VectorXd resid = Y - linear_predictor(beta_prev);
            if (wt_len) resid.array() *= weights.array();
            resid /= double(nobs);
            if (intercept)
            {
                const double rsum = resid.sum();
                res.tail(nvars) = (X.adjoint() * resid - colmeans_x * rsum).cwiseProduct(colsq_inv) + d * beta_prev.tail(nvars);
                res(0) = intval * rsum + d * beta_prev(0);
            } else
            {
                res = (X.adjoint() * resid).cwiseProduct(colsq_inv) + d * beta_prev;
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
            // L1 level lambda * alpha * pf_j, ridge (1 - alpha) * lambda * pf_j
            oem_net_threshold(beta, u, 0, lambda * alpha, (1.0 - alpha) * lambda * ridge_scale, penalty_factor, d, gamma);
        } else if (penalty == "scad") 
        {
            soft_threshold_scad(beta, u, lambda, penalty_factor, d, gamma);
            
        } else if (penalty == "scad.net") 
        {
            // L1 level lambda * alpha * pf_j, ridge (1 - alpha) * lambda * pf_j
            oem_net_threshold(beta, u, 2, lambda * alpha, (1.0 - alpha) * lambda * ridge_scale, penalty_factor, d, gamma);
        } else if (penalty == "mcp") 
        {
            soft_threshold_mcp(beta, u, lambda, penalty_factor, d, gamma);
        } else if (penalty == "mcp.net") 
        {
            // L1 level lambda * alpha * pf_j, ridge (1 - alpha) * lambda * pf_j
            oem_net_threshold(beta, u, 1, lambda * alpha, (1.0 - alpha) * lambda * ridge_scale, penalty_factor, d, gamma);
        } else if (penalty == "grp.lasso")
        {
            block_soft_threshold(beta, u, lambda, group_weights,
                                 d, grp_idx, ngroups, 
                                 unique_groups, groups);
        } else if (penalty == "grp.lasso.net")
        {
            double denom = d + (1.0 - alpha) * lambda * ridge_scale;
            double lam = lambda * alpha;
            
            block_soft_threshold(beta, u, lam, group_weights,
                                 denom, grp_idx, ngroups, 
                                 unique_groups, groups);
            // unpenalized coefficients take no ridge term
            oem_unpenalized_group_update(beta, u, grp_idx, unique_groups, d);
            
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
            double denom = d + (1.0 - alpha) * lambda * ridge_scale;
            double lam = lambda * alpha;
            
            
            block_soft_threshold_mcp(beta, u, lam, group_weights,
                                     denom, grp_idx, ngroups, 
                                     unique_groups, groups, gamma);
            // unpenalized coefficients take no ridge term
            oem_unpenalized_group_update(beta, u, grp_idx, unique_groups, d);
        } else if (penalty == "grp.scad.net")
        {
            double denom = d + (1.0 - alpha) * lambda * ridge_scale;
            double lam = lambda * alpha;
            
            block_soft_threshold_scad(beta, u, lam, group_weights,
                                      denom, grp_idx, ngroups, 
                                      unique_groups, groups, gamma);
            // unpenalized coefficients take no ridge term
            oem_unpenalized_group_update(beta, u, grp_idx, unique_groups, d);
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
                             XXdim( X_.rows() > X_.cols() ? X_.cols() + int(intercept_) : X_.rows() ),
                             XY(X_.cols() + int(intercept_)),
                             XX(XXdim, XXdim),     // add extra space if intercept and n > p
                             default_group_weights(bool(group_weights_.size() < 1)), // compute default weights if none given
                             ncores(ncores_),
                             grp_idx(unique_groups_.size()),
                             colsq_inv(X_.cols()),
                             colmeans_x(X_.cols())
    
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
        
        
        // XY = Z' W y / n
        VectorXd wy = Y;
        if (wt_len) wy.array() *= weights.array();
        if (intercept)
        {
            const double wysum = wy.sum();
            XY.tail(nvars) = (X.transpose() * wy - colmeans_x * wysum).cwiseProduct(colsq_inv);
            XY(0) = intval * wysum;
        } else 
        {
            XY = (X.transpose() * wy).cwiseProduct(colsq_inv);
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
        // coefficients on the original scale of x
        VectorXd beta_ret = beta;
        if (intercept)
        {
            beta_ret.tail(nvars).array() *= colsq_inv.array();
            beta_ret(0) = intval * beta(0) - colmeans_x.dot(beta_ret.tail(nvars));
        } else
        {
            beta_ret.array() *= colsq_inv.array();
        }
        return beta_ret;
    }
    
    virtual double get_loss()
    {
        return (Y - linear_predictor(beta)).array().square().sum();
    }
};



#endif // OEM_SPARSE_H
