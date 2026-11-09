#ifndef OEM_BIG_H
#define OEM_BIG_H



#include "oem_base.h"
#include "Spectra/SymEigsSolver.h"
#include "utils.h"
#include "oem_center.h"
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
    int XXdimCalc;
    Vector XY;                  // X'Y
    MatrixXd XX;                // X'X
    MatrixXd A;                 // A = d * I - X'X
    double d;                   // d value (largest eigenvalue of X'X)
    bool default_group_weights; // do we need to compute default group weights?
    
    
    std::vector<std::vector<int> > grp_idx; // vector of vectors of the indexes for all members of each group
    std::string penalty;        // penalty specified
    
    double lambda;              // L1 penalty
    double lambda0;             // minimum lambda to make coefficients all zero
    double alpha;               // alpha = mixing parameter for elastic net
    double gamma;               // extra tuning parameter for mcp/scad
    double tau;                 // mixing parameter for group sparse penalties
    
    double threshval;
    int wt_len;
    int nslices;
    
    double gigs;
    Eigen::RowVectorXd colsums;
    Eigen::RowVectorXd colsq;
    Eigen::VectorXd colsq_inv;
    VectorXd cvec;              // scaled column means v for implicit centering
    double int_scale;           // scale c of the intercept column
    bool centered;              // true if the solver uses the centered design
    
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
        double retval = 0;
        
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
    
    
    MatrixXd XtX() const {
        if (nslices <= 1)
        {
            return MatrixXd(XXdimCalc, XXdimCalc).setZero().selfadjointView<Lower>().
            rankUpdate(X.adjoint());
        } else 
        {
            MatrixXd XXtmp(XXdimCalc, XXdimCalc);
            XXtmp.setZero();
            
            int numrowscurfirst = std::floor(double(nobs) / double(nslices) );
            
        //#pragma omp parallel
        {
            MatrixXd XXtmp_private(XXdimCalc, XXdimCalc);
            XXtmp_private.setZero();
            
            // break up computation of X'X into 
            // X'X = X_1'X_1 + ... + X_ncores'X_ncores
            //#pragma omp for schedule(static) nowait
            for (int ff = 0; ff < nslices; ++ff)
            {
                
                if (ff + 1 == nslices)
                {
                    int numrowscur = nobs - (nslices - 1) * std::floor(double(nobs) / double(nslices));
                    XXtmp_private += MatrixXd(XXdimCalc, XXdimCalc).setZero().selfadjointView<Lower>().
                    rankUpdate(X.bottomRows(numrowscur).adjoint());
                } else 
                {
                    XXtmp_private += MatrixXd(XXdimCalc, XXdimCalc).setZero().selfadjointView<Lower>().
                    rankUpdate(X.middleRows(ff * numrowscurfirst, numrowscurfirst).adjoint());
                }
            }
            //#pragma omp critical
            {
                XXtmp += XXtmp_private; 
            }
            
        }
        return XXtmp;
        }
    }
    
    MatrixXd XXt() const {
        return MatrixXd(XXdimCalc, XXdimCalc).setZero().selfadjointView<Lower>().
        rankUpdate(X);
    }
    
    MatrixXd XtWX() const {
        
        if (nslices <= 1)
        {
            return MatrixXd(XXdimCalc, XXdimCalc).setZero().selfadjointView<Lower>().
            rankUpdate(X.adjoint() * (weights.array().sqrt().matrix()).asDiagonal() );
        } else 
        {
            MatrixXd XXtmp(XXdimCalc, XXdimCalc);
            XXtmp.setZero();
            
            int numrowscurfirst = std::floor(double(nobs) / double(nslices));
            
            //#pragma omp parallel
            {
                MatrixXd XXtmp_private(XXdimCalc, XXdimCalc);
                XXtmp_private.setZero();
                
                // break up computation of X'X into 
                // X'X = X_1'X_1 + ... + X_ncores'X_ncores
                
                //#pragma omp for schedule(static) nowait
                for (int ff = 0; ff < nslices; ++ff)
                {
                    
                    if (ff + 1 == nslices)
                    {
                        int numrowscur = nobs - (nslices - 1) * std::floor(double(nobs) / double(nslices));
                        XXtmp_private += MatrixXd(XXdimCalc, XXdimCalc).setZero().selfadjointView<Lower>().
                        rankUpdate(X.bottomRows(numrowscur).adjoint() * 
                        (weights.tail(numrowscur).array().sqrt().matrix()).asDiagonal());
                    } else 
                    {
                        XXtmp_private += MatrixXd(XXdimCalc, XXdimCalc).setZero().selfadjointView<Lower>().
                        rankUpdate(X.middleRows(ff * numrowscurfirst, numrowscurfirst).adjoint() * 
                        (weights.segment(ff * numrowscurfirst, numrowscurfirst).array().sqrt().matrix()).asDiagonal());
                    }
                }
                //#pragma omp critical
                {
                    XXtmp += XXtmp_private; 
                }
                
            }
            return XXtmp;
        }
    }
    
    MatrixXd XWXt() const {
        return MatrixXd(XXdimCalc, XXdimCalc).setZero().selfadjointView<Lower>().
        rankUpdate( (weights.array().sqrt().matrix()).asDiagonal() * X );
    }
    /*
    MatrixXd XXt() const {
        return MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().
        rankUpdate(X);
    }
    
    MatrixXd XWXt() const {
        return MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().
        rankUpdate( (weights.array().sqrt().matrix()).asDiagonal() * X );
    }
     */
    
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
                // Z W Z' / n has the same nonzero eigenvalues as Z'WZ / n and is
                // n x n; it is used only to find d.
                MatrixXd Zs = X * colsq_inv.asDiagonal();
                if (wt_len)
                {
                    VectorXd sw = weights.array().sqrt().matrix();
                    Zs = sw.asDiagonal() * Zs;
                    XX = MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().rankUpdate(Zs);
                    if (intercept) XX += sw * sw.transpose();
                } else
                {
                    XX = MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().rankUpdate(Zs);
                    if (intercept)
                    {
                        // centered design: (X S^{-1} - 1 v')(X S^{-1} - 1 v')' + c^2 1 1'
                        cvec = colsums.transpose().cwiseProduct(colsq_inv) / double(nobs);
                        VectorXd Zv = Zs * cvec;
                        XX -= Zv * VectorXd::Ones(nobs).transpose() + VectorXd::Ones(nobs) * Zv.transpose();
                        XX.array() += cvec.squaredNorm();
                        double drest = oem_largest_eigenvalue(XX / double(nobs));
                        int_scale = (drest > 0.0) ? std::sqrt(drest) : 1.0;
                        XX.array() += int_scale * int_scale;
                        centered = true;
                    }
                }
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
                // Z W Z' / n has the same nonzero eigenvalues as Z'WZ / n and is
                // n x n; it is used only to find d.
                MatrixXd Zs = X * colsq_inv.asDiagonal();
                if (wt_len)
                {
                    VectorXd sw = weights.array().sqrt().matrix();
                    Zs = sw.asDiagonal() * Zs;
                    XX = MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().rankUpdate(Zs);
                    if (intercept) XX += sw * sw.transpose();
                } else
                {
                    XX = MatrixXd(nobs, nobs).setZero().selfadjointView<Lower>().rankUpdate(Zs);
                    if (intercept)
                    {
                        // centered design: (X S^{-1} - 1 v')(X S^{-1} - 1 v')' + c^2 1 1'
                        cvec = colsums.transpose().cwiseProduct(colsq_inv) / double(nobs);
                        VectorXd Zv = Zs * cvec;
                        XX -= Zv * VectorXd::Ones(nobs).transpose() + VectorXd::Ones(nobs) * Zv.transpose();
                        XX.array() += cvec.squaredNorm();
                        double drest = oem_largest_eigenvalue(XX / double(nobs));
                        int_scale = (drest > 0.0) ? std::sqrt(drest) : 1.0;
                        XX.array() += int_scale * int_scale;
                        centered = true;
                    }
                }
            }
        }
        
        XX /= nobs;
        
        if (intercept && !wt_len && nobs > nvars + int(intercept))
        {
            oem_center_gram(XX, cvec, int_scale);
            centered = true;
        }
        
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
        if (nobs > nvars + int(intercept))
        {
            res.noalias() = A * beta_prev + XY;
        } else 
        {
            // u = d * theta + Z' W (y - Z theta) / n, without forming Z'Z
            VectorXd theta = centered ? oem_uncenter_theta(beta_prev, cvec, int_scale) : VectorXd(beta_prev);
            VectorXd resid;
            if (intercept)
            {
                resid = Y - X * (theta.tail(nvars).cwiseProduct(colsq_inv));
                resid.array() -= theta(0);
            } else
            {
                resid = Y - X * (theta.cwiseProduct(colsq_inv));
            }
            if (wt_len) resid.array() *= weights.array();
            resid /= double(nobs);
            if (intercept)
            {
                const double g0 = resid.sum();
                VectorXd gt = (X.adjoint() * resid).cwiseProduct(colsq_inv);
                if (centered)
                {
                    res(0) = d * beta_prev(0) + int_scale * g0;
                    res.tail(nvars) = d * beta_prev.tail(nvars) + gt - cvec * g0;
                } else
                {
                    res(0) = d * beta_prev(0) + g0;
                    res.tail(nvars) = d * beta_prev.tail(nvars) + gt;
                }
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
        oemBig(const Map<MatrixXd> &X_, 
               ConstGenericVector &Y_,
               const VectorXd &weights_,
               const VectorXi &groups_,
               const VectorXi &unique_groups_,
               VectorXd &group_weights_,
               VectorXd &penalty_factor_,
               bool &intercept_,
               bool &standardize_,
               const double tol_ = 1e-6,
               const double gigs_ = 4.0) :
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
                                 XXdimCalc( std::min(X_.cols(), X_.rows()) ),
                                 XY(X_.cols() + int(intercept_) ), // add extra space if intercept
                                 XX(XXdim, XXdim),                 // add extra space if intercept
                                 default_group_weights(bool(group_weights_.size() < 1)), // compute default weights if none given
                                                                                    grp_idx(unique_groups_.size()),
                                 gigs(gigs_),
                                 colsums(X_.cols()),
                                 colsq(X_.cols()),
                                 colsq_inv(X_.cols())
        
                                                                                    {}
        
        
        void init_oem()
        {
            int pc = X.cols();
            wt_len = weights.size();
            
            found_grp_idx = false;
            
            double xgigs = 8.0 * double(nobs) * double(pc) / std::pow(10.0, 9);
            
            // calculate number of rows per slice
            nslices = std::ceil(xgigs / gigs);
            
            // unit scaling unless standardizing, so code that applies
            // colsq_inv is unaffected
            colsq_inv.setOnes();
            if (standardize)
            {
                if (wt_len)
                {
                    // colsq = ( weights.asDiagonal() * (X.array().square().matrix()) ).colwise().sum() / (nobs - 1);
                    // don't want to access all of X at once
                    for (int i = 0; i < pc; ++i)
                    {
                        colsq(i) = (X.col(i).array().square() * weights.array()).sum() / (nobs - 1);
                    }
                } else 
                {
                    // colsq = X.array().square().matrix().colwise().sum() / (nobs - 1);
                    // don't want to access all of X at once
                    // variance with denominator n, as in glmnet
                    for (int i = 0; i < pc; ++i)
                    {
                        double m = X.col(i).mean();
                        colsq(i) = X.col(i).array().square().sum() / double(nobs) - m * m;
                    }
                }
                colsq = (colsq.array() <= 0.0).select(1.0, colsq);
                colsq_inv = 1.0 / colsq.array().sqrt();
            }
            
            
            
            if (wt_len)
            {
                if (intercept)
                {
                    // XY.tail(nvars) = X.transpose() * (Y.array() * weights.array()).matrix();
                    // don't want to access all of X at once
                    for (int i = 0; i < pc; ++i)
                    {
                        XY(i + 1) = X.col(i).dot((Y.array() * weights.array()).matrix());
                    }
                    
                    XY(0) = (weights.array().sqrt() * Y.array()).sum();
                } else
                {
                    // XY.noalias() = X.transpose() * (Y.array() * weights.array()).matrix();
                    // don't want to access all of X at once
                    for (int i = 0; i < pc; ++i)
                    {
                        XY(i) = X.col(i).dot((Y.array() * weights.array()).matrix());
                    }
                    
                }
            } else
            {
                if (intercept)
                {
                    // XY.tail(nvars) = X.transpose() * Y;
                    // don't want to access all of X at once
                    for (int i = 0; i < pc; ++i)
                    {
                        XY(i + 1) = X.col(i).dot(Y);
                    }
                    
                    XY(0) = Y.sum();
                } else 
                {
                    // XY.noalias() = X.transpose() * Y;
                    // don't want to access all of X at once
                    for (int i = 0; i < pc; ++i)
                    {
                        XY(i) = X.col(i).dot(Y);
                    }
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
                // colsums = X.colwise().sum();
                // don't want to access all of X at once
                for (int i = 0; i < pc; ++i)
                {
                    colsums(i) = X.col(i).sum();
                }
            }
            
            // compute XtX or XXt (depending on if n > p or not)
            // and compute A = dI - XtX (if n > p)
            centered = false;
            int_scale = 1.0;
            compute_XtX_d_update_A();
            if (centered && nobs > nvars + int(intercept))
            {
                oem_center_xy(XY, cvec, int_scale);
            }
        }
        
        double compute_lambda_zero() 
        { 
            lambda0 = XY.cwiseAbs().maxCoeff();
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
            VectorXd beta_ret = centered ? oem_uncenter_theta(beta, cvec, int_scale) : VectorXd(beta);
            if (intercept)
            {
                beta_ret.tail(nvars).array() *= colsq_inv.array();
            } else
            {
                beta_ret.array() *= colsq_inv.array();
            }
            return beta_ret;
        }
        
        virtual double get_loss()
        {
            VectorXd b = get_beta();
            VectorXd eta;
            if (intercept)
            {
                eta = X * b.tail(nvars);
                eta.array() += b(0);
            } else
            {
                eta = X * b;
            }
            if (wt_len)
            {
                return ((Y - eta).array().square() * weights.array()).sum();
            }
            return (Y - eta).array().square().sum();
        }
        };



#endif // OEM_BIG_H
