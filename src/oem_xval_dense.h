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
class oemXvalDense: public oemBase<Eigen::VectorXd> //Eigen::SparseVector<double>
{
protected:
    typedef float Scalar;
    typedef double Double;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRXd;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Map<const Matrix> MapMat;
    typedef Map<const Vector> MapVec;
    typedef Map<const MatrixXd> MapMatd;
    typedef Map<const MatrixRXd> MapMatRd;
    typedef Map<const VectorXd> MapVecd;
    typedef Map<VectorXi> MapVeci;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    
    const MapMatRd X;            // data matrix
    MapVec Y;                   // response vector
    VectorXd weights;
    VectorXi foldid;            // id vector for cv folds
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
    int nfolds;                 // number of cross validation folds
    std::vector<MatrixXd > xtx_list;
    std::vector<VectorXd > xty_list;
    std::vector<int > nobs_list;
    std::vector<VectorXd > colsq_list;
    VectorXd colsq_inv;
    VectorXd colsq;
    
    
    std::vector<std::vector<int> > grp_idx; // vector of vectors of the indexes for all members of each group
    std::string penalty;        // penalty specified
    
    double lambda;              // L1 penalty
    double lambda0;             // minimum lambda to make coefficients all zero
    double alpha;               // alpha = mixing parameter for elastic net
    double gamma;               // extra tuning parameter for mcp/scad
    double tau;                 // mixing parameter for group sparse penalties
    
    double threshval;
    int wt_len;
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
    
    
    MatrixXd XtX() const {
        return MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
        rankUpdate(X.adjoint());
    }
    
    MatrixXd XXt() const {
        return MatrixXd(XXdim, XXdim).setZero().selfadjointView<Lower>().
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
    
    
    // computing all the X'X and X'Y pieces
    // for all k folds
    void XtX_xval(std::vector<MatrixXd > &xtx_list_, 
                  std::vector<VectorXd > &xty_list_,
                  std::vector<int > &nobs_list_, 
                  std::vector<VectorXd > &colsq_list_) const {
        
        // MatrixRXd A = X;
        
        
        // static enforces k = i comes before k = i + 1
        #pragma omp parallel for schedule(static)
        for (int k = 1; k < nfolds + 1; ++k)
        {
            
            VectorXi idxbool = (foldid.array() == k).cast<int>();
            int nrow_cur = idxbool.size();
            int numelem = idxbool.sum();
            VectorXi idx(numelem);
            int c = 0;
            for (int i = 0; i < nrow_cur; ++i)
            {
                if (idxbool(i) == 1)
                {
                    idx(c) = i;
                    ++c;
                }
            }
            
            // store subset of matrix X and response Y for this fold
            MatrixXd sub(numelem, nvars);
            VectorXd sub_y(numelem);
            for (int r = 0; r < numelem; ++r)
            {
                sub.row(r) = X.row(idx(r));
                sub_y(r) = Y(idx(r));
            }
            
            MatrixXd AtAtmp(MatrixXd(nvars, nvars).setZero().
                            selfadjointView<Lower>().rankUpdate(sub.adjoint() ));
            
            VectorXd AtBtmp = sub.adjoint() * sub_y;
            
            VectorXd colsqtmp = sub.array().square().colwise().sum();
            
            sub.resize(0,0);
            
            // store the X'X and X'Y of the subset
            // of data for fold k
            xtx_list_[k-1] = AtAtmp;
            xty_list_[k-1] = AtBtmp;
            nobs_list_[k-1] = numelem;
            colsq_list_[k-1] = colsqtmp;
            
        }
    }
    
    
    // computing all the X'X and X'Y pieces
    // for all k folds
    // when an intercept is needed
    void XtX_xval_int(std::vector<MatrixXd > &xtx_list_, 
                      std::vector<VectorXd > &xty_list_,
                      std::vector<int > &nobs_list_, 
                      std::vector<VectorXd > &colsq_list_) const {
        
        //MatrixRXd A = X;
        
        
        // static enforces k = i comes before k = i + 1
        #pragma omp parallel for schedule(static)
        for (int k = 1; k < nfolds + 1; ++k)
        {
            
            VectorXi idxbool = (foldid.array() == k).cast<int>();
            int nrow_cur = idxbool.size();
            int numelem = idxbool.sum();
            VectorXi idx(numelem);
            int c = 0;
            for (int i = 0; i < nrow_cur; ++i)
            {
                if (idxbool(i) == 1)
                {
                    idx(c) = i;
                    ++c;
                }
            }
            
            // store subset of matrix X and response Y for this fold
            MatrixXd sub(numelem, nvars);
            VectorXd sub_y(numelem);
            for (int r = 0; r < numelem; ++r)
            {
                sub.row(r) = X.row(idx(r));
                sub_y(r) = Y(idx(r));
            }
            
            MatrixXd AtAtmp(MatrixXd(nvars+1, nvars+1).setZero());
            
            // compute X'X for this fold 
            // with intercept 
            AtAtmp.bottomRightCorner(nvars, nvars) = MatrixXd(nvars, nvars).setZero()
                .selfadjointView<Lower>().rankUpdate(sub.adjoint() );
            
            Eigen::RowVectorXd colsums = sub.colwise().sum();
            AtAtmp.block(0,1,1,nvars) = colsums;
            AtAtmp.block(1,0,nvars,1) = colsums.transpose();
            AtAtmp(0,0) = numelem;
            
            
            // compute X'Y for this fold 
            // with intercept 
            VectorXd AtBtmp(nvars + 1);
            AtBtmp.tail(nvars) = sub.adjoint() * sub_y;
            AtBtmp(0) = sub_y.sum();
            
            VectorXd colsqtmp = sub.array().square().colwise().sum();
            
            sub.resize(0,0);
            
            // store the X'X and X'Y of the subset
            // of data for fold k
            xtx_list_[k-1] = AtAtmp;
            xty_list_[k-1] = AtBtmp;
            nobs_list_[k-1] = numelem;
            colsq_list_[k-1] = colsqtmp;
            
        }
    }
    
    // computing all the X'X and X'Y pieces
    // for all k folds 
    // when observation weights are used
    void XtWX_xval(std::vector<MatrixXd > &xtx_list_, 
                   std::vector<VectorXd > &xty_list_,
                   std::vector<int > &nobs_list_, 
                   std::vector<VectorXd > &colsq_list_) const {
        
        //MatrixRXd A = X;
        
        
        // static enforces k = i comes before k = i + 1
        #pragma omp parallel for schedule(static)
        for (int k = 1; k < nfolds + 1; ++k)
        {
            
            VectorXi idxbool = (foldid.array() == k).cast<int>();
            int nrow_cur = idxbool.size();
            int numelem = idxbool.sum();
            VectorXi idx(numelem);
            int c = 0;
            for (int i = 0; i < nrow_cur; ++i)
            {
                if (idxbool(i) == 1)
                {
                    idx(c) = i;
                    ++c;
                }
            }
            
            // store subset of matrix X and response Y for this fold
            MatrixXd sub(numelem, nvars);
            VectorXd sub_y(numelem);
            VectorXd sub_weights(numelem);
            for (int r = 0; r < numelem; ++r)
            {
                int idx_tmp_val = idx(r);
                sub.row(r)      = X.row(idx_tmp_val);
                sub_y(r)        = Y(idx_tmp_val);
                sub_weights(r)  = weights(idx_tmp_val);
            }
            
            MatrixXd AtWAtmp(MatrixXd(nvars, nvars).setZero().
                                selfadjointView<Lower>().rankUpdate( ((sub_weights.array().sqrt().matrix()).asDiagonal() * sub).adjoint() ));
            
            VectorXd AtWBtmp = sub.adjoint() * (sub_y.array() * sub_weights.array()).matrix();
            
            // this commented code is wrong! we do not standardize with respect to weights
            //VectorXd colsqtmp = ((sub_weights.array().sqrt().matrix()).asDiagonal() * sub).array().square().colwise().sum();
            VectorXd colsqtmp = sub.array().square().colwise().sum();
            
            sub.resize(0,0);
            
            // store the X'X and X'Y of the subset
            // of data for fold k
            xtx_list_[k-1]   = AtWAtmp;
            xty_list_[k-1]   = AtWBtmp;
            nobs_list_[k-1]  = numelem;
            colsq_list_[k-1] = colsqtmp;
            
        }
    }
    
    
    // computing all the X'X and X'Y pieces
    // for all k folds
    // when an intercept is needed and weights are used
    void XtWX_xval_int(std::vector<MatrixXd > &xtx_list_, 
                       std::vector<VectorXd > &xty_list_,
                       std::vector<int > &nobs_list_, 
                       std::vector<VectorXd > &colsq_list_) const {
        
        //MatrixRXd A = X;
        
        
        // static enforces k = i comes before k = i + 1
        #pragma omp parallel for schedule(static)
        for (int k = 1; k < nfolds + 1; ++k)
        {
            
            VectorXi idxbool = (foldid.array() == k).cast<int>();
            int nrow_cur = idxbool.size();
            int numelem = idxbool.sum();
            VectorXi idx(numelem);
            int c = 0;
            for (int i = 0; i < nrow_cur; ++i)
            {
                if (idxbool(i) == 1)
                {
                    idx(c) = i;
                    ++c;
                }
            }
            
            // store subset of matrix X and response Y for this fold
            MatrixXd sub(numelem, nvars);
            VectorXd sub_y(numelem);
            VectorXd sub_weights(numelem);
            for (int r = 0; r < numelem; ++r)
            {
                int idx_tmp_val = idx(r);
                sub.row(r)      = X.row(idx_tmp_val);
                sub_y(r)        = Y(idx_tmp_val);
                sub_weights(r)  = weights(idx_tmp_val);
            }
            
            MatrixXd AtAtmp(MatrixXd(nvars+1, nvars+1).setZero());
            
            // compute X'X for this fold 
            // with intercept and weights
            AtAtmp.bottomRightCorner(nvars, nvars) = MatrixXd(nvars, nvars).setZero()
                  .selfadjointView<Lower>().rankUpdate((sub_weights.array().sqrt().matrix()).asDiagonal() * sub.adjoint() );
            
            Eigen::RowVectorXd colsums = (((sub_weights.array().matrix()).asDiagonal() * sub).colwise().sum()).matrix(); 
            
            AtAtmp.block(0,1,1,nvars) = colsums;
            AtAtmp.block(1,0,nvars,1) = colsums.transpose();
            AtAtmp(0,0) = sub_weights.sum();
            
            
            // compute X'Y for this fold 
            // with intercept 
            VectorXd AtBtmp(nvars + 1);
            VectorXd yw(numelem);
            yw = (sub_y.array() * sub_weights.array());
            AtBtmp.tail(nvars) = sub.adjoint() * yw;
            AtBtmp(0) = yw.sum();
            
            //VectorXd colsqtmp = ((sub_weights.array().sqrt().matrix()).asDiagonal() * sub).array().square().colwise().sum();
            VectorXd colsqtmp = sub.array().square().colwise().sum();
            
            sub.resize(0,0);
            
            // store the X'X and X'Y of the subset
            // of data for fold k
            xtx_list_[k-1]   = AtAtmp;
            xty_list_[k-1]   = AtBtmp;
            nobs_list_[k-1]  = numelem;
            colsq_list_[k-1] = colsqtmp;
            
        }
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
                for (int v = 0; v < nvars + intercept; ++v) 
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
    
    void compute_XtX_d_update_A(bool add_int_)
    {
        // clear out XX, XY
        XX.setZero();
        XY.setZero();
        
        // compute X'X
        // if weights specified, compute X'WX instead
        // also need to handle differently
        // if intercept == true
        if (add_int_)
        {
            
            if (wt_len)
            {
                
                if (nobs > nvars) 
                {
                    // this computes all the X'X and X'Y
                    // pieces for each fold
                    XtWX_xval_int(xtx_list, xty_list, nobs_list, colsq_list);
                } else 
                {
                    throw std::invalid_argument("dimension of x larger than number of observations");
                }
            } else 
            {
                if (nobs > nvars) 
                {
                    // this computes all the X'X and X'Y
                    // pieces for each fold
                    XtX_xval_int(xtx_list, xty_list, nobs_list, colsq_list);
                } else 
                {
                    throw std::invalid_argument("dimension of x larger than number of observations");
                }
            }
        } else 
        {
            if (wt_len)
            {
                if (nobs > nvars) 
                {
                    // this computes all the X'X and X'Y
                    // pieces for each fold
                    XtWX_xval(xtx_list, xty_list, nobs_list, colsq_list);
                } else 
                {
                    throw std::invalid_argument("dimension of x larger than number of observations");
                }
            } else 
            {
                if (nobs > nvars) 
                {
                    // this computes all the X'X and X'Y
                    // pieces for each fold
                    XtX_xval(xtx_list, xty_list, nobs_list, colsq_list);
                } else 
                {
                    throw std::invalid_argument("dimension of x larger than number of observations");
                }
            }
        }
        
        nobs = 0;
        colsq.setZero();
        for (int k = 0; k < nfolds; ++k)
        {
            // compute
            // X'X and X'Y for all the data
            // except current fold
            XX += xtx_list[k];
            XY += xty_list[k];
            nobs += nobs_list[k];
            colsq.array() += colsq_list[k].array();
        }
        
        colsq /= (nobs - 1);
        colsq = (colsq.array() == 0.0).select(1.0, colsq);
        colsq_inv = 1 / colsq.array().sqrt();
        
        
        if (standardize)
        {
            if (intercept)
            {
                XX.bottomRightCorner(nvars, nvars) = colsq_inv.asDiagonal() * XX.bottomRightCorner(nvars, nvars) * colsq_inv.asDiagonal();
                XX.row(0).tail(nvars).array() *= colsq_inv.array();
                XX.col(0).tail(nvars).array() *= colsq_inv.array();
                //XX.topRightCorner(1, nvars).array()   *= colsq_inv.array();
                //XX.bottomLeftCorner(nvars, 1).array() *= colsq_inv.array();
                XY.tail(nvars).array() *= colsq_inv.array();
            } else 
            {
                XX = colsq_inv.asDiagonal() * XX * colsq_inv.asDiagonal();
                XY.array() *= colsq_inv.array();
            }
        }
        
        XX /= nobs;
        XY /= nobs;
        
        
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
    
    void update_XtX_d_update_A(int fold_cur_)
    {
        
        XX.setZero();
        XY.setZero();
        nobs = 0;
        
        colsq.setZero();
        
        for (int k = 1; k < nfolds + 1; ++k)
        {
            // compute
            // X'X and X'Y for all the data
            // except current fold
            if (k != fold_cur_)
            {
                XX += xtx_list[k-1];
                XY += xty_list[k-1];
                nobs += nobs_list[k-1];
                colsq.array() += colsq_list[k-1].array();
            }
        }
        
        colsq /= (double(nobs) - 1.0);
        colsq = (colsq.array() == 0.0).select(1.0, colsq);
        colsq_inv = 1.0 / colsq.array().sqrt();
        
        
        if (standardize)
        {
            if (intercept)
            {
                XX.bottomRightCorner(nvars, nvars) = colsq_inv.asDiagonal() * XX.bottomRightCorner(nvars, nvars) * colsq_inv.asDiagonal();
                XX.row(0).tail(nvars).array() *= colsq_inv.array();
                XX.col(0).tail(nvars).array() *= colsq_inv.array();
                XY.tail(nvars).array() *= colsq_inv.array();
            } else 
            {
                XX = colsq_inv.asDiagonal() * XX * colsq_inv.asDiagonal();
                XY.array() *= colsq_inv.array();
            }
        }
        
        XX /= nobs;
        XY /= nobs;
        
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
        } else 
        {
            throw std::invalid_argument("dimension of x larger than number of observations");
        }
    }
    
    
    // define the u update in oem
    void next_u(Vector &res)
    {
        if (nobs > nvars)
        {
            res.noalias() = A * beta_prev + XY;
        } else 
        {
            throw std::invalid_argument("dimension of x larger than number of observations");
            // res.noalias() = X.adjoint() * (Y - X * beta_prev) / double(nobs) + d * beta_prev;
        }
    }
    
    
    // define the beta update in oem
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
    oemXvalDense(const Eigen::Ref<const MatrixRXd> &X_, 
                 ConstGenericVector &Y_,
                 const VectorXd &weights_,
                 const int &nfolds_,
                 const VectorXi &foldid_,
                 const VectorXi &groups_,
                 const VectorXi &unique_groups_,
                 VectorXd &group_weights_,
                 VectorXd &penalty_factor_,
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
                             foldid(foldid_),
                             groups(groups_),
                             unique_groups(unique_groups_),
                             penalty_factor(penalty_factor_),
                             group_weights(group_weights_),
                             penalty_factor_size(penalty_factor_.size()),
                             XXdim( std::min(X_.cols(), X_.rows()) + intercept_ * (X_.rows() > X_.cols()) ),
                             XY(X_.cols() + intercept_),      // add extra space if intercept 
                             XX(XXdim, XXdim),                // add extra space if intercept 
                             default_group_weights(bool(group_weights_.size() < 1)), // compute default weights if none given
                             nfolds(nfolds_),
                             xtx_list(nfolds_),
                             xty_list(nfolds_),
                             nobs_list(nfolds_),
                             colsq_list(nfolds_),
                             colsq_inv(X_.cols()),
                             colsq(X_.cols()),
                             grp_idx(unique_groups_.size())
    
    {}
    
    void init_xtx(bool add_int_)
    {
        wt_len = weights.size();
        
        // compute XtX or XXt (depending on if n > p or not)
        // and compute A = dI - XtX (if n > p)
        
        found_grp_idx = false;
        
        compute_XtX_d_update_A(add_int_);
        
        if (intercept)
        {
            u.resize(nvars + 1);
            beta.resize(nvars + 1);
            beta_prev.resize(nvars + 1);
        }
        
    }
    
    void update_xtx(int fold_)
    {
        update_XtX_d_update_A(fold_);
    }
    
    double compute_lambda_zero() 
    { 
        
        // XY should already be computed
        
        /*
        if (wt_len)
        {
            XY.noalias() = X.transpose() * (Y.array() * weights.array()).matrix();
        } else
        {
            XY.noalias() = X.transpose() * Y;
        }
        
        XY /= nobs;
         */
        
        
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



#endif // OEM_DENSE_TALL_H
