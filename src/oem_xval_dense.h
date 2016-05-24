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
    typedef Map<const VectorXd> MapVecd;
    typedef Map<VectorXi> MapVeci;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    
    const MapMatd X;            // data matrix
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
    double alpha;               // alpha = mixing parameter for elastic net
    double gamma;               // extra tuning parameter for mcp/scad
    bool default_group_weights; // do we need to compute default group weights?
    int nfolds;                 // number of cross validation folds
    std::vector<MatrixXd > xtx_list;
    std::vector<VectorXd > xty_list;
    std::vector<int > nobs_list;
    
    std::vector<std::vector<int> > grp_idx; // vector of vectors of the indexes for all members of each group
    std::string penalty;       // penalty specified
    
    double lambda;             // L1 penalty
    double lambda0;            // minimum lambda to make coefficients all zero
    
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
                  std::vector<int > &nobs_list_) const {
        
        MatrixRXd A = X;
        
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
                sub.row(r) = A.row(idx(r));
                sub_y(r) = Y(idx(r));
            }
            
            MatrixXd AtAtmp(MatrixXd(nvars, nvars).setZero().
                            selfadjointView<Lower>().rankUpdate(sub.adjoint() ));
            
            VectorXd AtBtmp = sub.adjoint() * sub_y;
            
            // store the X'X and X'Y of the subset
            // of data for fold k
            xtx_list_[k-1] = AtAtmp;
            xty_list_[k-1] = AtBtmp;
            nobs_list_[k-1] = numelem;
            
        }
    }
    
    
    // computing all the X'X and X'Y pieces
    // for all k folds
    // when an intercept is needed
    void XtX_xval_int(std::vector<MatrixXd > &xtx_list_, 
                      std::vector<VectorXd > &xty_list_,
                      std::vector<int > &nobs_list_) const {
        
        MatrixRXd A = X;
        
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
                sub.row(r) = A.row(idx(r));
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
            
            // store the X'X and X'Y of the subset
            // of data for fold k
            xtx_list_[k-1] = AtAtmp;
            xty_list_[k-1] = AtBtmp;
            nobs_list_[k-1] = numelem;
            
        }
    }
    
    // computing all the X'X and X'Y pieces
    // for all k folds 
    // when observation weights are used
    void XtWX_xval(std::vector<MatrixXd > &xtx_list_, 
                   std::vector<VectorXd > &xty_list_,
                   std::vector<int > &nobs_list_) const {
        
        MatrixRXd A = X;
        
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
                sub.row(r) = A.row(idx_tmp_val);
                sub_y(r) = Y(idx_tmp_val);
                sub_weights(r) = weights(idx_tmp_val);
            }
            
            MatrixXd AtWAtmp(MatrixXd(nvars, nvars).setZero().
                                selfadjointView<Lower>().rankUpdate((sub_weights.array().sqrt().matrix()).asDiagonal() * sub.adjoint() ));
            
            VectorXd AtWBtmp = sub.adjoint() * (sub_y.array() * sub_weights.array()).matrix();
            
            // store the X'X and X'Y of the subset
            // of data for fold k
            xtx_list_[k-1] = AtWAtmp;
            xty_list_[k-1] = AtWBtmp;
            nobs_list_[k-1] = numelem;
            
        }
    }
    
    
    // computing all the X'X and X'Y pieces
    // for all k folds
    // when an intercept is needed and weights are used
    void XtWX_xval_int(std::vector<MatrixXd > &xtx_list_, 
                       std::vector<VectorXd > &xty_list_,
                       std::vector<int > &nobs_list_) const {
        
        MatrixRXd A = X;
        
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
                sub.row(r) = A.row(idx_tmp_val);
                sub_y(r) = Y(idx_tmp_val);
                sub_weights(r) = weights(idx_tmp_val);
            }
            
            MatrixXd AtAtmp(MatrixXd(nvars+1, nvars+1).setZero());
            
            // compute X'X for this fold 
            // with intercept and weights
            AtAtmp.bottomRightCorner(nvars, nvars) = MatrixXd(nvars, nvars).setZero()
                  .selfadjointView<Lower>().rankUpdate((sub_weights.array().sqrt().matrix()).asDiagonal() * sub.adjoint() );
            
            Eigen::RowVectorXd colsums = (((sub_weights.array().matrix()).asDiagonal() * sub).colwise().sum()).matrix(); 
            
            AtAtmp.block(0,1,1,nvars) = colsums;
            AtAtmp.block(1,0,nvars,1) = colsums.transpose();
            AtAtmp(0,0) = numelem;
            
            
            // compute X'Y for this fold 
            // with intercept 
            VectorXd AtBtmp(nvars + 1);
            AtBtmp.tail(nvars) = sub.adjoint() * sub_y;
            AtBtmp(0) = sub_y.sum();
            
            // store the X'X and X'Y of the subset
            // of data for fold k
            xtx_list_[k-1] = AtAtmp;
            xty_list_[k-1] = AtBtmp;
            nobs_list_[k-1] = numelem;
            
        }
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
                    group_weights(g) = sqrt(double(grp_idx[g].size()));
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
                    XtWX_xval_int(xtx_list, xty_list, nobs_list);
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
                    XtX_xval_int(xtx_list, xty_list, nobs_list);
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
                    XtWX_xval(xtx_list, xty_list, nobs_list);
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
                    XtX_xval(xtx_list, xty_list, nobs_list);
                } else 
                {
                    throw std::invalid_argument("dimension of x larger than number of observations");
                }
            }
        }
        
        nobs = 0;
        for (int k = 0; k < nfolds; ++k)
        {
            // compute
            // X'X and X'Y for all the data
            // except current fold
            XX += xtx_list[k];
            XY += xty_list[k];
            nobs += nobs_list[k];
        }
        
        XX /= nobs;
        XY /= nobs;
        
        std::cout << "xtx(5,5)" << XX.topLeftCorner(5,5) << std::endl;
        
        std::cout << "xty(5)" << XY.head(5) << std::endl;
        
        std::cout << "xtx ncol = " << XX.cols() << std::endl;
        
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
    
    void update_XtX_d_update_A(int fold_cur_)
    {
        
        
        XX.setZero();
        XY.setZero();
        nobs = 0;
        
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
    oemXvalDense(const Eigen::Ref<const MatrixXd>  &X_, 
                 ConstGenericVector &Y_,
                 const VectorXd &weights_,
                 const int &nfolds_,
                 const VectorXi &foldid_,
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
                             foldid(foldid_),
                             groups(groups_),
                             unique_groups(unique_groups_),
                             penalty_factor(penalty_factor_),
                             group_weights(group_weights_),
                             penalty_factor_size(penalty_factor_.size()),
                             XXdim( std::min(X_.cols(), X_.rows()) + intercept_ * (X_.rows() > X_.cols())  ),
                             XY(X_.cols()),      // add extra space if intercept 
                             XX(XXdim, XXdim),   // add extra space if intercept 
                             alpha(alpha_),
                             gamma(gamma_),
                             default_group_weights(bool(group_weights_.size() < 1)), // compute default weights if none given
                             nfolds(nfolds_),
                             xtx_list(nfolds_),
                             xty_list(nfolds_),
                             nobs_list(nfolds_),
                             grp_idx(unique_groups_.size())
    
    {}
    
    void init_xtx(bool add_int_)
    {
        wt_len = weights.size();
        
        // compute XtX or XXt (depending on if n > p or not)
        // and compute A = dI - XtX (if n > p)
        compute_XtX_d_update_A(add_int_);
        
        // get indexes of members of each group.
        // best to do just once in the beginning
        get_group_indexes();
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
        loss = (Y - X * beta).array().square().sum();
        return loss;
    }
};



#endif // OEM_DENSE_TALL_H
