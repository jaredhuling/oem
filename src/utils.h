#ifndef _oem_UTILS_H
#define _oem_UTILS_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector> 
#include <functional> 
#include <algorithm> 
#include <iostream>
#include <cmath>
#include <numeric>
#include <boost/noncopyable.hpp>

using namespace Rcpp;
using namespace RcppEigen;

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::SparseMatrix;
using Eigen::RowVectorXd;
using Eigen::Lower;
using Eigen::Upper;
using Eigen::Ref;
using Eigen::Map;
using Rcpp::as;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorX;

typedef Map<const MatrixX> MapMat;
typedef Map<const VectorX> MapVec;
typedef Eigen::Triplet<double> T;
typedef Eigen::Map<Eigen::SparseMatrix<double>> MSpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Map<MatrixXd> MapMatd;
typedef Map<VectorXd> MapVecd;
//typedef Eigen::Map<const MatrixXd> MapMatdnc;
//typedef Eigen::Map<const VectorXd> MapVecdnc;
typedef Map<ArrayXd>  MapArrayd;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatR;
typedef Eigen::SparseMatrix<int, Eigen::RowMajor> SpMatIntR;

   
double threshold(double num);

VectorXd cumsum(const VectorXd& x);

VectorXd cumsumrev(const VectorXd& x);

// MATRIX PRODUCTS

VectorXd sliced_crossprod(const MatrixXd& X, const VectorXd& y, const VectorXi& idx);

VectorXd sliced_matvecprod(const MatrixXd& A, const VectorXd& b, const std::vector<int>& idx);

void sliced_crossprod_inplace(VectorXd &res, const MatrixXd& X, const VectorXd& y, const std::vector<int>& idx);


//computes X'WX where W is diagonal (input w as vector)
MatrixXd XtWX(const MapMatd& xx, const MatrixXd& ww);

//computes XWX' where W is diagonal (input w as vector)
MatrixXd XWXt(const MapMatd& xx, const MatrixXd& ww);

//SpMat X'WX where W is diagonal (input w as vector)
SpMat XtWX(const MSpMat& xx, const MatrixXd& ww);

//computes XWX' where W is diagonal (input w as vector)
SpMat XWXt(const MSpMat& xx, const MatrixXd& ww);

//computes X'X 
//MatrixXd XtX(const MapMatd &xx);

//MatrixXd XtX(MapMat &xx);

//MatrixXd XtX(const MapMat &xx);

void XtX(MatrixXd &xTx, const MatrixXd &xx);

MatrixXd XtX(const MatrixXd &xx);

//MatrixXd XtX(MatrixXd &xx);

//computes XX'
//MatrixXd XXt(const MapMatd& xx);

//MatrixXd XXt(MapMat& xx);

//MatrixXd XXt(const MapMat& xx);

MatrixXd XXt(const MapMat& xx);

//MatrixXd XXt(MatrixXd& xx);

//computes X'X 
MatrixXd XtX_scaled(const MapMatd &xx, RowVectorXd &colmeans, RowVectorXd &colstd);

//computes XX'
MatrixXd XXt_scaled(const MapMatd& xx, RowVectorXd &colmeans, RowVectorXd &colstd);

//computes X'X 
SpMat XtX(const MSpMat& xx);

//computes XX'
SpMat XXt(const MSpMat& xx);

// soft thresholding

/*
void soft_threshold(SpVec &res, const VectorXd &vec, const double &penalty);

void soft_threshold(VectorXd &res, const VectorXd &vec, const double &penalty);

void soft_threshold(SpVec &res, const VectorXd &vec, const double &penalty, VectorXd &pen_fact);

void soft_threshold(SpVec &res, const VectorXd &vec, const double &penalty, VectorXd &pen_fact, double &d);

void soft_threshold(VectorXd &res, const VectorXd &vec, const double &penalty, VectorXd &pen_fact);

void soft_threshold(VectorXd &res, const VectorXd &vec, const double &penalty, VectorXd &pen_fact, double &d);

void soft_threshold_mcp(VectorXd &res, const VectorXd &vec, const double &penalty, 
                        VectorXd &pen_fact, double &d, double &alpha);
 
 */

void update_active_set(VectorXd &u, std::vector<int> &active, std::vector<int> &inactive,
                       double &lambdak, double &lambdakminus1, const int &penalty);

void initiate_active_set(VectorXd &u, std::vector<int> &active, std::vector<int> &inactive,
                         double &lambdak, double &lambdamax, const int &nvars, const int &penalty);

void block_soft_threshold(SpVec &res, const VectorXd &vec, const double &penalty,
                                 const int &ngroups, VectorXi &unique_grps, VectorXi &grps);

void block_soft_threshold(VectorXd &res, const VectorXd &vec, const double &penalty,
                          const int &ngroups, VectorXi &unique_grps, VectorXi &grps);

/*
void block_soft_threshold(SpVec &res, const VectorXd &vec, const double &penalty,
                                 const int &ngroups, const MapVeci &unique_grps, const MapVeci &grps);
 */ 
  
bool stopRule(const VectorXd& cur, const VectorXd& prev, const double& tolerance);

bool stopRule(const SpVec& cur, const SpVec& prev, const double& tolerance);

bool stopRuleMat(const MatrixXd& cur, const MatrixXd& prev, const double& tolerance);


/*
template <typename T, typename T2>
T extract(const T2& full, const T& ind)
{
    int num_indices = ind.innerSize();
    T target(num_indices);
    for (int i = 0; i < num_indices; i++)
    {
        target[i] = full[ind[i]];
    }
    return target;
}
*/


// Scale of the response used by glmnet for Gaussian models: the standard
// deviation of y (denominator n) with an intercept, and the root mean square
// of y without one. Returns 1 when that scale is zero.
inline double oem_response_scale(const VectorXd &y, bool intercept)
{
    double center = intercept ? y.mean() : 0.0;
    double s = std::sqrt((y.array() - center).square().mean());
    return (s > 0.0) ? s : 1.0;
}

// Thresholding rule for one coordinate, given u = d * b + score:
// type 0 is the lasso, 1 is MCP, and 2 is SCAD, with penalty level pen.
// These are the rules used by soft_threshold, soft_threshold_mcp, and
// soft_threshold_scad in the solver classes.
inline double oem_threshold_scalar(double u, double pen, double d, int type, double gamma)
{
    if (type == 1)
    {
        if (std::abs(u) > gamma * d * pen) return u / d;
        double denom = d - 1.0 / gamma;
        if (u > pen) return (u - pen) / denom;
        if (u < -pen) return (u + pen) / denom;
        return 0.0;
    } else if (type == 2)
    {
        if (std::abs(u) > gamma * d * pen) return u / d;
        if (std::abs(u) > (d + 1.0) * pen)
        {
            double gam_u = (gamma - 1.0) * u;
            double gam_pen = gamma * pen;
            double denom = (gamma - 1.0) * d - 1.0;
            if (gam_u > gam_pen) return (gam_u - gam_pen) / denom;
            if (gam_u < -gam_pen) return (gam_u + gam_pen) / denom;
            return 0.0;
        }
        if (u > pen) return (u - pen) / d;
        if (u < -pen) return (u + pen) / d;
        return 0.0;
    }
    if (u > pen) return (u - pen) / d;
    if (u < -pen) return (u + pen) / d;
    return 0.0;
}

// Update for the elementwise ".net" penalties
//   lambda * pf_j * ( alpha * P(|b_j|) + (1 - alpha) / 2 * b_j^2 ),
// with P the lasso, MCP, or SCAD penalty (type as in oem_threshold_scalar).
// Coordinate j uses L1 level lam_l1 * pf_j and denominator d + lam_l2 * pf_j,
// so a coordinate with pf_j = 0, such as the intercept, is neither
// thresholded nor shrunk by the ridge term.
inline void oem_net_threshold(VectorXd &res, const VectorXd &u, int type,
                              double lam_l1, double lam_l2, const VectorXd &pf,
                              double d, double gamma)
{
    for (int j = 0; j < u.size(); ++j)
    {
        res(j) = oem_threshold_scalar(u(j), lam_l1 * pf(j), d + lam_l2 * pf(j), type, gamma);
    }
}

// For the group ".net" penalties, the coefficients in group 0 (unpenalized
// variables and, where the solver carries it, the intercept) take the
// unpenalized update u / d, with no ridge term.
inline void oem_unpenalized_group_update(VectorXd &res, const VectorXd &u,
                                         const std::vector<std::vector<int> > &grp_idx,
                                         const VectorXi &unique_groups, double d)
{
    for (int g = 0; g < unique_groups.size(); ++g)
    {
        if (unique_groups(g) != 0) continue;
        for (std::vector<int>::size_type v = 0; v < grp_idx[g].size(); ++v)
        {
            res(grp_idx[g][v]) = u(grp_idx[g][v]) / d;
        }
    }
}

#endif
