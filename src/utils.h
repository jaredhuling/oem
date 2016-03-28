#ifndef _oem2_UTILS_H
#define _oem2_UTILS_H


#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector> 
#include <functional> 
#include <algorithm> 
#include <iostream>
#include <cmath>
#include <numeric>


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::SparseMatrix;
typedef Eigen::SparseVector<double> SparseVector;
typedef Eigen::Map<VectorXi> MapVeci;
using Eigen::Lower;
using Eigen::Ref;

   
double threshold(double num);

VectorXd cumsum(const VectorXd& x);

VectorXd cumsumrev(const VectorXd& x);

VectorXd sliced_crossprod(const MatrixXd& X, const VectorXd& y, const VectorXi& idx);

VectorXd sliced_matvecprod(const MatrixXd& A, const VectorXd& b, const std::vector<int>& idx);

void sliced_crossprod_inplace(VectorXd &res, const MatrixXd& X, const VectorXd& y, const std::vector<int>& idx);

//computes X'WX where W is diagonal (input w as vector)
MatrixXd XtWX(const MatrixXd& xx, const MatrixXd& ww);

//computes X'X 
MatrixXd XtX(const MatrixXd& xx);

void soft_threshold(SparseVector &res, const VectorXd &vec, const double &penalty);

void soft_threshold(VectorXd &res, const VectorXd &vec, const double &penalty);

void update_active_set(VectorXd &u, std::vector<int> &active, std::vector<int> &inactive,
                       double &lambdak, double &lambdakminus1, const int &penalty);

void initiate_active_set(VectorXd &u, std::vector<int> &active, std::vector<int> &inactive,
                         double &lambdak, double &lambdamax, const int &nvars, const int &penalty);

void block_soft_threshold(SparseVector &res, const VectorXd &vec, const double &penalty,
                                 const int &ngroups, VectorXi &unique_grps, VectorXi &grps);

void block_soft_threshold(VectorXd &res, const VectorXd &vec, const double &penalty,
                          const int &ngroups, VectorXi &unique_grps, VectorXi &grps);

/*
void block_soft_threshold(SparseVector &res, const VectorXd &vec, const double &penalty,
                                 const int &ngroups, const MapVeci &unique_grps, const MapVeci &grps);
 */ 
  
bool stopRule(const VectorXd& cur, const VectorXd& prev, const double& tolerance);

bool stopRule(const SparseVector& cur, const SparseVector& prev, const double& tolerance);

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

#endif