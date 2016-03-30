#ifndef _oem_LANCZOS_H
#define _oem_LANCZOS_H


#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/SVD>
#include <vector> 
#include <functional> 
#include <algorithm> 
#include <iostream>
#include <cmath>


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;
using Eigen::JacobiSVD;
using Eigen::Ref;
using Eigen::MappedSparseMatrix;
using Eigen::SparseMatrix;
typedef Eigen::MappedSparseMatrix<double> MSpMat;
typedef Eigen::SparseMatrix<double> SpMat;

void GKLBidiag(const Ref<const MatrixXd>& AA, double& eigenv, Ref<VectorXd> v, 
               Ref<VectorXd> alpha, Ref<VectorXd> beta, const int nrow, bool ls);

void GKLBidiagSparse(const SparseMatrix<double>& AA, double& eigenv, Ref<VectorXd> v, 
                     Ref<VectorXd> alpha, Ref<VectorXd> beta, const int nrow, bool ls);


#endif