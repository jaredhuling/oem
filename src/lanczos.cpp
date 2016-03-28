

#include "lanczos.h"



void GKLBidiag(const Ref<const MatrixXd>& AA, double& eigenv, Ref<VectorXd> vv, 
               Ref<VectorXd> alpha, Ref<VectorXd> beta, const int nrow, bool ls) {

  VectorXd u(nrow);
  VectorXd Uprev(vv.size());
  VectorXd Vnew(vv.size());
  const int kmax(alpha.size());
  MatrixXd B(MatrixXd::Zero(kmax, kmax));
  vv.normalize();
    
  for (int i = 0; i < kmax; i++) {
    Uprev = u;
    u = AA * vv;
    if (i > 0) {
      u -= beta(i - 1) * Uprev;
      //add reorthogonalization
    }
    alpha(i) = u.norm();
    u /= alpha(i);
    Vnew = AA.adjoint() * u;
    Vnew -= alpha(i) * vv;
    beta(i) = Vnew.norm();
    //std::cout << "lanc err: " << std::sqrt(wksq) << std::endl;
    //std::cout << "alphabeta sq: " << (std::pow(alpha(i), 2) + std::pow(beta(i), 2))<< std::endl;
    //wksq -= (std::pow(alpha(i), 2) + std::pow(beta(i), 2));
    
    /*
    if (std::sqrt(wksq) <= toler)
    {
      finalk = i + 1;
      break;
    }
     */
    vv = Vnew.array() / beta(i);
  }
  
  B.diagonal() = alpha;
  for (int i = 0; i < kmax - 1; i++) {
    B(i, i + 1) = beta(i);
  }
  
  JacobiSVD<MatrixXd> svd(B);
  
  eigenv = pow(svd.singularValues()(0), 2);
  if (ls) {
    if (svd.singularValues()(kmax-1) < 1e-10) {
      eigenv += pow(svd.singularValues()(kmax-2), 2);
    } else {
      eigenv += pow(svd.singularValues()(kmax-1), 2);
    }
    eigenv = eigenv / 2;
  }

}



void GKLBidiagSparse(const SparseMatrix<double>& AA, double& eigenv, Ref<VectorXd> vv, 
                     Ref<VectorXd> alpha, Ref<VectorXd> beta, const int nrow, bool ls) {

  VectorXd u(nrow);
  VectorXd Uprev(vv.size());
  VectorXd Vnew(vv.size());
  const int kmax(alpha.size());
  MatrixXd B(MatrixXd::Zero(kmax, kmax));
  
  
  vv.normalize();
    
  for (int i = 0; i < kmax; i++) {
    Uprev = u;
    u = AA * vv;
    if (i > 0) {
      u -= beta(i - 1) * Uprev;
      //add reorthogonalization
    }
    alpha(i) = u.norm();
    u /= alpha(i);
    Vnew = AA.adjoint() * u;
    Vnew -= alpha(i) * vv;
    beta(i) = Vnew.norm();
    vv = Vnew.array() / beta(i);
  }
  
  B.diagonal() = alpha;
  for (int i = 0; i < kmax - 1; i++) {
    B(i, i + 1) = beta(i);
  }
  
  JacobiSVD<MatrixXd> svd(B);
  
  eigenv = pow(svd.singularValues()(0), 2);
  if (ls) {
    if (svd.singularValues()(kmax-1) < 1e-10) {
      eigenv += pow(svd.singularValues()(kmax-2), 2);
    } else {
      eigenv += pow(svd.singularValues()(kmax-1), 2);
    }
    eigenv = eigenv / 2;
  }

}


