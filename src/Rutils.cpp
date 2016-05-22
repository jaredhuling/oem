
#include "Rutils.h"

using namespace Rcpp;
using namespace RcppEigen;

//port faster cross product 
RcppExport SEXP crossprodcpp(SEXP X)
{
  using namespace Rcpp;
  using namespace RcppEigen;
  try {
    using Eigen::Map;
    using Eigen::MatrixXd;
    using Eigen::Lower;
    const Eigen::Map<MatrixXd> A(as<Map<MatrixXd> >(X));
    const int n(A.cols());
    MatrixXd AtA(MatrixXd(n, n).setZero().
    selfadjointView<Lower>().rankUpdate(A.adjoint()));
    return wrap(AtA);
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}


//port faster scaledcross product 
RcppExport SEXP tcrossprodcpp_scaled(SEXP X)
{
    using namespace Rcpp;
    using namespace RcppEigen;
    try {
        using Eigen::Map;
        using Eigen::MatrixXd;
        using Eigen::Lower;
        const Eigen::Map<MatrixXd> A(as<Map<MatrixXd> >(X));
        const int n(A.rows());
        
        
        // these need to be RowVectorXd if 
        // we do not transpose A first
        Eigen::RowVectorXd mean = A.colwise().mean();
        Eigen::RowVectorXd std = ((A.rowwise() - mean).array().square().colwise().sum() / (n - 1)).sqrt();
        
        MatrixXd AAt(MatrixXd(n, n).setZero().
                         selfadjointView<Lower>().rankUpdate(((A.rowwise() - mean).array().rowwise() / std.array()).matrix() ));
        
        return wrap(AAt);
    } catch (std::exception &ex) {
        forward_exception_to_r(ex);
    } catch (...) {
        ::Rf_error("C++ exception (unknown reason)");
    }
    return R_NilValue; //-Wall
}

//port faster scaled cross product 
RcppExport SEXP crossprodcpp_scaled(SEXP X)
{
    using namespace Rcpp;
    using namespace RcppEigen;
    try {
        using Eigen::Map;
        using Eigen::MatrixXd;
        using Eigen::Lower;
        const Eigen::Map<MatrixXd> A(as<Map<MatrixXd> >(X));
        const int n(A.rows());
        const int p(A.cols());
        
        
        // these need to be RowVectorXd if 
        // we do not transpose A first
        Eigen::RowVectorXd mean = A.colwise().mean();
        Eigen::RowVectorXd std = ((A.rowwise() - mean).array().square().colwise().sum() / (n - 1)).sqrt();
        
        
        // this currently induces a copy, need to fix if possible
        MatrixXd AtA(MatrixXd(p, p).setZero().
                         selfadjointView<Lower>().rankUpdate(((A.rowwise() - mean).array().rowwise() / std.array()).array().matrix().adjoint() ));
        
        return wrap(AtA);
    } catch (std::exception &ex) {
        forward_exception_to_r(ex);
    } catch (...) {
        ::Rf_error("C++ exception (unknown reason)");
    }
    return R_NilValue; //-Wall
}



//port faster cross product 
RcppExport SEXP largestEig(SEXP X)
{
    using namespace Rcpp;
    using namespace RcppEigen;
    try {
        using Eigen::Map;
        using Eigen::MatrixXd;
        using Eigen::Lower;
        
        Rcpp::NumericMatrix xx(X);
        const int n = xx.rows();
        
        MatrixXd A(n, n);
        
        // Copy data 
        std::copy(xx.begin(), xx.end(), A.data());
        
        Spectra::DenseSymMatProd<double> op(A);
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, 1, 4);
        
        eigs.init();
        eigs.compute(1000, 0.0001);
        Eigen::VectorXd eigenvals = eigs.eigenvalues();
        double d = eigenvals[0];
        
        return wrap(d);
    } catch (std::exception &ex) {
        forward_exception_to_r(ex);
    } catch (...) {
        ::Rf_error("C++ exception (unknown reason)");
    }
    return R_NilValue; //-Wall
}

//port faster cross product 
RcppExport SEXP xpwx(SEXP X, SEXP W)
{
  using namespace Rcpp;
  using namespace RcppEigen;
  try {
    using Eigen::Map;
    using Eigen::MatrixXd;
    using Eigen::Lower;
    const Eigen::Map<MatrixXd> A(as<Map<MatrixXd> >(X));
    const Eigen::Map<MatrixXd> diag(as<Map<MatrixXd> >(W));
    const int n(A.cols());
    MatrixXd AtA(MatrixXd(n, n).setZero().
    selfadjointView<Lower>().rankUpdate(A.adjoint() * diag.sqrt()));
    return wrap(AtA);
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}

//port faster cross product 
RcppExport SEXP xxt(SEXP X)
{
    using namespace Rcpp;
    using namespace RcppEigen;
    try {
        using Eigen::Map;
        using Eigen::MatrixXd;
        using Eigen::Lower;
        const Eigen::Map<MatrixXd> A(as<Map<MatrixXd> >(X));
        
        const int n(A.rows());
        MatrixXd AtA(MatrixXd(n, n).setZero().
                         selfadjointView<Lower>().rankUpdate(A));
        return wrap(AtA);
    } catch (std::exception &ex) {
        forward_exception_to_r(ex);
    } catch (...) {
        ::Rf_error("C++ exception (unknown reason)");
    }
    return R_NilValue; //-Wall
}

//port faster subtract
RcppExport SEXP subcpp(SEXP BB, SEXP CC)
{
  using namespace Rcpp;
  using namespace RcppEigen;
  try {
    using Eigen::Map;
    using Eigen::MatrixXd;
    typedef Eigen::Map<Eigen::MatrixXd> MapMatd;
    const MapMatd B(as<MapMatd>(BB));
    const MapMatd C(as<MapMatd>(CC));
    return wrap(B - C);
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}


//port faster add
RcppExport SEXP addcpp(SEXP BB, SEXP CC)
{
  using namespace Rcpp;
  using namespace RcppEigen;
  try {
    using Eigen::Map;
    using Eigen::MatrixXd;
    typedef Eigen::Map<Eigen::MatrixXd> MapMatd;
    const MapMatd B(as<MapMatd>(BB));
    const MapMatd C(as<MapMatd>(CC));
    return wrap(B + C);
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}



//port faster subtract sparse matrices
RcppExport SEXP subSparsecpp(SEXP BB, SEXP CC)
{
  using namespace Rcpp;
  using namespace RcppEigen;
  try {
    using Eigen::MappedSparseMatrix;
    using Eigen::SparseMatrix;
    typedef Eigen::MappedSparseMatrix<double> MSpMat;
    typedef Eigen::SparseMatrix<double> SpMat;
    const SpMat B(as<MSpMat>(BB));
    const SpMat C(as<MSpMat>(CC));
    return wrap(B - C);
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}


//port faster subaddtract sparse matrices
RcppExport SEXP addSparsecpp(SEXP BB, SEXP CC)
{
  using namespace Rcpp;
  using namespace RcppEigen;
  try {
    using Eigen::MappedSparseMatrix;
    using Eigen::SparseMatrix;
    typedef Eigen::MappedSparseMatrix<double> MSpMat;
    typedef Eigen::SparseMatrix<double> SpMat;
    const SpMat B(as<MSpMat>(BB));
    const SpMat C(as<MSpMat>(CC));
    return wrap(B + C);
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}


//Lanczos Bidiagonalization
RcppExport SEXP LanczosBidiag(SEXP A, SEXP v, SEXP k)
{
  using namespace Rcpp;
  using namespace RcppEigen;
  try {
    using Rcpp::List;
    using Eigen::Map;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using Eigen::JacobiSVD;
    typedef Eigen::Map<VectorXd> MapVecd;
    typedef Eigen::Map<Eigen::MatrixXd> MapMatd;
    const MapMatd AA(as<MapMatd>(A));
    const MapVecd Vinit(as<MapVecd>(v));
    const int kk(as<int>(k));
    //const bool reorth(as<bool>(reorthog));
    
    VectorXd v(Vinit);
    VectorXd u(AA.rows());
    VectorXd Uprev(v.size());
    VectorXd Vnew(v.size());
    MatrixXd B(MatrixXd::Zero(kk, kk));
    double d(0);
    //VectorXd Unew(v.size());
    
    v.normalize();
    
    VectorXd beta(kk);
    VectorXd alpha(kk);
    
    for (int i = 0; i < kk; i++) {
      Uprev = u;
      u = AA * v;
      if (i > 0) {
        u -= beta(i - 1) * Uprev;
        //add reorthogonalization
      }
      alpha(i) = u.norm();
      u /= alpha(i);
      Vnew = AA.adjoint() * u - alpha(i) * v;
      beta(i) = Vnew.norm();
      v = Vnew.array() / beta(i);
    }
    
    B.diagonal() = alpha;
    for (int i = 0; i < kk - 1; i++) {
      B(i, i + 1) = beta(i);
    }
    
    JacobiSVD<MatrixXd> svd(B);
    d = svd.singularValues()(0);
    
    return List::create(Named("d") = d,
                        Named("u") = u,
                        Named("v") = v,
                        Named("alpha") = alpha,
                        Named("beta") = beta);
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}

//Lanczos Bidiagonalization for Sparse Matrices
RcppExport SEXP LanczosBidiagSparse(SEXP A, SEXP v, SEXP k)
{
  using namespace Rcpp;
  using namespace RcppEigen;
  try {
    using Rcpp::List;
    using Eigen::Map;
    using Eigen::MatrixXd;
    using Eigen::MappedSparseMatrix;
    using Eigen::SparseMatrix;
    typedef Eigen::MappedSparseMatrix<double> MSpMat;
    typedef Eigen::SparseMatrix<double> SpMat;
    
    using Eigen::VectorXd;
    using Eigen::JacobiSVD;
    typedef Eigen::Map<VectorXd> MapVecd;
    const SpMat AA(as<MSpMat>(A));
    const MapVecd Vinit(as<MapVecd>(v));
    const int kk(as<int>(k));
    //const bool reorth(as<bool>(reorthog));
    
    VectorXd v(Vinit);
    VectorXd u(AA.rows());
    VectorXd Uprev(v.size());
    VectorXd Vnew(v.size());
    MatrixXd B(MatrixXd::Zero(kk, kk));
    double d(0);
    //VectorXd Unew(v.size());
    
    v.normalize();
        
    VectorXd beta(kk);
    VectorXd alpha(kk);
    
    for (int i = 0; i < kk; i++) {
      Uprev = u;
      u = AA * v;
      if (i > 0) {
        u -= beta(i - 1) * Uprev;
        //add reorthogonalization
      }
      alpha(i) = u.norm();
      u /= alpha(i);
      Vnew = AA.adjoint() * u - alpha(i) * v;
      beta(i) = Vnew.norm();
      v = Vnew.array() / beta(i);
    }
    
    B.diagonal() = alpha;
    for (int i = 0; i < kk - 1; i++) {
      B(i, i + 1) = beta(i);
    }
    
    JacobiSVD<MatrixXd> svd(B);
    d = svd.singularValues()(0);
    
    return List::create(Named("d") = d,
                        Named("u") = u,
                        Named("v") = v,
                        Named("alpha") = alpha,
                        Named("beta") = beta);
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}

//port faster cross product 
RcppExport SEXP BidiagPoly(SEXP X, SEXP alpha, SEXP beta)
{
  using namespace Rcpp;
  using namespace RcppEigen;
  try {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    typedef Eigen::Map<VectorXd> MapVecd;
    const MapVecd alph(as<MapVecd>(alpha));
    const MapVecd bet(as<MapVecd>(beta));
    const double xx(as<double>(X));
    
    double p0(1 / alph(0));
    double q0(1);
    double p(p0);
    double q(q0);
    
    for (int k = 0; k < alph.size() - 1; k++) {
      q = (xx * p0 - alph(k) * q0) / bet(k);
      p  = (q - bet(k) * p0) / alph(k + 1);
      p0 = p;
      q0 = q;
    }
    
    return wrap(p);
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}

