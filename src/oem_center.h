#ifndef OEM_CENTER_H
#define OEM_CENTER_H

#include <RcppEigen.h>
#include "Spectra/SymEigsSolver.h"

// Implicit centering for solvers that carry the intercept as coefficient 0 of
// the design Z = [1, X S^{-1}]. The solver instead works with the design
//   Z~ = Z T,  T = [ c  -v' ]
//                  [ 0   I  ],
// that is Z~ = [c 1, X S^{-1} - 1 v'], where v = S^{-1} m holds the (scaled)
// column means. Centering makes the intercept column orthogonal to the others,
// which greatly improves the conditioning of Z~'Z~, and c scales the intercept
// column so that its curvature matches the largest eigenvalue of the other
// columns, so that the intercept converges in about one iteration. With
// theta~ the coefficients for Z~, the coefficients for Z are T theta~.

inline double oem_largest_eigenvalue(const Eigen::MatrixXd &M)
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

// Replaces XX = Z'WZ / n (intercept first) with T' XX T, and sets v and c.
// The column means are the weighted means implied by XX, so the cross terms
// between the intercept and the other columns are zero.
inline void oem_center_gram(Eigen::MatrixXd &XX, Eigen::VectorXd &v, double &c)
{
    const int p = XX.cols() - 1;
    const double a = XX(0, 0);
    const Eigen::VectorXd b = XX.block(1, 0, p, 1);
    v = b / a;
    Eigen::MatrixXd G = XX.bottomRightCorner(p, p);
    G -= v * b.transpose() + b * v.transpose();
    G += a * v * v.transpose();
    double drest = oem_largest_eigenvalue(G);
    c = (drest > 0.0) ? std::sqrt(drest / a) : 1.0;
    XX.setZero();
    XX(0, 0) = a * c * c;
    XX.bottomRightCorner(p, p) = G;
}

// Replaces XY = Z'Wy / n with T' XY.
inline void oem_center_xy(Eigen::VectorXd &XY, const Eigen::VectorXd &v, double c)
{
    const int p = XY.size() - 1;
    const double y0 = XY(0);
    XY(0) = c * y0;
    XY.tail(p) -= v * y0;
}

// Coefficients for Z from coefficients for Z~.
inline Eigen::VectorXd oem_uncenter_theta(const Eigen::VectorXd &theta, const Eigen::VectorXd &v, double c)
{
    Eigen::VectorXd out = theta;
    out(0) = c * theta(0) - v.dot(theta.tail(theta.size() - 1));
    return out;
}

#endif // OEM_CENTER_H
