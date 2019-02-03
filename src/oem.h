#ifndef _OEM_HPP
#define _OEM_HPP

//#include <Rcpp.h>
#include <RcppArmadillo.h>
#include <vector>
#include <iostream>
#include <cmath>

// constants
enum PENALTY {OLS, LASSO, SCAD, ELASTIC, NGARROTE, MCP};
enum STOPRULE{L1, L2, INF};
int stopping = INF;
double alpha = 3; // tuning parameter in SCAD
double tol = 1e-4;
int pos = 0;
bool eval;
Rcpp::NumericVector values(100 * 500); 


// declaration of functions
// inline arma::colvec indicator(arma::uvec, unsigned);
inline arma::colvec positive(arma::colvec);
RcppExport SEXP oemfit(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
		       SEXP);
double powerM(arma::mat A); 
inline arma::colvec signVec(arma::colvec);
inline double sign(double);
inline double threshold_o(double);
inline bool stopRule(const arma::colvec&, const arma::colvec&);
//void print(arma::colvec input) { std::cout << input << std::endl;}
//void print(arma::mat input) {std::cout << input << std::endl;}
//void print(arma::uvec input) { std::cout << input << std::endl;}

int sampleSize;

// class definition
class penalty {
public:
  penalty(arma::mat&);
  const arma::mat& getX() { return blockX; };
  const arma::mat& getA() { return A;};
  double getEigen() { return eigenVal;};
protected:
  int numVariables;
  double lambda;
  arma::mat blockX;
  arma::mat A;
  double eigenVal;
  arma::colvec beta;
};

class oem {
public:
  oem(int, int, int, arma::mat&, arma::colvec&, int);
  arma::colvec calc (double, arma::colvec);
  int getIter() {return tolIter;};
  int getDF() {return df;};
  double objective(double, const arma::colvec&);
  void getEigen(Rcpp::NumericVector& output)
  {
    for (int i = 0; i < numGroup; i++)
      output[i] = penObj[i].getEigen();
  }
private:
  inline arma::colvec solution(arma::colvec&, double, double) const;
  const int numVariables;
  const int maxIter;
  const int numGroup;
  const arma::mat& design;
  const arma::colvec& response;
  const int penType;
  arma::colvec resid;
  std::vector<penalty> penObj;
  std::vector<int> index;
  // results to be returned to R
  int tolIter; // total iterations spent
  int df;      // degrees of freedom
};

#endif // _OEM_HPP
