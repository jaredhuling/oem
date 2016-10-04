#include "oem.h"
// #include "debug.h"

int powerIter = 0;
bool oem_cond;   // which oem method to use

// class definitions
penalty::penalty(arma::mat& input) : blockX(input) {
  numVariables = input.n_cols;
  // apply power to a smaller square matrix
  if (oem_cond) 
    A = blockX.t() * blockX / sampleSize;
  else
    A = blockX * blockX.t() / sampleSize;
  eigenVal = powerM(A);
  // get eigen value from LAPACK
  // arma::cx_mat eigVector;
  // arma::cx_vec eigValue;
  // arma::eig_gen(eigValue, eigVector, tmp);
  // eigenVal = std::real(eigValue(eigValue.n_rows - 1));
}

oem::oem(int p, int maxIter_, int num, 
	 arma::mat& X, arma::colvec& Y, int pen) :
  numVariables(p), maxIter(maxIter_),
  numGroup(num), design(X), response(Y), penType(pen)
{
  for (int i = 0; i < numGroup; i++) 
    index.push_back(numVariables / numGroup * i);
  index.push_back(numVariables);
  for (int i = 0; i < numGroup; i++) {
    arma::mat tt(design.cols(index[i], index[i + 1] - 1));
    penalty tmp(tt);
    penObj.push_back(tmp);
  }
  // reuse resid for different purpose
  if ( oem_cond )
    resid = penObj[0].getX().t() * response / sampleSize;
  else
    resid = response;
}

double oem::objective(double lambda, const arma::colvec& beta) 
{
  using namespace arma;
  colvec theta = abs(beta);
  double scad = 0;
  for (unsigned i = 0; i < beta.n_rows; i++) {
    if (theta(i) <= lambda)
      scad += lambda * theta(i);
    else if (theta(i) <= alpha * lambda)
      scad += pow(lambda, 2) + (alpha * lambda * (theta(i) - lambda) - 
				(pow(theta(i), 2) - pow(lambda, 2)) / 2.0) /
	(alpha - 1);
    else
      scad += (alpha + 1) * pow(lambda, 2) / 2.0;
  }
  double lsq = pow(norm(response - design * beta, 2), 2) / sampleSize / 2.0;
  scad += lsq;
  return scad;
}

arma::colvec oem::solution(arma::colvec& u, double lambda, double eigenVal) const
{
  using namespace arma;
  switch(penType) {
  case OLS:
    return u / eigenVal;
  case LASSO: 
    {
      return signVec(u) % positive(abs(u) - lambda) / eigenVal;
    }
  case SCAD: 
    {
      colvec ret = zeros<mat> (u.n_rows, 1);
      for (unsigned i = 0; i < u.n_rows; i++)
	if (std::abs(u(i)) <= (eigenVal + 1) * lambda)
	  ret(i) = sign(u(i)) * threshold(std::abs(u(i)) - lambda) / eigenVal;
	else if (std::abs(u(i)) <= alpha * lambda * eigenVal)
	  ret(i) = sign(u(i)) * (std::abs(u(i)) - alpha * lambda / (alpha - 1)) 
	    / (eigenVal - 1 / (alpha - 1));
	else
	  ret(i) = u(i) / eigenVal;
      return ret;
    }
  case ELASTIC:
    break;
  case NGARROTE:
    break;
  case MCP:
    {
    colvec ret = zeros<mat> (u.n_rows, 1);
    for (unsigned i = 0; i < u.n_rows; i++)
      if (std::abs(u(i)) <= alpha * lambda * eigenVal)
	ret(i) = sign(u(i)) * threshold(std::abs(u(i)) - lambda) 
	  / (eigenVal - 1 / alpha);
      else
	ret(i) = u(i) / eigenVal;
    return ret;
    }
  }
  return u;
}

arma::colvec oem::calc (double lambda,
			arma::colvec init) {
  using namespace arma;
  using namespace std;
  
  colvec beta = init;
  for (int iter = 1; iter <= maxIter; ++iter) {
    if(eval)
      values(pos++) = objective(lambda, beta);
    colvec orig = beta;
    // determine which oem approach to use
    if (oem_cond) {
      colvec prev = beta;
      colvec u = resid + (penObj[0].getEigen() * eye<mat>(numVariables, numVariables)
			  - penObj[0].getA()) * prev;
      beta = solution(u, lambda, penObj[0].getEigen());
    } else {
      for (int i = 0; i < numGroup; i++) {
	colvec prev = beta.subvec(index[i], index[i + 1] - 1);
	colvec u = penObj[i].getX().t() * resid / sampleSize + 
	  penObj[i].getEigen() * prev;
	beta.subvec(index[i], index[i + 1] - 1) = 
	  solution(u, lambda, penObj[i].getEigen());
	if (numGroup != numVariables || beta(index[i]) != prev(0))
	  resid -= penObj[i].getX() * 
	    (beta.subvec(index[i], index[i + 1] - 1) - prev);
      }
    }
    if (stopRule(beta, orig)) {
      tolIter = iter;
      uvec tmp = find(abs(beta) > 0);
      df = tmp.n_rows;
      return beta;
    }

  }  
  tolIter = maxIter;
  uvec tmp = find(abs(beta) > 0);
  df = tmp.n_rows;
  return beta;
}

double sign(double num) 
{
  if (num == 0) return 0;
  if (num < 0) return -1;
  return 1;
}

double threshold(double num) 
{
  return num > 0 ? num : 0;
}

/*
arma::colvec indicator(arma::uvec index, unsigned size) 
{
  using namespace arma;
  colvec ret = zeros<mat>(size, 1);
  ret.elem(index) = ones<mat>(index.n_rows, 1);
  return ret;
}
*/
arma::colvec positive(arma::colvec input) 
{
  using namespace arma;
  colvec ret(input);
  uvec tmp = find(input < 0);
  ret.elem(tmp) = zeros<mat>(tmp.n_rows, 1);
  return ret;
}

// main port function to R
RcppExport SEXP oemfit(SEXP X,        // design matrix
		       SEXP Y,
		       SEXP maxIter,
		       SEXP tolerance,
		       SEXP lambda,
		       SEXP method,
		       SEXP numGroup,
		       SEXP alpha_,
		       SEXP evaluate_,
		       SEXP oem_condition) 
{
  using namespace Rcpp;
  using namespace arma;
  try {
    // store data into Armadillo objects
    NumericMatrix Xr(X);
    NumericVector Yr(Y);
    int n = Xr.nrow();
    int p = Xr.ncol();
    sampleSize = n;
    // set up starting point
    pos = 0;
    mat Xa(Xr.begin(), n, p, false);
    colvec Ya(Yr.begin(), Yr.size(), false);
    // generate lambda points to evaluate
    NumericVector lam(lambda);
    colvec lambda_grid(lam.begin(), lam.size(), false);
    
    //int penType = INTEGER(method)[0];
    tol = REAL(tolerance)[0];
    NumericMatrix betaRet(p, lambda_grid.n_rows);
    NumericVector iters(lambda_grid.n_rows);
    NumericVector df(lambda_grid.n_rows);
    mat beta(betaRet.begin(), p, lambda_grid.n_rows, false);
    alpha = REAL(alpha_)[0];
    eval = INTEGER(evaluate_)[0];

    Function proctime("proc.time");
    NumericVector t1 = proctime();
    oem_cond = INTEGER(oem_condition)[0];
    oem oemObj(p, INTEGER(maxIter)[0], INTEGER(numGroup)[0],
	       Xa, Ya, INTEGER(method)[0]);
    NumericVector t2 = proctime();
    beta.col(0) = oemObj.calc(lambda_grid(0), zeros<mat>(beta.n_rows, 1));
    iters(0) = oemObj.getIter();
    df(0) = oemObj.getDF();
    for (unsigned i = 1; i < lambda_grid.n_rows; i++) {
      beta.col(i) = oemObj.calc(lambda_grid(i), beta.col(i - 1));
      iters(i) = oemObj.getIter();
      df(i) = oemObj.getDF();
    }
    NumericVector t3 = proctime();
    // store eigenvalues 
    NumericVector eigenvalues(INTEGER(numGroup)[0]);
    oemObj.getEigen(eigenvalues);

    return List::create(Named("beta") = betaRet,
			Named("df") = df,
			Named("iter") = iters,
			Named("numGroup") = numGroup,
			Named("tol") = tolerance,
			Named("eigen") = eigenvalues,
			Named("object") = values,
			Named("alpha") = alpha_,
			Named("cond") = oem_cond,
			Named("power") = powerIter,
			Named("time1") = t2[2] - t1[2],
			Named("time2") = t3[2] - t2[2]);
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}

double powerM(arma::mat A) {
  using namespace arma;
  try {
    if (A.n_rows == 1) return A(0, 0);
    colvec z0 = randu<vec>(A.n_rows);
    colvec z(A.n_rows);
    double dist = norm(z0, 2); 
    for(unsigned iter = 0; iter < 100; iter++) {
      z = A * z0 / dist;
      double tmp = std::abs( 1 - norm(z, 2) / norm(z0, 2) );
      if (tmp < tol) {
	powerIter = iter;
	break;
      }
      dist = norm(z, 2);
      z0 = z;
    }
    return norm(z, 2);
  } catch(std::exception &ex) {
    forward_exception_to_r (ex);
  } catch(...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return 0; // -Wall
}

arma::colvec signVec(arma::colvec input) {
  using namespace arma;
  colvec ret = zeros<mat>(input.n_rows, 1);
  uvec tmp = find(input > 0);
  ret.elem(tmp) = ones<mat>(tmp.n_rows, 1);
  tmp = find(input < 0);
  ret.elem(tmp) = -ones<mat>(tmp.n_rows, 1);
  return ret;
}

bool stopRule(const arma::colvec& vec1,
	      const arma::colvec& vec2) {
  using namespace arma;
  switch(stopping) {
  case L2:
    {
      uvec tmp1 = find(abs(vec1) > 0);
      uvec tmp2 = find(abs(vec2) > 0);
      if (tmp1.n_rows == 0 && tmp2.n_rows == 0) return 1;
      if (tmp1.n_rows != tmp2.n_rows) return 0;
      if (norm(vec1 - vec2, 2) / norm(vec2.elem(tmp2), 2) < tol)
	return 1;
      return 0;
    }
  case INF:
    {
      for (unsigned i = 0; i < vec1.n_rows; i++) {
	if ( (vec1(i) != 0 && vec2(i) == 0) ||
	     (vec1(i) == 0 && vec2(i) != 0) )
	  return 0;
	if (vec1(i) != 0 && vec2(i) != 0 &&
	    std::abs( (vec1(i) - vec2(i)) / vec2(i)) > tol)
	  return 0;
      }
      return 1;
      /* Tian's suggestion
      colvec tmp = abs(vec1 - vec2) / (abs(vec2) + tol * tol);
      return tmp.max() < tol;
      */
    }
  }
  return 1; // -Wall
}

