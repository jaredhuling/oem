#ifndef DEBUG_H_
#define DEBUG_H_

#include <RcppArmadillo.h>

void print(const arma::colvec& input)
{
  std::cout << input << std::endl;
}

void print(const arma::uvec& input)
{
  std::cout << input << std::endl;
}

void print(const arma::mat& input)
{
  std::cout << input << std::endl;
}

#endif // DEBUG_H_
