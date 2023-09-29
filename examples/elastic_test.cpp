#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>
#include <armadillo>


#include <saltatlas/dnnd/detail/distance.hpp>

int main() {
  int N = 100;
  arma::vec f1(N);
  arma::vec f2(N);
  arma::vec time;
  time = arma::linspace(0,2*arma::datum::pi,N); 
  f1 = arma::sin(time);
  f2 = arma::cos(time);

  double out;
  out = saltatlas::dndetail::distance::elastic(N,f1.memptr(),f2.memptr());

  std::cout << out;

  return 0;

}

