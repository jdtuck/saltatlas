#pragma once

#include <cassert>
#include <cstdlib>
#include <utility>
#include <armadillo>

#include <saltatlas/dnnd/detail/utilities/DP.hpp>

namespace saltatlas::dndetail {

  arma::vec grad(arma::vec f, double binsize){
    int n = f.n_rows;
    arma::vec g(n, arma::fill::zeros);
    arma::vec h(n, arma::fill::zeros);
    arma::vec a = arma::regspace(1,n);
    h = binsize * a;
    g[0] = (f[1]-f[0])/(h[1]-h[0]);
    g[n-1] = (f[n-1]-f[n-2])/(h[n-1]-h[n-2]);

    h = h.rows(2,n-1) - h.rows(0,n-3);
    g.rows(1,n-2) = (f.rows(2,n-1)-f.rows(0,n-3))/h[0];

    return g;
  }

  arma::vec warp(arma::vec q1, arma::vec q2){

    double lam = 0.0;
    int disp = 0;
    int pen = 1;
    int n1 = 1;
    int M = q1.n_rows;
    arma::vec gam(M, arma::fill::zeros);

    double norm1 = arma::norm(q1,2);
    double norm2 = arma::norm(q2,2);
    q1 /= norm1;
    q2 /= norm2;

    DP(q2.memptr(), q1.memptr(), n1, M, lam, pen, disp,gam.memptr());
    
    return gam;

  }
}  // namespace saltatlas::dndetail
