// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cmath>
#include <string_view>
#include <armadillo>

#include <saltatlas/dnnd/detail/utilities/blas.hpp>
#include <saltatlas/dnnd/detail/utilities/functional.hpp>

namespace saltatlas::dndetail::distance {
template <typename T>
using metric_type = T(const std::size_t, const T *const, const T *const);

template <typename T>
inline auto invalid(const std::size_t, const T *const, const T *const) {
  assert(false);
  return T{};
}

template <typename T>
inline auto l2(const std::size_t len, const T *const f0, const T *const f1) {
  T d = 0;
  for (std::size_t i = 0; i < len; ++i) {
    const auto x = (f0[i] - f1[i]);
    d += x * x;
  }
  return static_cast<T>(std::sqrt(d));
}

template <typename T>
inline auto elastic(const std::size_t len, const T *const f0, const T *const f1) {
  double alpha = 0.5;
  
  T tst = 0;
  for (std::size_t i = 0; i < len; ++i) {
    const auto x = (f0[i] - f1[i]);
    tst += x * x;
  }
  T dist = 0;
  if (tst == 0){
    dist = 0;
  }
  else{
    arma::vec q0(f0, len);
    arma::vec q1(f1, len);

    arma::vec gam0 = warp(q0, q1);
    arma::vec time = arma::linspace(0,1,len);
    arma::vec gam(len);
    gam = (gam0 - gam[0]) / (gam[len-1] - gam[0]);
    arma::vec gam_dev(len);
    gam_dev = grad(gam, 1/(len-1));

    arma::vec tmp;
    arma::vec time_new;
    time_new = (time[len-1]-time[0]) * gam + time[0];
    arma::interp1(time, q1, time_new, tmp);

    arma::vec qw;
    qw = tmp * arma::sqrt(gam_dev);

    arma::vec y;
    y = arma::square(qw-q0);

    arma::vec dd;
    dd = arma::diff(time);
    tmp = dd*(y.rows(0,len-2)+y.rows(1,len-1))/2;
    T da = std::sqrt(sum(tmp));

    arma::vec psi;
    psi = arma::sqrt(grad(gam, 1/(len-1)));
    double q1dotq2 = arma::trapz(time, psi);
    if (q1dotq2 > 1){
      q1dotq2 = 1;
    } else if (q1dotq2 < -1){
      q1dotq2 = -1;
    }
    T dp = std::acos(q1dotq2);

    dist = (1-alpha) * da + alpha * dp;

  }

  return static_cast<T>(dist);
}

template <typename T>
inline auto cosine(const std::size_t len, const T *const f0,
                   const T *const f1) {
  const T n0 = std::sqrt(dndetail::blas::inner_product(len, f0, f0));
  const T n1 = std::sqrt(dndetail::blas::inner_product(len, f1, f1));
  if (n0 == 0 && n1 == 0)
    return static_cast<T>(0);
  else if (n0 == 0 || n1 == 0)
    return static_cast<T>(1);

  const T x = dndetail::blas::inner_product(len, f0, f1);
  return static_cast<T>(1.0 - x / (n0 * n1));
}

template <typename T>
inline auto jaccard_index(const std::size_t len, const T *const f0,
                          const T *const f1) {
  std::size_t num_non_zero = 0;
  std::size_t num_equal    = 0;
  for (std::size_t i = 0; i < len; ++i) {
    const bool x_true = !!f0[i];
    const bool y_true = !!f1[i];
    num_non_zero += x_true | y_true;
    num_equal += x_true & y_true;
  }

  if (num_non_zero == 0)
    return T{0};
  else
    return static_cast<T>(num_non_zero - num_equal) /
           static_cast<T>(num_non_zero);
}

enum class metric_id : uint8_t { invalid, l2, cosine, jaccard };

inline metric_id convert_to_metric_id(const std::string_view &metric_name) {
  if (metric_name == "l2") {
    return metric_id::l2;
  } else if (metric_name == "cosine") {
    return metric_id::cosine;
  } else if (metric_name == "jaccard") {
    return metric_id::jaccard;
  }
  return metric_id::invalid;
}

template <typename T>
inline metric_type<T> &metric(const metric_id &id) {
  if (id == metric_id::l2) {
    return l2<T>;
  } else if (id == metric_id::cosine) {
    return cosine<T>;
  } else if (id == metric_id::jaccard) {
    return jaccard_index<T>;
  }
  return invalid<T>;
}

template <typename T>
inline metric_type<T> &metric(const std::string_view metric_name) {
  return metric<T>(convert_to_metric_id(metric_name));
}
}  // namespace saltatlas::dndetail::distance
