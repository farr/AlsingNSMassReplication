functions {
  real mp_mt_likelihood_lpdf(real[] mt_mean, real[] f, real[] mp, real[] mt, real[] mt_std) {
    int n = size(mt_mean);
    real lps[n];

    for (i in 1:n) {
      real mt_m_mp = mt[i] - mp[i];
      real mt_m_mp2 = mt_m_mp*mt_m_mp;
      real log_mt_m_mp = log(mt_m_mp);
      real log_mt = log(mt[i]);
      real log_mp = log(mp[i]);
      real log_f = log(f[i]);

      lps[i] = normal_lpdf(mt_mean[i] | mt[i], mt_std[i]) - (log(3) + 2.0*log_mt_m_mp - 4.0/3.0*log_mt + 1.0/3.0*log_f + 0.5*log1p(-f[i]^(2.0/3.0)*mt[i]^(4.0/3.0)/(mt_m_mp*mt_m_mp)));
    }

    return sum(lps);
  }

  real mp_q_likelihood_lpdf(real[] q_mean, real[] f, real[] mp, real[] q, real[] q_std) {
    int n = size(q_mean);
    real lps[n];

    for (i in 1:n) {
      real opq = 1.0 + q[i];
      real log_q = log(q[i]);
      real log_opq = log(opq);
      real log_f = log(f[i]);
      real log_mp = log(mp[i]);

      lps[i] = normal_lpdf(q_mean[i] | q[i], q_std[i]) - (log(3) + 1.0/3.0*log_f + 2.0/3.0*log_mp + 2.0*log_q - 4.0/3.0*log_opq + 0.5*log1p(-(f[i]/mp[i])^(2.0/3.0)*opq^(4.0/3.0)/(q[i]*q[i])));
    }

    return sum(lps);
  }

  real two_gaussian_pop_lpdf(real[] mp, vector log_As, vector mus, vector sigmas, vector log_norms) {
    int n = size(mp);
    real lps[n];

    for (i in 1:n) {
      lps[i] = log_sum_exp(log_As[1] + normal_lpdf(mp[i] | mus[1], sigmas[1]) - log_norms[1],
                           log_As[2] + normal_lpdf(mp[i] | mus[2], sigmas[2]) - log_norms[2]);
    }

    return sum(lps);
  }
}

data {
  int n_gaussian;
  int n_mt;
  int n_q;

  real mp_mean[n_gaussian];
  real mp_std[n_gaussian];

  real f_mt[n_mt];
  real mt_mean[n_mt];
  real mt_std[n_mt];

  real f_q[n_q];
  real q_mean[n_q];
  real q_std[n_q];
}

transformed data {
  int x_i[0];

  real x_r_mt[n_mt];
  real x_r_q[n_q];

  for (i in 1:n_mt) {
    x_r_mt[i] = f_mt[i];
  }

  for (i in 1:n_q) {
    x_r_q[i] = f_q[i];
  }
}

parameters {
  real<lower=0, upper=1> dmmax;

  simplex[2] As;
  real<lower=1, upper=2.5> mu1;
  real<lower=mu1, upper=2.5> mu2;
  vector<lower=0>[2] sigmas;

  /* For the Gaussian population */
  real mp_gaussian_raw[n_gaussian];

  /* Same, but for measurements of f and mt. */
  real<lower=0, upper=1> mp_mt_raw[n_mt];
  real<lower=0> mt_raw[n_mt];

  /* Same but for measurements of f and q. */
  real<lower=0> mp_q_raw[n_q];
  real q_raw[n_q];
}

transformed parameters {
  real mmax;
  vector[2] mus = to_vector({mu1, mu2});
  vector[2] log_norms;
  real mp_gaussian[n_gaussian];
  real mp_mt[n_mt];
  real mp_mt_logjac[n_mt];
  real mt[n_mt];
  real mp_q[n_q];
  real q[n_q];

  for (i in 1:n_gaussian) {
    mp_gaussian[i] = mp_mean[i] + mp_std[i]*mp_gaussian_raw[i];
    if (mp_gaussian[i] < 0.0) reject("Gaussian pulsar mass measurement permitted negative mass!");
  }

  for (i in 1:n_mt) {
    real mp_max;
    mt[i] = mt_mean[i] + mt_std[i]*mt_raw[i];

    if (mt[i] < f_mt[i]) reject("Gaussian total mass measurement permits inconsistent mass function!");

    mp_max = mt[i] - f_mt[i]^(1.0/3.0)*mt[i]^(2.0/3.0);
    mp_mt[i] = mp_max*mp_mt_raw[i];
    mp_mt_logjac[i] = log(mp_max);
  }

  for (i in 1:n_q) {
    real mp_min;
    q[i] = q_mean[i] + q_std[i]*q_raw[i];
    mp_min = f_q[i]*(1+q[i])*(1+q[i])/(q[i]*q[i]*q[i]);
    mp_q[i] = mp_min + mp_q_raw[i];
  }

  {
    real mg = max(mp_gaussian);
    real mst = max(mp_mt);
    real mq = max(mp_q);
    real ms[3] = {mg, mst, mq};
    mmax = max(ms) + dmmax;
  }

  for (i in 1:2) {
    log_norms[i] = log_diff_exp(normal_lcdf(mmax | mus[i], sigmas[i]),
                                normal_lcdf(0 | mus[i], sigmas[i]));
  }
}

model {
  vector[2] log_As = log(As);

  real lpsg[n_gaussian];
  // real lpsm[n_mt];
  // real lpsq[n_q];

  /* As long as alpha = beta, this is the same as A[1] ~ beta(2*alpha-1,
  /* 2*beta-1), or p(A[1]) ~ A[1]^(2*alpha-2)(1-A[1])^(2*alpha-2) */
  As ~ beta(2,2);

  /* Flat prior on `dmmax` => flat prior on `mmax`, conditioned on all the `mp`s. */
  mus ~ normal(1.75, 1);
  sigmas ~ normal(0, 2);

  /* Priors on mp; for mt and q, prior is flat. */
  mp_gaussian ~ two_gaussian_pop(log_As, mus, sigmas, log_norms);
  /* Jacobian to mp_gaussian_raw. */
  target += -sum(log(mp_std));
  mp_mt ~ two_gaussian_pop(log_As, mus, sigmas, log_norms);
  /* Jacobian mp_mt to mp_mt_raw */
  target += sum(mp_mt_logjac);
  mp_q ~ two_gaussian_pop(log_As, mus, sigmas, log_norms);

  /* Likelihoods */
  mp_mean ~ normal(mp_gaussian, mp_std);
  mt_mean ~ mp_mt_likelihood(f_mt, mp_mt, mt, mt_std);
  q_mean ~ mp_q_likelihood(f_q, mp_q, q, q_std);
}

generated quantities {
  real mp_draw;

  if (uniform_rng(0,1) < As[1]) {
    mp_draw = -1.0;
    while ((mp_draw < 0) || (mp_draw > mmax)) {
      mp_draw = normal_rng(mus[1], sigmas[1]);
    }
  } else {
    mp_draw = -1.0;
    while ((mp_draw < 0) || (mp_draw > mmax)) {
      mp_draw = normal_rng(mus[2], sigmas[2]);
    }
  }
}
