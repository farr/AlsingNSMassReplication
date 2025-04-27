functions {
  real mp_mt_likelihood_lpdf(array[] real mt_mean, array[] real f, array[] real mp, array[] real mt, array[] real mt_std) {
    int n = size(mt_mean);
    array[n] real lps;

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

  real mp_q_likelihood_lpdf(array[] real q_mean, array[] real f, array[] real mp, array[] real q, array[] real q_std) {
    int n = size(q_mean);
    array[n] real lps;

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

  real n_gaussian_pop_lpdf(array[] real mp, vector log_As, vector mus, vector sigmas, vector log_norms) {
    int np = size(mp);
    int ng = size(log_As);
    array[np] real lps;

    for (i in 1:np) {
      array[ng] real lp_temp;
      for (j in 1:ng) {
        lp_temp[j] = log_As[j] + normal_lpdf(mp[i] | mus[j], sigmas[j]) - log_norms[j];
      }

      lps[i] = log_sum_exp(lp_temp);
    }

    return sum(lps);
  }
}

data {
  int n_gaussian;
  int n_mt;
  int n_q;

  int num_gaussian_components;

  array[n_gaussian] real mp_mean;
  array[n_gaussian] real mp_std;

  array[n_mt] real f_mt;
  array[n_mt] real mt_mean;
  array[n_mt] real mt_std;

  array[n_q] real f_q;
  array[n_q] real q_mean;
  array[n_q] real q_std;
}

parameters {
  real<lower=0, upper=1> dmmax;

  simplex[num_gaussian_components] As;
  // Support both 1 or more components
  vector<lower=0, upper=1>[num_gaussian_components] delta_mus;

  vector<lower=0>[num_gaussian_components] sigmas;

  /* For the Gaussian population */
  array[n_gaussian] real mp_gaussian_raw;

  /* Same, but for measurements of f and mt. */
  array[n_mt] real<lower=0, upper=1> mp_mt_raw;
  array[n_mt] real<lower=0> mt_raw;

  /* Same but for measurements of f and q. */
  array[n_q] real<lower=0> mp_q_raw;
  array[n_q] real q_raw;
}

transformed parameters {
  real mmax;
  vector[num_gaussian_components] mus;
  vector[num_gaussian_components] log_norms;
  array[n_gaussian] real mp_gaussian;
  array[n_mt] real mp_mt;
  array[n_mt] real mp_mt_logjac;
  array[n_mt] real mt;
  array[n_q] real mp_q;
  array[n_q] real q;

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
    array[3] real ms = {mg, mst, mq};
    mmax = max(ms) + dmmax;
  }

  if (num_gaussian_components == 1) {
    mus[1] = 1 + 1.5 * delta_mus[1];  // reuse delta_mus[1] from a simplex<lower=0>[1] (redefined as a real)
  } else {
    mus[1] = 1 + delta_mus[1];
    for (i in 2:num_gaussian_components) {
      mus[i] = mus[i-1] + (2.5-1)*delta_mus[i];
    }
  }

  for (i in 1:num_gaussian_components) {
    log_norms[i] = log_diff_exp(normal_lcdf(mmax | mus[i], sigmas[i]),
                                normal_lcdf(0 | mus[i], sigmas[i]));
  }
}


model {
  vector[num_gaussian_components] log_As = log(As);

  array[n_gaussian] real lpsg;

  /* As long as alpha = beta, this is the same as A[1] ~ beta(2*alpha-1,
  /* 2*beta-1), or p(A[1]) ~ A[1]^(2*alpha-2)(1-A[1])^(2*alpha-2) */
  As ~ beta(2,2);

  /* Flat prior on `dmmax` => flat prior on `mmax`, conditioned on all the `mp`s. */
  mus ~ normal(1.75, 1);
  sigmas ~ normal(0.01, 2);

  /* Priors on mp; for mt and q, prior is flat. */
  mp_gaussian ~ n_gaussian_pop(log_As, mus, sigmas, log_norms);
  /* Jacobian to mp_gaussian_raw. */
  target += -sum(log(mp_std));
  mp_mt ~ n_gaussian_pop(log_As, mus, sigmas, log_norms);
  /* Jacobian mp_mt to mp_mt_raw */
  target += sum(mp_mt_logjac);
  mp_q ~ n_gaussian_pop(log_As, mus, sigmas, log_norms);

  /* Likelihoods */
  mp_mean ~ normal(mp_gaussian, mp_std);
  mt_mean ~ mp_mt_likelihood(f_mt, mp_mt, mt, mt_std);
  q_mean ~ mp_q_likelihood(f_q, mp_q, q, q_std);
}

generated quantities {
  real mp_draw;
  real x = uniform_rng(0,1);

  for (i in 1:num_gaussian_components) {
    if (x < As[i]) {
      mp_draw = -1.0;
      while ((mp_draw < 0) || (mp_draw > mmax)) {
        mp_draw = normal_rng(mus[i], sigmas[i]);
      }
      break;
    } else {
      x -= As[i];
    }
  }
}
