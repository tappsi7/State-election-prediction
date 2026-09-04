data {
  int<lower=1> N_nat;                          // number of national polls
  int<lower=2> P;                              // number of parties
  int<lower=1> T;                              // number of weeks, 1 = first, T = election week
  int<lower=1> J_nat;                          // number of national polling institutes
  array[N_nat] int<lower=1> sample_nat;
  array[N_nat] int<lower=1, upper=T> week_nat;
  array[N_nat] int<lower=1, upper=J_nat> institute_nat;
  array[N_nat] simplex[P] y_nat;

  int<lower=0> N_state;                        // number of state polls
  int<lower=1> J_state;                        // number of state-level institutes
  array[N_state] int<lower=1> sample_state;
  array[N_state] int<lower=1, upper=T> week_state;
  array[N_state] int<lower=1, upper=J_state> institute_state;
  array[N_state] simplex[P] y_state;

  vector[P-1] prior_delta0_mean;               // last-election state-minus-national gap, log-ratio scale
}

parameters {
  // --- national block ---
  real mu_kappa;
  real<lower=1e-8> sigma_kappa;
  vector[P-1] eta0;
  array[T-1] vector[P-1] z_rw;
  vector[J_nat] log_kappa_raw;
  vector<lower=1e-8>[P-1] sigma_rw;
  vector<lower=1e-8>[P-1] sigma_house;
  array[J_nat - 1] vector[P-1] house_raw;

  // --- state block ---
  vector[P-1] delta0;
  real<lower=0,upper=1> phi_delta;
  vector<lower=1e-8>[P-1] sigma_delta;
  array[T-1] vector[P-1] z_delta;
  real mu_kappa_state;
  real<lower=1e-8> sigma_kappa_state;
  vector[J_state] log_kappa_state_raw;
  vector<lower=1e-8>[P-1] sigma_house_state;
  array[J_state - 1] vector[P-1] house_state_raw;
}

transformed parameters {
  // --- national ---
  array[T] simplex[P] theta;
  array[J_nat] vector[P-1] house;
  array[N_nat] simplex[P] pi_nat;
  vector[J_nat] kappa = exp(mu_kappa + sigma_kappa * log_kappa_raw);
  vector[N_nat] phi_poll_nat;
  for (i in 1:N_nat)
    phi_poll_nat[i] = 2 * (sample_nat[i] / kappa[institute_nat[i]]);

  array[T] vector[P-1] eta_free;
  eta_free[1] = eta0;
  for (t in 2:T)
    eta_free[t] = eta_free[t-1] + z_rw[t-1] .* sigma_rw;

  {
    vector[P-1] house_sum = rep_vector(0, P-1);
    for (j in 1:(J_nat - 1)) {
      house[j] = house_raw[j];
      house_sum += house_raw[j];
    }
    house[J_nat] = -house_sum;
  }

  for (t in 1:T)
    theta[t] = softmax(append_row(eta_free[t], 0));

  for (i in 1:N_nat)
    pi_nat[i] = softmax(append_row(eta_free[week_nat[i]] + house[institute_nat[i]], 0));

  // --- state ---
  array[T] vector[P-1] delta;
  array[T] simplex[P] theta_state;
  array[J_state] vector[P-1] house_state;
  array[N_state] simplex[P] pi_state;
  vector[J_state] kappa_state = exp(mu_kappa_state + sigma_kappa_state * log_kappa_state_raw);
  vector[N_state] phi_poll_state;
  for (i in 1:N_state)
    phi_poll_state[i] = 2 * (sample_state[i] / kappa_state[institute_state[i]]);

  delta[1] = delta0 + z_delta[1] .* sigma_delta;
  for (t in 2:T)
    delta[t] = delta0 + phi_delta * (delta[t-1] - delta0) + z_delta[t-1] .* sigma_delta;

  {
    vector[P-1] house_sum_s = rep_vector(0, P-1);
    for (j in 1:(J_state - 1)) {
      house_state[j] = house_state_raw[j];
      house_sum_s += house_state_raw[j];
    }
    house_state[J_state] = -house_sum_s;
  }

  for (t in 1:T)
    theta_state[t] = softmax(append_row(eta_free[t] + delta[t], 0));

  for (i in 1:N_state)
    pi_state[i] = softmax(append_row(
      eta_free[week_state[i]] + delta[week_state[i]] + house_state[institute_state[i]], 0));
}

model {
  // --- national priors ---
  eta0 ~ normal(0, 5);
  for (t in 1:(T-1)) z_rw[t] ~ normal(0, 2);
  sigma_rw ~ lognormal(log(0.02), 0.5);
  sigma_house ~ lognormal(log(0.1), 0.5);
  mu_kappa ~ normal(log(3), 0.7);
  sigma_kappa ~ normal(0, 1);
  log_kappa_raw ~ std_normal();
  for (j in 1:(J_nat - 1)) house_raw[j] ~ normal(0, sigma_house);

  // --- state priors ---
  delta0 ~ normal(prior_delta0_mean, 0.3);
  phi_delta ~ beta(8, 2);
  sigma_delta ~ lognormal(log(0.02), 0.5);
  for (t in 1:(T-1)) z_delta[t] ~ normal(0, 1);
  mu_kappa_state ~ normal(log(3), 0.7);
  sigma_kappa_state ~ normal(0, 1);
  log_kappa_state_raw ~ std_normal();
  sigma_house_state ~ lognormal(log(0.1), 0.5);
  for (j in 1:(J_state - 1)) house_state_raw[j] ~ normal(0, sigma_house_state);

  // --- likelihoods ---
  for (i in 1:N_nat)
    y_nat[i] ~ dirichlet(pi_nat[i] * phi_poll_nat[i]);
  for (i in 1:N_state)
    y_state[i] ~ dirichlet(pi_state[i] * phi_poll_state[i]);
}

generated quantities {
  simplex[P] election_day_prediction_nat = theta[T];
  simplex[P] election_day_prediction_state = theta_state[T];
  vector[P-1] delta_election_day = delta[T];   // final estimated state lean, log-ratio scale

  vector[N_nat] log_lik_nat;
  vector[N_state] log_lik_state;
  for (i in 1:N_nat)
    log_lik_nat[i] = dirichlet_lpdf(y_nat[i] | pi_nat[i] * phi_poll_nat[i]);
  for (i in 1:N_state)
    log_lik_state[i] = dirichlet_lpdf(y_state[i] | pi_state[i] * phi_poll_state[i]);
}

