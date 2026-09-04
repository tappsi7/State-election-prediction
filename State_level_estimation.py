
import itertools
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings( "ignore", module = "plotnine\..*" )
import cmdstanpy as csp
from cmdstanpy.utils import cxx_toolchain_path
cxx_toolchain_path()
csp.utils.get_logger().setLevel(logging.ERROR)
import numpy as np
import statistics as stat
import pandas as pd
import plotnine as pn
import patchworklib as pw
import time
import json
import urllib.request
import io

WEBHOOK_URL = ""

def send_discord_message(text):
    payload = {"content": text}
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    
    req = urllib.request.Request(
        WEBHOOK_URL, 
        data=json.dumps(payload).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 204:
                print("Message sent successfully!")
    except Exception as e:
        print(f"Failed to send message: {e}")

nt_dat = pd.read_csv("combined.csv")

st_dat = pd.read_csv("SAN.csv")

nt_dat["Date"] = pd.to_datetime(nt_dat["Date"])

nt_dat = nt_dat[~(nt_dat['Date'] < '2016-03-13')]   
nt_dat = nt_dat[~(nt_dat['Date'] > '2026-09-05')] 
nt_dat = nt_dat.copy()

st_dat = st_dat[~(st_dat['Date'] < '2016-03-13')]   
st_dat = st_dat[~(st_dat['Date'] > '2026-09-05')] 
st_dat = st_dat.copy()

nt_dat["Sonstige"] = nt_dat["Sonstige"] + nt_dat["FW"]


nt_dat = nt_dat.drop(columns="FW")

nt_dat = nt_dat.rename(columns={"CDU/CSU": "CDU"})

party_cols = ["CDU", "SPD", "GRÜNE", "FDP", "LINKE", "AfD", "BSW", "Sonstige"]

def prepare_stan_inputs(df, party_cols, election_date, T, eps=1e-6, seed=None):
    df = df.copy()
    rng = np.random.default_rng(seed)

    # --- week assignment, anchored to a shared election_date/T across datasets ---
    df["Date"] = pd.to_datetime(df["Date"])
    weeks_before = (election_date - df["Date"]).dt.days // 7
    df["week"] = T - weeks_before
    df = df[df["week"].between(1, T)].copy()
    df["week"] = df["week"].astype(np.int32)

    # --- institute index ---
    df["institute_id"] = df["institute"].astype("category").cat.codes + 1

    # --- party shares -> simplex, zero-handling, renormalize ---
    y = df[party_cols].to_numpy(dtype=float) / 100
    mask = y <= 0
    if mask.any():
        y[mask] = rng.uniform(eps, 10 * eps, size=mask.sum())
    y = y / y.sum(axis=1, keepdims=True)

    df["N Polled"] = df["N Polled"].astype(int)

    return df, y


election_date = pd.Timestamp("2026-09-06")
T = 260


nt_dat, y_nat = prepare_stan_inputs(nt_dat, party_cols, election_date, T, seed=1)
st_dat, y_state = prepare_stan_inputs(st_dat, party_cols, election_date, T, seed=2)

def alr_transform(shares, eps=1e-6):
    """Additive log-ratio transform, last element as reference (fixed at 0)."""
    shares = np.asarray(shares, dtype=float)
    shares = np.clip(shares, eps, None)   # guard against exact zeros
    shares = shares / shares.sum()        # ensure it's a proper simplex
    return np.log(shares[:-1] / shares[-1])


# results in % or raw votes, same order as party_cols
last_election_nat = [24.7, 25.1, 14.7, 11.4, 4.9, 10.4, 0.0, 8.4]
last_election_state = [37.1, 8.4, 5.9, 9.4, 11, 20.8, 0.0, 10.3]

eta_nat_last = alr_transform(last_election_nat)
eta_state_last = alr_transform(last_election_state)

prior_delta0_mean = (eta_state_last - eta_nat_last).tolist()

stan_data = {
    "N_nat": len(nt_dat),
    "P": y_nat.shape[1],
    "T": T,
    "J_nat": int(nt_dat["institute_id"].max()),
    "sample_nat": nt_dat["N Polled"].tolist(),
    "week_nat": nt_dat["week"].tolist(),
    "institute_nat": nt_dat["institute_id"].tolist(),
    "y_nat": y_nat.tolist(),

    "N_state": len(st_dat),
    "J_state": int(st_dat["institute_id"].max()),
    "sample_state": st_dat["N Polled"].tolist(),
    "week_state": st_dat["week"].tolist(),
    "institute_state": st_dat["institute_id"].tolist(),
    "y_state": y_state.tolist(),

    "prior_delta0_mean": prior_delta0_mean,  # (P-1)-length list, see note below
}

from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file="State_level_dirichlet.stan")

P = y_nat.shape[1]
J = int(nt_dat["institute_id"].max())
K = int(st_dat["institute_id"].max())


def init_fun():
    return {
        "eta0": np.zeros(P-1),
        "z_rw": np.zeros((T-1, P-1)),
        "sigma_rw": np.repeat(0.02, P-1),
        "sigma_house": np.repeat(0.1, P-1),
        #"sigma_mode": np.repeat(0.1, P-1),
        "house_raw": np.zeros((J-1, P-1)),
        #"mode_rarw": np.zeros((M-1, P-1)),
        "phi": np.repeat(500, J),

        "delta0": np.zeros(P-1),
        "z_delta": np.zeros((T-1, P-1)),
        "sigma_house_state": np.repeat(0.1, P-1),
        "house_state_raw": np.zeros((K-1, P-1))
    }

t0 = time.time()
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    output_dir='output',
    show_progress=True,
    inits=init_fun()         # <-- this was missing
)
elapsed = time.time() - t0
print(f"{elapsed:.1f}s")

draws = fit.draws()
np.save("draws.npy", draws)

# Save all posterior draws as a pandas DataFrame
draws_df = fit.draws_pd()

draws_df.to_csv(
    "posterior_draws.csv",
    index=False
)

print("Posterior draws saved.")
print("Shape:", draws_df.shape)

# Extract election-day state predictions
state_election_cols = [
    f"election_day_prediction_state[{i}]"
    for i in range(1, P + 1)
]

state_election_draws = draws_df[state_election_cols]

# Calculate posterior mean
state_election_mean = state_election_draws.mean(axis=0) * 100

# Put into a nice table
state_prediction = pd.DataFrame({
    "Party": party_cols,
    "Mean prediction (%)": state_election_mean.values
})

print("\nElection-day state prediction:")
print(state_prediction.to_string(index=False))
