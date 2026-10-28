"""Run the full 8-config grid for a universe; save per-fold and summary tables."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, pipeline as P, experiment as E
tag = sys.argv[1]; universe = sys.argv[2] if len(sys.argv) > 2 else "setups"
b15 = P.load_15m(); out = []; os.makedirs(f"results/{tag}", exist_ok=True)
for bar, hor in P.GRID:
    t = time.time()
    res, pr, _ = E.run_config(b15, bar, hor, universe=universe, n_random=10)
    res.to_csv(f"results/{tag}/folds_{bar}_{hor}.csv", index=False)
    s = E.summarize(res); s.update(bar=bar, hor=hor, secs=round(time.time() - t)); out.append(s)
    print(bar, hor, {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items() if k in
          ("n_test","base_rate","auc_lgb","auc_lr","every_ev_bps","lgb_n","lgb_ev_bps","lgb_sharpe","rand_ev_bps","lgb_folds_positive","n_folds","secs")}, flush=True)
pd.DataFrame(out).to_csv(f"results/{tag}/grid_summary.csv", index=False)
