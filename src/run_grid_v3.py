import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, pipeline as P, experiment as E
tag = sys.argv[1]; margin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
b15 = P.load_15m(); out = []; os.makedirs(f"results/{tag}", exist_ok=True)
for bar, hor in P.GRID:
    res, pr, _ = E.run_config_v3(b15, bar, hor, margin_bps=margin)
    res.to_csv(f"results/{tag}/folds_{bar}_{hor}.csv", index=False); pr.to_parquet(f"results/{tag}/preds_{bar}_{hor}.parquet")
    s = res.drop(columns="fold").agg({c: ("sum" if c.endswith(("_n", "total")) or c == "n_test" else "mean") for c in res.columns if c != "fold"}).to_dict()
    s.update(bar=bar, hor=hor, lgb_pos_folds=int((res.lgb_total > 0).sum()), lr_pos_folds=int((res.lr_total > 0).sum()), folds=len(res))
    out.append(s)
    print(bar, hor, " ".join(f"{k}={s[k]:.3g}" for k in ["auc_lgb","ev_corr_lgb","ev_corr_lr","every_ev_bps","lgb_n","lgb_ev_bps","lgb_sharpe","lgb_pos_folds","lr_n","lr_ev_bps","lr_sharpe","rand_ev_bps","rand_sharpe"]), flush=True)
pd.DataFrame(out).to_csv(f"results/{tag}/grid_summary.csv", index=False)
