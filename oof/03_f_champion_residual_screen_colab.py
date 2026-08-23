"""정확한 자체 champion OOF 위에서 F residual 채널을 strict-forward 검증한다."""
from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT=Path(__file__).resolve().parents[1]; FEATURE=ROOT/"common"/"model_features.py"
SEEDS=(17,42,777); SCALES=(0.005,0.01,0.025,0.05,0.075)
CONFIGS=((2,100,300),(3,100,300),(3,500,500))

def module():
    s=importlib.util.spec_from_file_location("f_residual_features",FEATURE)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def read(path):
    a=np.load(path,allow_pickle=True); return {k:a[k] for k in a.files}
def bss(y,p):
    y=np.asarray(y,float);p=np.clip(np.asarray(p,float),1e-6,1-1e-6);r=y.mean()
    return float(1e5*(1-np.mean((y-p)**2)/(r*(1-r))))
def boot(ids,y,b,c,seed,n=500):
    g=(np.asarray(y)-np.asarray(b))**2-(np.asarray(y)-np.asarray(c))**2
    t=pd.DataFrame({"id":pd.Series(ids).astype(str),"g":g,"n":1}).groupby("id",observed=True).sum().to_numpy(float)
    rng=np.random.default_rng(seed);return float(np.mean([t[(z:=rng.integers(0,len(t),len(t))),0].sum()/t[z,1].sum()>0 for _ in range(n)]))
def params(seed,depth,l2,iters,task):
    d=dict(iterations=iters,depth=depth,learning_rate=.025,loss_function="RMSE",l2_leaf_reg=l2,
      random_strength=.2,bootstrap_type="Bernoulli",subsample=.8,random_seed=seed,
      allow_writing_files=False,verbose=False)
    d.update(task_type="GPU",devices="0") if task=="GPU" else d.update(thread_count=-1)
    return d

def main(train_path,oof_dir,output,task):
    frame=pd.read_csv(train_path,encoding="utf-8-sig",low_memory=False); fm=module()
    prior=float(frame.loc[frame.season.astype(int)<2022,"control_success"].mean())
    x=fm.engineer(frame.drop(columns=["row_id","control_success"]),prior)
    for c in fm.CAT_COLS:x[c]=x[c].astype(str)
    cats=[x.columns.get_loc(c) for c in fm.CAT_COLS]; season=frame.season.astype(int).to_numpy()
    assets={2022:read(oof_dir/"own_f_base_oof_2022.npz"),
      2023:read(oof_dir/"own_champion_oof_2023.npz"),2024:read(oof_dir/"own_champion_oof_2024.npz")}
    folds=[]
    for year in (2023,2024):
        train_years=list(range(2022,year)); tx=[];ty=[];tw=[]
        for sy in train_years:
            rows=frame.loc[season==sy].reset_index(drop=True);a=assets[sy]
            if not np.array_equal(rows.row_id.astype(str),a["row_id"].astype(str)):raise ValueError(f"{sy} 정렬 불일치")
            mask=rows.game_type.astype(str).eq("F").to_numpy(); base=a["p_f_general6"] if sy==2022 else a["p_champion"]
            tx.append(x.loc[(season==sy)&frame.game_type.astype(str).eq("F")]);ty.append(a["target"][mask]-base[mask])
            tw.append(np.full(mask.sum(),.55**((year-1)-sy)))
        vx=x.loc[(season==year)&frame.game_type.astype(str).eq("F")];vr=frame.loc[season==year].reset_index(drop=True)
        va=assets[year];vm=vr.game_type.astype(str).eq("F").to_numpy();yb=va["target"].astype(float);base=va["p_champion"].astype(float)
        train_pool=Pool(pd.concat(tx),np.concatenate(ty),weight=np.concatenate(tw),cat_features=cats);valid_pool=Pool(vx,cat_features=cats)
        candidates=[]
        for depth,l2,iters in CONFIGS:
            members=[];secs=[]
            for seed in SEEDS:
                m=CatBoostRegressor(**params(seed,depth,l2,iters,task));t=time.perf_counter();m.fit(train_pool)
                secs.append(time.perf_counter()-t);members.append(m.predict(valid_pool));print(f"fold={year} d={depth} l2={l2} seed={seed} sec={secs[-1]:.1f}",flush=True)
            corr=np.mean(members,axis=0)
            for scale in SCALES:
                p=base.copy();p[vm]=np.clip(p[vm]+scale*corr,1e-6,1-1e-6)
                candidates.append({"depth":depth,"l2":l2,"iterations":iters,"scale":scale,
                  "overall_delta":bss(yb,p)-bss(yb,base),"f_delta":bss(yb[vm],p[vm])-bss(yb[vm],base[vm]),
                  "bootstrap":boot(vr.loc[vm,"pitcher_id"].to_numpy(),yb[vm],base[vm],p[vm],825000+year+int(scale*1000)),"seconds":secs})
        folds.append({"year":year,"candidates":candidates});output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({"status":"running","folds":folds},ensure_ascii=False,indent=2)+"\n")
    summaries=[]
    for depth,l2,iters in CONFIGS:
      for scale in SCALES:
        r=[next(q for q in f["candidates"] if q["depth"]==depth and q["l2"]==l2 and q["scale"]==scale) for f in folds]
        passed=min(q["overall_delta"] for q in r)>0 and min(q["f_delta"] for q in r)>0 and min(q["bootstrap"] for q in r)>=.8
        summaries.append({"depth":depth,"l2":l2,"iterations":iters,"scale":scale,
          "fold_2023_delta":r[0]["overall_delta"],"fold_2024_delta":r[1]["overall_delta"],
          "fold_2023_f_delta":r[0]["f_delta"],"fold_2024_f_delta":r[1]["f_delta"],
          "minimum_bootstrap":min(q["bootstrap"] for q in r),"passed":bool(passed)})
    summaries.sort(key=lambda z:(z["passed"],min(z["fold_2023_delta"],z["fold_2024_delta"])),reverse=True)
    selected=next((z for z in summaries if z["passed"]),None);report={"selected":selected,"top":summaries[:10],"folds":folds,
      "decision":"continue_own_target_oof" if selected else "keep_own_champion_oof"}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");print(json.dumps({"selected":selected,"top":summaries[:10],"decision":report["decision"]},ensure_ascii=False,indent=2))

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--train",type=Path,required=True);p.add_argument("--oof-dir",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--task-type",choices=("CPU","GPU"),default="GPU");a=p.parse_args()
 main(a.train.resolve(),a.oof_dir.resolve(),a.output.resolve(),a.task_type)
