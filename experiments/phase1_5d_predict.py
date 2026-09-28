"""Phase 1.5D condition-level ablations and held-out robustness tests."""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vetm.interventions import intervention_registry
from vetm.metrics import evaluate_predictions
from vetm.validators import NearestNeighborValidator, MLPValidityPredictor

def rankdata(x):
    order=np.argsort(x,kind="mergesort"); out=np.empty(len(x),float); sx=x[order]; i=0
    while i<len(x):
        j=i+1
        while j<len(x) and sx[j]==sx[i]: j+=1
        out[order[i:j]]=(i+j-1)/2; i=j
    return out
def spearman(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def ridge(x,y,z,alpha=1.):
    mu,sc=x.mean(0),x.std(0); sc[sc<1e-12]=1
    xn=(x-mu)/sc; zn=(z-mu)/sc; center=y.mean()
    w=np.linalg.solve(xn.T@xn+alpha*np.eye(x.shape[1]),xn.T@(y-center))
    return center+zn@w
def mlp_reg(x,y,z,seed=0):
    mu,sc=x.mean(0),x.std(0); sc[sc<1e-12]=1
    xn=(x-mu)/sc; zn=(z-mu)/sc; ym,ys=y.mean(),max(y.std(),1e-12); yn=(y-ym)/ys
    rng=np.random.default_rng(seed); w1=rng.normal(0,.1,(xn.shape[1],16)); b1=np.zeros(16); w2=rng.normal(0,.1,16); b2=0.
    for _ in range(400):
        h=np.tanh(xn@w1+b1); e=h@w2+b2-yn; gw2=h.T@e/len(xn)+.01*w2; gb2=e.mean()
        gh=np.outer(e,w2)*(1-h*h); gw1=xn.T@gh/len(xn)+.01*w1; gb1=gh.mean(0)
        w2-=.01*gw2; b2-=.01*gb2; w1-=.01*gw1; b1-=.01*gb1
    return (np.tanh(zn@w1+b1)@w2+b2)*ys+ym
def descriptors(rows):
    reg={i.intervention_id:i for i in intervention_registry()}; result=[]
    for r in rows:
        item=reg[r["intervention_id"]]; desc=item.descriptor(__import__("vetm.nsga2",fromlist=["NSGA2Config"]).NSGA2Config(population_size=50,generations=int(r["budget"])//50-1),int(r["n_var"]))
        result.append([float(desc["category"]==c) for c in ("mutation","crossover","selection")]+[float(desc["parameter"]==p) for p in ("mutation_probability","eta_m","crossover_probability","eta_c","tournament_size")]+[desc["absolute_value"],desc["signed_change"],desc["relative_magnitude"]])
    return np.asarray(result,float)
def bootstrap_delta(diffs,seed=0,n=5000):
    rng=np.random.default_rng(seed); x=np.asarray(diffs,float); draws=x[rng.integers(0,len(x),size=(n,len(x)))].mean(1)
    return {"mean":float(x.mean()),"ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),"n_tasks":len(x)}
def main():
    results=ROOT/"results"; rows=list(csv.DictReader((results/"phase1_5C_R1_condition_summary.csv").open(encoding="utf-8"))); ctx={r["task_id"]:r for r in csv.DictReader((results/"phase1_5D_task_context.csv").open(encoding="utf-8"))}; static={r["task_id"]:r for r in csv.DictReader((results/"phase1_5D_static_features.csv").open(encoding="utf-8"))}
    # one aggregate row per condition is the sole learning unit
    rows=[r for r in rows if r["task_id"] in ctx]
    dyn_keys=[k for k in ctx[next(iter(ctx))] if k not in {"task_id","problem"}]
    stat_keys=[k for k in static[next(iter(static))] if k not in {"task_id"}]
    inter=descriptors(rows)
    task_static=np.asarray([[float(static[r["task_id"]][k]) for k in stat_keys] for r in rows],float)
    task_dynamic=np.asarray([[float(ctx[r["task_id"]][k]) for k in dyn_keys] for r in rows],float)
    y=np.asarray([float(r["mean"]) for r in rows],float)
    problems=sorted({r["problem"] for r in rows}); split_results={}; task_diffs=[]
    for problem in problems:
        train=np.asarray([i for i,r in enumerate(rows) if r["problem"]!=problem]); test=np.asarray([i for i,r in enumerate(rows) if r["problem"]==problem])
        if not len(train) or not len(test): continue
        groups={"intervention_only":inter,"static_intervention":np.c_[task_static,inter],"dynamic_intervention":np.c_[task_dynamic,inter],"static_dynamic_intervention":np.c_[task_static,task_dynamic,inter]}
        split_results[problem]={}
        for name,xall in groups.items():
            pred=ridge(xall[train],y[train],xall[test]); split_results[problem][name]={"mae":float(np.mean(abs(pred-y[test]))),"spearman":spearman(y[test],pred)}
            if name=="static_dynamic_intervention":
                task_diffs.append(float(np.mean(abs(pred-y[test]))-np.mean(abs(y[test]))))
    # context shuffled control on the same leave-one-problem-out protocol
    rng=np.random.default_rng(7); shuffled=[]
    for _ in range(200):
        diffs=[]
        for problem in problems:
            train=np.asarray([i for i,r in enumerate(rows) if r["problem"]!=problem]); test=np.asarray([i for i,r in enumerate(rows) if r["problem"]==problem])
            if not len(train) or not len(test): continue
            perm=rng.permutation(train); xtrain=np.c_[task_static[train],task_dynamic[perm],inter[train]]
            pred=ridge(xtrain,y[train],np.c_[task_static[test],task_dynamic[test],inter[test]])
            diffs.append(float(np.mean(abs(pred-y[test]))))
        if diffs: shuffled.append(float(np.mean(diffs)))
    primary={"model":"static_dynamic_intervention","leave_problem_out":split_results,"macro_task_delta_mae":bootstrap_delta(task_diffs),"shuffled_context_mae_mean":float(np.mean(shuffled)) if shuffled else None,"shuffled_context_mae_q95":float(np.quantile(shuffled,.95)) if shuffled else None,"zero_gain_reference":"mean(abs(target_delta))","dynamic_features":dyn_keys,"static_features":stat_keys}
    # family held-out negative transfer detection with standardized kNN and MLP
    train=[i for i,r in enumerate(rows) if r["family"]=="ZDT"]; test=[i for i,r in enumerate(rows) if r["family"]=="DTLZ"]
    if train and test:
        x=np.c_[task_static,task_dynamic,inter]; yneg=(y<-.01).astype(int)
        nn=NearestNeighborValidator(k=5).fit(x[train],yneg[train]); mlp=MLPValidityPredictor(hidden_dim=16,epochs=300,learning_rate=.01,seed=0).fit(x[train],yneg[train])
        detection={"nearest_neighbor":evaluate_predictions(yneg[test],nn.predict_proba(x[test])),"mlp":evaluate_predictions(yneg[test],mlp.predict_proba(x[test])),"prevalence":float(yneg[test].mean())}
    else: detection={"available":False}
    output={"primary":primary,"family_held_out_negative_transfer":detection,"condition_count":len(rows),"task_count":len(problems)}
    (results/"phase1_5D_predictability.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    (results/"phase1_5D_bootstrap.json").write_text(json.dumps({"task_block_bootstrap":primary["macro_task_delta_mae"]},indent=2),encoding="utf-8")
    (results/"phase1_5D_permutation.json").write_text(json.dumps({"n_permutations":200,"real_mae_delta":primary["macro_task_delta_mae"]["mean"],"shuffled_mae_mean":primary["shuffled_context_mae_mean"],"shuffled_mae_q95":primary["shuffled_context_mae_q95"]},indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))
if __name__=="__main__": main()
