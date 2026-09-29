"""Phase 1.5E oracle-free causal fingerprint predictor and controls."""
from __future__ import annotations
import csv,json
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
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def standardize(x,z):
    mu,sc=x.mean(0),x.std(0); sc[sc<1e-12]=1
    return (x-mu)/sc,(z-mu)/sc
def ridge(x,y,z):
    mu,sc=x.mean(0),x.std(0);sc[sc<1e-12]=1
    xn=(x-mu)/sc;zn=(z-mu)/sc;c=y.mean()
    return c+zn@np.linalg.solve(xn.T@xn+np.eye(x.shape[1]),xn.T@(y-c))
def logistic(x,y,z,epochs=500):
    xn,zn=standardize(x,z); xn=np.c_[np.ones(len(xn)),xn]; zn=np.c_[np.ones(len(zn)),zn]; w=np.zeros(xn.shape[1])
    for _ in range(epochs):
        p=1/(1+np.exp(-np.clip(xn@w,-30,30))); g=xn.T@(p-y)/len(xn); g[1:]+=.01*w[1:]; w-=.08*g
    return 1/(1+np.exp(-np.clip(zn@w,-30,30)))
def task_bootstrap(diffs,n=5000):
    x=np.asarray(diffs,float);rng=np.random.default_rng(42);draw=x[rng.integers(0,len(x),size=(n,len(x)))].mean(1)
    return {"mean":float(x.mean()),"ci95_low":float(np.quantile(draw,.025)),"ci95_high":float(np.quantile(draw,.975)),"n_tasks":len(x)}
def main():
    results=ROOT/"results"
    rows=list(csv.DictReader((results/"phase1_5C_R1_condition_summary.csv").open(encoding="utf-8")))
    context={r["task_id"]:r for r in csv.DictReader((results/"phase1_5E_R1_causal_fingerprint.csv").open(encoding="utf-8"))}
    static={r["task_id"]:r for r in csv.DictReader((results/"phase1_5E_oracle_free_static.csv").open(encoding="utf-8"))}
    registry={i.intervention_id:i for i in intervention_registry()}
    ckeys=[k for k in context[next(iter(context))] if k not in {"task_id","problem","n_var","n_obj","budget","probe_seeds"}]
    skeys=[k for k in static[next(iter(static))] if k not in {"task_id"}]
    def int_feat(row):
        item=registry[row["intervention_id"]]; desc=item.descriptor(__import__("vetm.nsga2",fromlist=["NSGA2Config"]).NSGA2Config(population_size=50,generations=int(row["budget"])//50-1),int(row["n_var"]))
        return [float(desc["category"]==c) for c in ("mutation","crossover","selection")]+[float(desc["parameter"]==p) for p in ("mutation_probability","eta_m","crossover_probability","eta_c","tournament_size")]+[desc["absolute_value"],desc["signed_change"],desc["relative_magnitude"]]
    inter=np.asarray([int_feat(r) for r in rows],float)
    passive=np.asarray([[float(static[r["task_id"]][k]) for k in skeys] for r in rows],float)
    causal=np.asarray([[float(context[r["task_id"]][k]) for k in ckeys] for r in rows],float)
    y=np.asarray([float(r["mean"]) for r in rows],float)
    problems=sorted({r["problem"] for r in rows})
    groups={"intervention_only":inter,"passive_intervention":np.c_[passive,inter],"causal_intervention":np.c_[causal,inter],"passive_causal_intervention":np.c_[passive,causal,inter]}
    split_results={}; predictions={name:np.full(len(rows),np.nan) for name in groups}
    predictions["zero_gain"]=np.zeros(len(rows)); predictions["global_intervention_mean"]=np.full(len(rows),np.nan)
    for problem in problems:
        train=np.asarray([i for i,r in enumerate(rows) if r["problem"]!=problem]);test=np.asarray([i for i,r in enumerate(rows) if r["problem"]==problem])
        split_results[problem]={}
        train_global=defaultdict(list)
        for i in train: train_global[rows[i]["intervention_id"]].append(y[i])
        predictions["global_intervention_mean"][test]=[
            np.mean(train_global[rows[i]["intervention_id"]]) if train_global[rows[i]["intervention_id"]] else y[train].mean()
            for i in test
        ]
        for name,x in groups.items():
            pred=ridge(x[train],y[train],x[test])
            predictions[name][test]=pred
            split_results[problem][name]={"mae":float(np.mean(abs(pred-y[test]))),"spearman":spearman(y[test],pred)}
    negative_results={}
    negative_label=(y < -0.01).astype(int)
    for problem in problems:
        train=np.asarray([i for i,r in enumerate(rows) if r["problem"]!=problem]); test=np.asarray([i for i,r in enumerate(rows) if r["problem"]==problem])
        x=groups["causal_intervention"]
        nn=NearestNeighborValidator(k=5).fit(x[train],negative_label[train]).predict_proba(x[test])
        logit=logistic(x[train],negative_label[train],x[test])
        mlps=[MLPValidityPredictor(hidden_dim=16,epochs=300,learning_rate=.01,seed=seed).fit(x[train],negative_label[train]).predict_proba(x[test]) for seed in range(5)]
        negative_results[problem]={
            "prevalence":float(negative_label[test].mean()),
            "nearest_neighbor":evaluate_predictions(negative_label[test],nn),
            "logistic":evaluate_predictions(negative_label[test],logit),
            "mlp_mean":{key:float(np.mean([evaluate_predictions(negative_label[test],p)[key] for p in mlps])) for key in ("accuracy","precision","recall","f1","brier")},
            "mlp_seeds":5,
        }
    # Context shuffled by complete task vector among training tasks for 1000 permutations.
    rng=np.random.default_rng(7); shuffled=[]; causal_all=groups["causal_intervention"]
    for _ in range(1000):
        errors=[]
        for problem in problems:
            train=np.asarray([i for i,r in enumerate(rows) if r["problem"]!=problem]);test=np.asarray([i for i,r in enumerate(rows) if r["problem"]==problem])
            task_ids=list(dict.fromkeys(rows[i]["task_id"] for i in train)); perm=rng.permutation(task_ids); mapping=dict(zip(task_ids,perm))
            index={task:[] for task in task_ids}
            for i in train:index[rows[i]["task_id"]].append(i)
            shuffled_train=causal_all[train].copy()
            for task,idxs in index.items():
                source=mapping[task]; source_idxs=index[source]
                shuffled_train[[list(train).index(j) for j in idxs],:causal.shape[1]]=causal[source_idxs[0]]
            pred=ridge(shuffled_train,y[train],causal_all[test])
            errors.extend(abs(pred-y[test]).tolist())
        shuffled.append(np.mean(errors))
    real_mae=float(np.mean(abs(predictions["causal_intervention"]-y)))
    perm_p=float(np.mean(np.asarray(shuffled)<=real_mae))
    task_ids=sorted({r["task_id"] for r in rows})
    task_errors={}
    task_spearman=[]
    for task_id in task_ids:
        idx=np.asarray([i for i,r in enumerate(rows) if r["task_id"]==task_id])
        task_errors[task_id]={name:float(np.mean(abs(pred[idx]-y[idx]))) for name,pred in predictions.items()}
        correlation=spearman(y[idx],predictions["causal_intervention"][idx])
        if correlation is not None: task_spearman.append(correlation)
    macro_mae={name:float(np.mean([error[name] for error in task_errors.values()])) for name in predictions}
    bootstrap={
        baseline:task_bootstrap([error["causal_intervention"]-error[baseline] for error in task_errors.values()])
        for baseline in ("zero_gain","intervention_only","global_intervention_mean")
    }
    macro_spearman=float(np.mean(task_spearman)) if task_spearman else None
    primary_gate=bool(
        macro_mae["causal_intervention"]<macro_mae["zero_gain"]
        and macro_mae["causal_intervention"]<macro_mae["intervention_only"]
        and macro_mae["causal_intervention"]<macro_mae["global_intervention_mean"]
        and bootstrap["zero_gain"]["ci95_high"]<0
        and bootstrap["intervention_only"]["ci95_high"]<0
        and macro_spearman is not None and macro_spearman>.30
        and real_mae<float(np.mean(shuffled)) and perm_p<.05
    )
    summary={"condition_count":len(rows),"task_configuration_blocks":len(task_ids),"primary":"causal_intervention_ridge",
             "leave_problem_out":split_results,"per_task_configuration_mae":task_errors,
             "macro_task_mae":macro_mae,"macro_task_spearman_causal":macro_spearman,
             "task_block_bootstrap":bootstrap,
             "shuffled_context_mae_mean":float(np.mean(shuffled)),
             "shuffled_context_mae_q95":float(np.quantile(shuffled,.95)),
             "permutation_p_value":perm_p,"primary_gate":primary_gate,
             "causal_feature_columns":ckeys,"static_feature_columns":skeys}
    summary["negative_transfer_detection"]=negative_results
    (results/"phase1_5E_R1_predictability.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (results/"phase1_5E_R1_bootstrap.json").write_text(json.dumps(bootstrap,indent=2),encoding="utf-8")
    (results/"phase1_5E_R1_permutation.json").write_text(json.dumps({"n_permutations":1000,"p_value":perm_p,"real_mae":real_mae,"shuffled_mean":float(np.mean(shuffled))},indent=2),encoding="utf-8")
    (results/"phase1_5E_R1_summary.json").write_text(json.dumps({"primary_gate":primary_gate,"phase2":"GO" if primary_gate else "NO-GO","macro_task_mae":macro_mae,"task_block_bootstrap":bootstrap,"macro_task_spearman":macro_spearman,"permutation_p":perm_p},indent=2),encoding="utf-8")
    print(json.dumps({"primary_gate":primary_gate,"macro_task_mae":macro_mae,"bootstrap":bootstrap,"spearman":macro_spearman,"p":perm_p},indent=2))
if __name__=="__main__": main()
