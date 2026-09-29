"""生成 Phase 1.5E oracle-free one-step causal fingerprints。"""
from __future__ import annotations
import csv
import json
import io
import subprocess
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vetm.interventions import intervention_registry
from vetm.nsga2 import NSGA2Config, crowding_distance, fast_non_dominated_sort, run_nsga2
from vetm.problems import get_problem

PANEL=[
    ("mutation_half", {"mutation_probability":"half_default"}),
    ("mutation_double", {"mutation_probability":"double_default"}),
    ("mutation_eta_7", {"eta_m":7.0}),
    ("mutation_eta_30", {"eta_m":30.0}),
    ("crossover_prob_08", {"crossover_probability":.8}),
    ("crossover_eta_12", {"eta_c":12.0}),
    ("crossover_eta_30", {"eta_c":30.0}),
    ("selection_tournament_6", {"tournament_size":6}),
]

def front_indices(objectives):
    return fast_non_dominated_sort(objectives)[0]

def probe_actions(n_var, registry):
    """保持固定锚点；与目标干预数值碰撞的锚点在该任务上禁用。"""
    active, excluded = [], []
    for anchor, parameters in PANEL:
        concrete = dict(parameters)
        if concrete.get("mutation_probability") == "half_default":
            concrete["mutation_probability"] = 0.5 / n_var
        if concrete.get("mutation_probability") == "double_default":
            concrete["mutation_probability"] = 2.0 / n_var
        collision = any(
            key in intervention.parameters
            and np.isclose(float(intervention.parameters[key]), float(value))
            for intervention in registry for key, value in concrete.items()
        )
        if collision:
            excluded.append(anchor)
        else:
            active.append((anchor, concrete))
    return active, excluded

def finite_crowding(objectives):
    distances = crowding_distance(objectives, np.arange(len(objectives)))
    finite = distances[np.isfinite(distances)]
    return float(finite.mean()) if len(finite) else 0.0

def offspring_survival_fraction(initial, final):
    initial_rows = {row.tobytes() for row in np.asarray(initial)}
    return float(np.mean([row.tobytes() not in initial_rows for row in np.asarray(final)]))

def rankdata(values):
    order=np.argsort(values,kind="mergesort"); ranks=np.empty(len(values),float); sorted_values=values[order]; i=0
    while i<len(values):
        j=i+1
        while j<len(values) and sorted_values[j]==sorted_values[i]: j+=1
        ranks[order[i:j]]=(i+j-1)/2; i=j
    return ranks

def correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None

def empirical_hv(objectives, center, scale, reference):
    normalized=(objectives-center)/scale
    from vetm.transfer_matrix import unit_hypervolume
    return unit_hypervolume(normalized, reference)

def dominance_rate(a,b):
    wins=0; total=max(1,len(a)*len(b))
    for x in a:
        wins += int(np.sum(np.all(x[None,:] <= b,axis=1) & np.any(x[None,:] < b,axis=1)))
    return float(wins/total)

def main():
    rows=list(csv.DictReader((ROOT/"results/phase1_5C_R1_transfer_matrix.csv").open(encoding="utf-8")))
    task_meta={}
    for row in rows:
        task_meta[row["task_id"]]={"task_id":row["task_id"],"problem":row["problem"],"n_var":int(row["n_var"]),"n_obj":int(row["n_obj"]),"budget":int(row["budget"])}
    registry=intervention_registry()
    target_values=[(item.parameters.get("mutation_probability"),"mutation_probability") for item in registry]
    feature_rows=[]; seed_rows=[]; static_rows=[]; cost_rows=[]
    for task in task_meta.values():
        problem=get_problem(task["problem"],n_var=task["n_var"],n_obj=task["n_obj"])
        cfg=NSGA2Config(population_size=50,generations=1)
        active, excluded = probe_actions(problem.n_var, registry)
        per_seed=[]
        for seed in (200,201):
            rng=np.random.default_rng(seed)
            initial=rng.uniform(problem.lower,problem.upper,size=(50,problem.n_var))
            initial_objectives=problem.evaluate(initial)
            center=np.median(initial_objectives,axis=0)
            scale=np.maximum(np.percentile(initial_objectives,75,axis=0)-np.percentile(initial_objectives,25,axis=0),1e-12)
            reference=np.max((initial_objectives-center)/scale,axis=0)+1.1
            baseline=run_nsga2(problem,cfg,seed=seed,initial_population=initial,initial_objectives=initial_objectives)
            base_front=baseline["front"]
            base_nd=len(base_front)/50
            base_norm_front=(base_front-center)/scale
            base_spread=float(np.mean(np.std(base_norm_front,axis=0)))
            base_mean=float(np.mean(base_norm_front))
            base_crowd=finite_crowding(baseline["objectives"])
            base_hv=empirical_hv(base_front,center,scale,reference)
            base_survival=offspring_survival_fraction(initial, baseline["population"])
            seed_features={}
            for anchor,p in active:
                branch=run_nsga2(problem,NSGA2Config(population_size=50,generations=1,**p),seed=seed,initial_population=initial,initial_objectives=initial_objectives)
                front=branch["front"]; nd=len(front)/50
                norm_front=(front-center)/scale
                spread=float(np.mean(np.std(norm_front,axis=0)))
                crowd=finite_crowding(branch["objectives"])
                hv=empirical_hv(front,center,scale,reference)
                improvement=float(np.mean(np.sum(norm_front,axis=1))-base_mean)
                seed_features[anchor]={"delta_nd_fraction":nd-base_nd,"dominance_win_rate":dominance_rate(front,base_front)-dominance_rate(base_front,front),"delta_objective_spread":spread-base_spread,"delta_crowding":crowd-base_crowd,"delta_empirical_hv":hv-base_hv,"delta_empirical_mean":improvement,"delta_offspring_survival":offspring_survival_fraction(initial, branch["population"])-base_survival}
            per_seed.append(seed_features)
            seed_row={"task_id":task["task_id"],"probe_seed":seed}
            for anchor,features in seed_features.items():
                if features is None:
                    seed_row[anchor+"_available"]=0
                else:
                    seed_row[anchor+"_available"]=1
                    seed_row.update({anchor+"_"+key:value for key,value in features.items()})
            seed_rows.append(seed_row)
        aggregate={"task_id":task["task_id"],"problem":task["problem"],"n_var":task["n_var"],"n_obj":task["n_obj"],"budget":task["budget"],"probe_seeds":2}
        for anchor,_ in PANEL:
            values=[seed[anchor] for seed in per_seed if anchor in seed]
            if not values:
                aggregate[anchor+"_available"]=0
                for key in ("delta_nd_fraction","dominance_win_rate","delta_objective_spread","delta_crowding","delta_empirical_hv","delta_empirical_mean","delta_offspring_survival"):
                    aggregate[f"{anchor}_{key}_mean"]=0.0
                    aggregate[f"{anchor}_{key}_std"]=0.0
                continue
            aggregate[anchor+"_available"]=1
            for key in values[0]:
                aggregate[f"{anchor}_{key}_mean"]=float(np.mean([v[key] for v in values]))
                aggregate[f"{anchor}_{key}_std"]=float(np.std([v[key] for v in values],ddof=1))
        feature_rows.append(aggregate)
        lower=np.asarray(problem.lower); upper=np.asarray(problem.upper)
        static_rows.append({"task_id":task["task_id"],"n_var":task["n_var"],"n_obj":task["n_obj"],"budget":task["budget"],"lower_min":float(lower.min()),"lower_max":float(lower.max()),"lower_mean":float(lower.mean()),"lower_std":float(lower.std()),"upper_min":float(upper.min()),"upper_max":float(upper.max()),"upper_mean":float(upper.mean()),"upper_std":float(upper.std()),"bound_width_mean":float((upper-lower).mean()),"bound_width_std":float((upper-lower).std())})
        anchor_count=len(active)
        total_cost=2*50*(2+anchor_count)
        cost_rows.append({"task_id":task["task_id"],"probe_seeds":2,"initial_population_evaluations":100,"baseline_one_step_evaluations":100,"anchor_count":anchor_count,"excluded_anchor_ids":excluded,"probe_anchor_evaluations":100*anchor_count,"total_probe_evaluations":total_cost,"calibrated_budget":task["budget"],"probe_fraction":float(total_cost/task["budget"])})
    for name,data in [("phase1_5E_R1_causal_fingerprint.csv",feature_rows),("phase1_5E_causal_fingerprint.csv",feature_rows),("phase1_5E_oracle_free_static.csv",static_rows),("phase1_5E_R1_fingerprint_by_seed.csv",seed_rows),("phase1_5E_probe_cost.json",cost_rows),("phase1_5E_R1_probe_cost.json",cost_rows)]:
        if name.endswith(".json"):
            (ROOT/"results"/name).write_text(json.dumps({"tasks":data},indent=2),encoding="utf-8")
        else:
            with (ROOT/"results"/name).open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=sorted({key for row in data for key in row})); w.writeheader(); w.writerows(data)
    response_keys=[key for key in feature_rows[0] if any(key.endswith(suffix) for suffix in ("_mean","_std")) and key not in {"n_var","n_obj"}]
    old_raw={}
    old_blob=subprocess.run(
        ["git","show","d0e5c1f7478fb7c90dabc624430c1a002708c402:results/phase1_5E_causal_fingerprint.csv"],
        cwd=ROOT,capture_output=True,text=True,encoding="utf-8",check=True,
    ).stdout
    for row in csv.DictReader(io.StringIO(old_blob)):
        keys=[key for key in row if key.endswith(("_delta_empirical_mean_mean","_delta_objective_spread_mean")) and row[key]]
        old_raw[row["task_id"]]=float(max(abs(float(row[key])) for key in keys)) if keys else None
    scale_audit={}
    for task in feature_rows:
        values=np.asarray([float(task[key]) for key in response_keys if key in task and np.isfinite(float(task[key]))])
        scale_audit[task["task_id"]]={"max_abs_normalized_response":float(np.max(np.abs(values))) if len(values) else None,"median_abs_normalized_response":float(np.median(np.abs(values))) if len(values) else None,"nonfinite_count":int(sum(1 for key in response_keys if key in task and not np.isfinite(float(task[key])))),"old_raw_mean_spread_max_abs":old_raw.get(task["task_id"])}
    (ROOT/"results"/"phase1_5E_R1_scale_audit.json").write_text(json.dumps({"tasks":scale_audit,"dtlz1":{k:v for k,v in scale_audit.items() if "DTLZ1" in k}},indent=2),encoding="utf-8")
    stability={}
    by_task={}
    for row in seed_rows: by_task.setdefault(row["task_id"],[]).append(row)
    for task_id,seed_data in by_task.items():
        keys=sorted(key for key in seed_data[0] if key not in {"task_id","probe_seed"} and not key.endswith("_available") and all(key in row and row[key] not in ("",None) for row in seed_data))
        a=np.asarray([float(row[key]) for key in keys for row in seed_data[:1]])
        b=np.asarray([float(row[key]) for key in keys for row in seed_data[1:2]])
        stability[task_id]={"feature_count":len(keys),"pearson":correlation(a,b),"spearman":correlation(rankdata(a),rankdata(b)),"cosine":float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)>0 and np.linalg.norm(b)>0 else None,"normalized_l2":float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))}
    distribution={}
    for metric in ("pearson","spearman","cosine","normalized_l2"):
        values=np.asarray([item[metric] for item in stability.values() if item[metric] is not None],float)
        distribution[metric]={"median":float(np.median(values)) if len(values) else None,"q25":float(np.quantile(values,.25)) if len(values) else None,"q75":float(np.quantile(values,.75)) if len(values) else None,"n":int(len(values))}
    (ROOT/"results"/"phase1_5E_R1_stability.json").write_text(json.dumps({"tasks":stability,"distribution":distribution},indent=2),encoding="utf-8")
    print(json.dumps({"task_count":len(feature_rows),"probe_seeds":2},indent=2))
if __name__=="__main__": main()
