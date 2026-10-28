# -*- coding: utf-8 -*-
"""
run_mine.py — Fases 3-4 do MINE no protocolo do DREAM (mesmos folds/execuções)
====================================================================================
Reproduz runMINE.m com 10 técnicas:
  0 DS (lit., medida 18)   1 MINE-S   2 DW (lit.)   3 MINE-W   4 DWS (lit.)   5 MINE-WS
  6 IND                    7 MEAN     8 MEDIAN      9 ORACLE
"""
from __future__ import annotations
import argparse, json, os, time, zlib
import numpy as np
import pandas as pd

from mine_core import (ATTRIBUTES, extract_att_values, simple_DS, simple_DW, simple_DWS,
                       mine_process, ga_minimize)

# Dependências do seu projeto (mantidas)
from main.testDataTesting import test_data_testing
from main.configs import (num_execucoes, num_folds, tam_competencia,
                               atributos_competencia, conf_GA)

TECHS = ["DS", "MINE_S", "DW", "MINE_W", "DWS", "MINE_WS", "IND", "MEAN", "MEDIAN", "ORACLE"]
K = tam_competencia()
MEASURE_LIT = 18
N_EXEC, N_FOLDS = num_execucoes(), num_folds()
assert list(atributos_competencia()) == ATTRIBUTES, "Ordem das medidas difere do MINE original."

_GA = conf_GA()
N_ATT = len(ATTRIBUTES)
# configs.m original: ip = [ones(1,cAtt); eye(cAtt)]  (9 x 8).
# ATENÇÃO: conf_GA()['initial_population'] = zeros(8,8) NÃO corresponde ao original
# (pesos nulos anulam a função de competência). Usa-se aqui a população inicial do MATLAB.
GA_KW = dict(pop_size=_GA["population_size"], elite=_GA["elite_count"],
             stall_generations=_GA["stall_gen_limit"],
             initial_population=np.vstack([np.ones(N_ATT), np.eye(N_ATT)]))


def _run_ga(opt, L, seed, d_plus, Pp, Mp, threshold):
    rng = np.random.default_rng(seed)
    fit = lambda w: float(np.mean(mine_process(w, d_plus, Pp, 0, Mp, opt, threshold)))
    return (opt, L) + ga_minimize(fit, N_ATT, rng, **GA_KW)


def run_MINE(dataset_name, model_type, size_L=(10, 100), threshold="original", n_jobs=1,
             phase1_dir="../../dream/results/phase1/",
             phase2_dir="../main/MINE/results/PHASE2/",
             out_dir="../main/MINE/results/PHASE3-4/"):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{phase1_dir}{dataset_name}-consolidated.json") as f:
        consolidated = json.load(f)

    t0 = time.time()
    ckpt = f"{out_dir}{dataset_name}-exec"
    for ex in range(N_EXEC):
        if os.path.exists(f"{ckpt}{ex + 1:02d}-results.csv"):      # retomada
            continue
        rows, weights_log = [], []
        ed = next(it for it in consolidated if it["Execution"] == ex)
        tr_idx, va_idx, te_idx = (np.array(ed[k]) for k in ("TrainIndex", "ValidIndex", "TestIndex"))
        data = np.array(ed["Data"])

        npz = np.load(f"{phase2_dir}{dataset_name}/{dataset_name}-{ex + 1}-100.npz", allow_pickle=True)
        pool, unique = npz["pool"], npz["unique"]

        for i in range(N_FOLDS):
            tr, va, te = (m[:, i].astype(bool) for m in (tr_idx, va_idx, te_idx))
            d_test = data[te]
            d_plus = np.vstack((data[tr], data[va]))          # train + valid (como no MATLAB)

            P_test = np.column_stack([test_data_testing(d_test[:, 1:], pool[i][j], model_type) for j in range(100)])
            P_plus = np.column_stack([test_data_testing(d_plus[:, 1:], pool[i][j], model_type) for j in range(100)])
            p_unique = test_data_testing(d_test[:, 1:], unique[i], model_type)

            MV_test = extract_att_values(d_test, d_plus, P_test, P_plus, K)
            MV_plus = extract_att_values(d_plus, d_plus, P_plus, P_plus, K)   # knn exclui o próprio ponto

            y = d_test[:, 0]

            # --- GA (3 variantes x |size_L|) — otimização SOMENTE em train+valid ---
            jobs = [(opt, L, zlib.crc32(f"{dataset_name}|{ex}|{i}|{L}|{opt}".encode()),
                     d_plus, P_plus[:, :L], MV_plus[:, :L, :], threshold)
                    for L in size_L for opt in (1, 2, 3)]
            if n_jobs == 1:
                ga_out = [_run_ga(*j) for j in jobs]
            else:
                from joblib import Parallel, delayed
                ga_out = Parallel(n_jobs=n_jobs)(delayed(_run_ga)(*j) for j in jobs)
            ga_res = {(o, L): (w, f, g) for o, L, w, f, g in ga_out}

            for L in size_L:
                Pt, Pp = P_test[:, :L], P_plus[:, :L]
                Mt, Mp = MV_test[:, :L, :], MV_plus[:, :L, :]
                err = {}

                err["DS"], _ = simple_DS(d_test, Pt, 0, Mt, ATTRIBUTES, MEASURE_LIT)
                err["DW"], _ = simple_DW(d_test, Pt, 0, Mt, ATTRIBUTES, MEASURE_LIT)
                err["DWS"], _ = simple_DWS(d_test, Pt, 0, Mt, ATTRIBUTES, 50, MEASURE_LIT, threshold)

                for opt, name in ((1, "MINE_S"), (2, "MINE_W"), (3, "MINE_WS")):
                    w_best, f_best, gens = ga_res[(opt, L)]
                    err[name] = mine_process(w_best, d_test, Pt, 0, Mt, opt, threshold)
                    weights_log.append({"Dataset": dataset_name, "Execucao": ex + 1, "Fold": i + 1,
                                        "SizeL": L, "Tecnica": name, "Generations": gens,
                                        "FitnessTrain": f_best,
                                        **{f"w_att{a}": v for a, v in zip(ATTRIBUTES, w_best)}})

                err["IND"] = (p_unique - y) ** 2
                err["MEAN"] = (Pt.mean(axis=1) - y) ** 2
                err["MEDIAN"] = (np.median(Pt, axis=1) - y) ** 2
                err["ORACLE"] = np.min((Pt - y[:, None]) ** 2, axis=1)

                for t in TECHS:
                    rows.append({"Dataset": dataset_name, "Execucao": ex + 1, "SizeL": L,
                                 "Tecnica": t, "Fold": i + 1, "Erro": float(np.mean(err[t]))})
        pd.DataFrame(weights_log).to_csv(f"{ckpt}{ex + 1:02d}-weights.csv", sep=";", index=False)
        pd.DataFrame(rows).to_csv(f"{ckpt}{ex + 1:02d}-results.csv", sep=";", index=False)
        print(f"{dataset_name} | exec {ex + 1}/{N_EXEC} | {time.time() - t0:.0f}s", flush=True)

    # Consolidação (formato compatível com MINE_Phase3_4_ConsolidatedResults.csv)
    for kind in ("results", "weights"):
        parts = [pd.read_csv(f"{ckpt}{e + 1:02d}-{kind}.csv", sep=";") for e in range(N_EXEC)]
        pd.concat(parts).to_csv(f"{out_dir}{dataset_name}-MINE-{kind}.csv", sep=";", index=False)
    with open(f"{out_dir}{dataset_name}-time.txt", "a") as f:
        f.write(f"{time.time() - t0}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("model_type")
    ap.add_argument("--sizeL", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--threshold", choices=["original", "corrected"], default="original")
    ap.add_argument("--n_jobs", type=int, default=1, help="processos paralelos para o GA")
    a = ap.parse_args()
    run_MINE(a.dataset, a.model_type, tuple(a.sizeL), a.threshold, a.n_jobs)
