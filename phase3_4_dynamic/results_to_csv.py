# -*- coding: utf-8 -*-
"""
results_phase3-4_to_csv.py — Consolida os resultados do MINE
============================================================================
Lê, para cada dataset, os arquivos gerados por run_mine.py em PHASE3-4/:
    <dataset>-MINE-results.csv   (erros por Execução x SizeL x Técnica x Fold)
    <dataset>-MINE-weights.csv   (pesos do GA)
    <dataset>-time.txt           (tempo total, em segundos)

Gera em PHASE3-4/csv/:
    MINE_Phase3_4_ConsolidatedResults.csv  -> Formato do consolidado
                                              (Dataset;Execucao;SizeL;Tecnica;Fold;Erro)
    MINE_Phase3_4_Weights.csv              -> pesos do GA de todos os datasets
    Execution_times_phase3-4.csv           -> tempo total por dataset
    MINE_Phase3_4_Summary.csv              -> MSE médio por Dataset x SizeL x Técnica

Uso:
    python results_phase3-4_to_csv.py
"""
import os
import numpy as np
import pandas as pd

from main.configs import datasets_used, num_execucoes, num_folds

# =====================================================
# Caminhos
# =====================================================
results_path = '../main/MINE/results/PHASE3-4/'
output_csv_path = os.path.join(results_path, 'csv')
os.makedirs(output_csv_path, exist_ok=True)

TECHNIQUES = ["DS", "MINE_S", "DW", "MINE_W", "DWS", "MINE_WS",
              "IND", "MEAN", "MEDIAN", "ORACLE"]


def read_time(path):
    """O arquivo recebe uma linha por execução concluída do run_MINE.
    Soma as linhas (uma reexecução de dataset já concluído adiciona ~0 s)."""
    with open(path) as f:
        values = [float(x) for x in f.read().split()]
    return sum(values), len(values)


def process_results():
    datasets = datasets_used()
    n_exec, n_folds = num_execucoes(), num_folds()

    results, weights, times, problems = [], [], [], []

    for ds in datasets:
        f_res = os.path.join(results_path, f'{ds}-MINE-results.csv')
        f_w = os.path.join(results_path, f'{ds}-MINE-weights.csv')
        f_t = os.path.join(results_path, f'{ds}-time.txt')

        if not os.path.exists(f_res):
            problems.append(f'{ds}: arquivo de resultados não encontrado')
            continue

        r = pd.read_csv(f_res, sep=';')

        # ---------- verificações de integridade ----------
        n_L = r['SizeL'].nunique()
        expected = n_exec * n_folds * n_L * len(TECHNIQUES)
        if len(r) != expected:
            problems.append(f'{ds}: {len(r)} linhas (esperado {expected})')
        if not np.isfinite(r['Erro']).all():
            problems.append(f'{ds}: há valores NaN/Inf em Erro')
        missing = set(TECHNIQUES) - set(r['Tecnica'])
        if missing:
            problems.append(f'{ds}: técnicas ausentes {sorted(missing)}')

        results.append(r)

        if os.path.exists(f_w):
            weights.append(pd.read_csv(f_w, sep=';'))
        else:
            problems.append(f'{ds}: arquivo de pesos não encontrado')

        if os.path.exists(f_t):
            total, n_lines = read_time(f_t)
            times.append({"Dataset": ds,
                          "Tempo de Execução (segundos)": total,
                          "Tempo de Execução (horas)": total / 3600,
                          "Registros_no_time_txt": n_lines})
        else:
            problems.append(f'{ds}: arquivo de tempo não encontrado')

    # =====================================================
    # Salvar
    # =====================================================
    df_res = pd.concat(results, ignore_index=True)
    df_res = df_res[["Dataset", "Execucao", "SizeL", "Tecnica", "Fold", "Erro"]]
    df_res.to_csv(os.path.join(output_csv_path, 'MINE_Phase3_4_ConsolidatedResults.csv'),
                  sep=';', index=False)

    if weights:
        pd.concat(weights, ignore_index=True).to_csv(
            os.path.join(output_csv_path, 'MINE_Phase3_4_Weights.csv'), sep=';', index=False)

    df_t = pd.DataFrame(times)
    if not df_t.empty:
        total_row = {"Dataset": "TOTAL",
                     "Tempo de Execução (segundos)": df_t["Tempo de Execução (segundos)"].sum(),
                     "Tempo de Execução (horas)": df_t["Tempo de Execução (horas)"].sum(),
                     "Registros_no_time_txt": np.nan}
        df_t = pd.concat([df_t, pd.DataFrame([total_row])], ignore_index=True)
        df_t.to_csv(os.path.join(output_csv_path, 'Execution_times_phase3-4.csv'), index=False)

    summary = (df_res.groupby(["Dataset", "SizeL", "Tecnica"])["Erro"].mean()
               .unstack("Tecnica")[TECHNIQUES])
    summary.to_csv(os.path.join(output_csv_path, 'MINE_Phase3_4_Summary.csv'), sep=';')

    # =====================================================
    # Relatório
    # =====================================================
    print(f'✔ Datasets consolidados: {df_res["Dataset"].nunique()} de {len(datasets)}')
    print(f'✔ Linhas de resultados: {len(df_res)}')
    if not df_t.empty:
        tot = df_t.iloc[-1]
        print(f'✔ Tempo total: {tot["Tempo de Execução (segundos)"]:.0f} s '
              f'({tot["Tempo de Execução (horas)"]:.1f} h)')
    print(f'✔ Arquivos salvos em: {output_csv_path}')
    if problems:
        print('\n⚠ Verificar:')
        for p in problems:
            print('  -', p)
    else:
        print('✔ Nenhuma inconsistência encontrada.')


if __name__ == "__main__":
    process_results()
