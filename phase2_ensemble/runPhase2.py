import time
import pandas as pd
import os

from main.configs import *
from runGenerateEnsemble100 import run_generate_ensemble_100

print("Executando a Fase 2")

# Parâmetros
executions = num_execucoes()

# Carregar Configurações
datasets_used = datasets_used_reduzido()

results_path_phase1 = '../../dream/results/phase1/'

# Lista para armazenar resultados
execution_times = []

# Medir o tempo total
start_time = time.time()

for dataset in datasets_used:
    # Medir o tempo de execução do dataset
    dataset_start_time = time.time()

    # Carregar as matrizes de um arquivo json
    consolidated_file = f'{results_path_phase1}{dataset}-consolidated.json'

    # Carregar as matrizes de um arquivo json
    final_file = f'{results_path_phase1}{dataset}-FINAL.json'

    if not os.path.exists(consolidated_file):
        raise FileNotFoundError(consolidated_file)
    if not os.path.exists(final_file):
        raise FileNotFoundError(final_file)

    for j in range(executions):
        run_generate_ensemble_100(dataset, j, final_file, consolidated_file)

    # Tempo Final da Execução do DataSet[i]
    dataset_end_time = time.time()
    execution_time = dataset_end_time - dataset_start_time

    # Adicionar resultado à lista
    execution_times.append({"Dataset": dataset, "Tempo de Execução (segundos)": execution_time})

    print(f"Tempo para processar {dataset}: {execution_time:.2f} segundos")

    # Salvar resultados em um arquivo CSV
    output_file = '../main/MINE/results/PHASE2/csv/Execution_times_phase2.csv'
    df = pd.DataFrame(execution_times)
    df.to_csv(output_file, index=False)

# Medir o tempo total
end_time = time.time()
total_execution_time = end_time - start_time
execution_times.append({"Dataset": "TOTAL", "Tempo de Execução (segundos)": total_execution_time})

print(f"Tempo total de execução: {total_execution_time:.2f} segundos")

# Salvar resultados em um arquivo CSV
output_file = '../main/MINE/results/PHASE2/csv/Execution_times_phase2.csv'
df = pd.DataFrame(execution_times)
df.to_csv(output_file, index=False)

print(f"Resultados salvos em {output_file}")

print("Fim da Fase 2")
