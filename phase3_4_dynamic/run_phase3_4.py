import json
import os
import time
import pandas as pd

from main.configs import *
from run_mine import run_MINE

# Load configurations
results_path_phase1 = '../../dream/results/phase1/'
datasets_used = datasets_used()

output_dir = '../main/MINE/results/PHASE3-4/'
os.makedirs(output_dir, exist_ok=True)

# Lista para armazenar resultados
execution_times = []

# Loop over datasets
for dataset_name in datasets_used:
    # Carregar as matrizes de um arquivo json
    json_final_file = f'{results_path_phase1}{dataset_name}-FINAL.json'

    with open(json_final_file, 'r') as f:
        final_data = json.load(f)

    if "BestMedianModel" not in final_data:
        raise KeyError(
            f"[{dataset_name}] Campo 'BestMedianModel' não encontrado em {json_final_file}"
        )
    best_model_type = final_data["BestMedianModel"].get("Name")

    dataset_start_time = time.time()

    # L = 10 e 100; GA em 4 processos paralelos
    run_MINE(dataset_name, best_model_type, size_L=(10, 100), n_jobs=4,
             phase1_dir=results_path_phase1, out_dir=output_dir)

    execution_time = time.time() - dataset_start_time
    execution_times.append({"Dataset": dataset_name, "Tempo de Execução (segundos)": execution_time})
    print(f"\nTempo para processar {dataset_name}: {execution_time:.2f} segundos")

    pd.DataFrame(execution_times).to_csv(f'{output_dir}Execution_times_phase3-4.csv', index=False)
