import warnings
import json
import os

from main.configs import *
from BaggingHomogeneous100 import bagging_homogeneous_100
from main.best_params import get_model_instance, load_best_params

models_list = models_used()

def run_generate_ensemble_100(dataset, j, json_final_file, json_consolidated_file):

    results_path_phase2 = '../main/MINE/results/PHASE2/'

    dataset_folder = os.path.join(results_path_phase2, dataset)
    os.makedirs(dataset_folder, exist_ok=True)

    folds = num_folds()

    print(f'Dataset: {dataset} | Execução: {j + 1}')

    # Carregar informações dos arquivos JSON
    with open(json_final_file, 'r') as f:
        final_data = json.load(f)

    with open(json_consolidated_file, 'r') as f:
        consolidated_data = json.load(f)

    if "BestMedianModel" not in final_data:
        raise KeyError(
            f"[{dataset}] Campo 'BestMedianModel' não encontrado em {json_final_file}"
        )
    best_median = final_data.get("BestMedianModel", {})
    model_name = best_median.get("Name")
    print(f'Best Model Name: {model_name}')

    best_params = load_best_params(dataset, model_name)

    if best_params is None:
        raise FileNotFoundError(
            f"Hiperparâmetros não encontrados: dataset={dataset}, modelo={model_name}"
        )

    execution_data = next(item for item in consolidated_data if item["Execution"] == j)
    train_index = np.array(execution_data["TrainIndex"])
    valid_index = np.array(execution_data["ValidIndex"])
    test_index = np.array(execution_data["TestIndex"])
    data = np.array(execution_data["Data"])

    warnings.filterwarnings('ignore')

    pool = []
    unique = []

    for i_fold in range(folds):

        train = train_index[:, i_fold].astype(bool)
        valid = valid_index[:, i_fold].astype(bool)

        data_train = data[train, :]
        data_valid = data[valid, :]

        train_plus = np.vstack([data_train, data_valid])  # Novo conjunto de treinamento

        X_train = train_plus[:, 1:]
        y_train = train_plus[:, 0]

        # ==================================================
        # 🔹 Modelo único (re-treinamento)
        # ==================================================
        model_unique = get_model_instance(model_name)
        model_unique.set_params(**best_params)
        model_unique.fit(X_train, y_train)
        unique.append(model_unique)

        pool.append(bagging_homogeneous_100(100, model_name, train_plus, 0, best_params=best_params))

    poolall = np.array(pool)
    uniqueall = np.array(unique)

    filename_phase2 = os.path.join(
        dataset_folder,
        f'{dataset}-{j + 1}-100.npz'
    )
    np.savez(filename_phase2, pool=poolall, unique=uniqueall)
