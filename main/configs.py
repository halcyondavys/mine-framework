import numpy as np

# pastas de referências
def path_DREAM():
    results_path_phase1    = '../main/DREAM/results/PHASE1/'
    results_path_phase2    = '../main/DREAM/results/PHASE2/'
    output_csv_path_phase1 = '../main/DREAM/results/PHASE1/csv/'
    output_csv_path_phase2 = '../main/DREAM/results/PHASE2/csv/'
    model_save_root        = '../main/DREAM/models_treinados/'
    return results_path_phase1, results_path_phase2, output_csv_path_phase1, output_csv_path_phase2, model_save_root


# Lista de conjuntos de dados
def datasets_used():
    datasetsused = [
        'abalone',
        'airfoil_self_noise',
        'albrecht',
        'bank32nh',
        'bank8FM',
        'ccpp',
        'china',
        'cocomonasa60',
        'cocomo81',
        'concrete',
        'cpu_act',
        'cpu_small',
        'delta_ailerons',
        'delta_elevators',
        'desharnais',
        'energy_efficiency',
        'friedman',
        'housing',
        'kin8nm',
        'machine',
        'maxwell',
        'nasa93',
        'parkinsons_updrs',
        'puma32H',
        'puma8NH',
        'stocks',
        'triazines',
        'wineq-red',
        'wineq-white',
        'wiscoinBreastCancer'
    ]
    return datasetsused

# Descomente e altere o nome do conjunto de dados se quiser testar apenas um
def datasets_used_reduzido():
    datasetsused = [
        'puma8NH' ,'stocks', 'triazines', 'wineq-white', 'wiscoinBreastCancer'
    ]
    return datasetsused

# Lista dos Modelos
def models_used():
    modelsused = [
        'cart',
        'linear',
        'mlp',
        'svr',
        'knn',
        'rbf',
        'elm',
        'ridge',
        'huber'
    ]
    return modelsused

def models_fixed_used():
    modelsused = [
        ('cart', 0, 0),
        ('linear', 0, 0),
        ('mlp1', 0, 10),
        ('mlp2', 0, [5, 10]),
        ('svr-rbf', 0, 0),
        ('svr-poly1', 0, 0),
        ('svr-poly3', 0, 0),
        ('rbf', 0, 10),
        ('knn3', 3, 0),
        ('knn5', 5, 0)
    ]
    return modelsused

# Número de execuções
def num_execucoes():
    Numexecutions = 20
    return Numexecutions

# Número de folds
def num_folds():
    folds = 10
    return folds

# Tamanhos de ensemble usados
def tam_ensemble():
    sizeL = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    return sizeL

# Tamanho da Região de Competência
def tam_competencia():
    K = 10
    return K

# Atributos a serem extraídos da região de competência
def atributos_competencia():
    attributes = [5, 7, 9, 11, 13, 15, 18, 19]
    #attributes = [5]
    return attributes

# Configurações do GA
def conf_GA():
    ga_options = {
        'display': 'off',
        'population_size': 80,
        'pop_init_range': [0, 1],
        'fitness_limit': 0,
        'stall_gen_limit': 40,
        'initial_population': np.zeros((len(atributos_competencia()), len(atributos_competencia()))),
        'elite_count': len(atributos_competencia()),
        'use_parallel': True
    }
    return ga_options

def param_grid_models():
    return {
        'mlp': {
            # Arquiteturas leves e eficientes
            'hidden_layer_sizes': [
                (10,), (20,), (50,),
                (10,10), (20,10), (50,20)
            ],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': np.logspace(-5, -3, 10),
            'learning_rate_init': np.logspace(-4, -2, 10),
            'max_iter': [300],
            'early_stopping': [True],
            'validation_fraction': [0.15]
        },
        'svr': {
            'kernel': ['rbf', 'linear'],
            'C': np.logspace(-2, 2, 20),
            'gamma': np.logspace(-3, 1, 20),
            'epsilon': np.linspace(0.001, 0.1, 20)
        },
        'rbf': {
            'alpha': [0.1, 1.0],
            'gamma': [0.01, 0.1],
            'kernel': ['rbf']
        },
        'knn': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'cart': {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'linear': {},
        'elm': {
            'n_hidden': [10, 15, 20],
            'activation_func': ['tanh', 'sigmoid'],
            'alpha': [0.3, 0.7],
            'rbf_width': [0.3, 0.7]
        },
        'ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'huber': {
            'epsilon': [1.35, 1.5, 1.75],
            'alpha': [0.0001, 0.001],
            'max_iter': [1000, 2000]
        }
    }
