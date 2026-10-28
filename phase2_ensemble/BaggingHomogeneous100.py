import numpy as np
from main.best_params import get_model_instance

def bagging_homogeneous_100(S, model_name, data, column_class, best_params=None):

    bootsam = np.random.choice(data.shape[0], size=(S, data.shape[0]), replace=True)

    pool = []

    for i in range(S):
        sample = data[bootsam[i], :]  # Obter os dados para treinamento
        y      = sample[:, column_class]   # Obter a coluna de rótulos
        X      = np.delete(sample , column_class, axis=1)

        # 🔹 Modelo já parametrizado
        model = get_model_instance(model_name)

        if best_params:
            model.set_params(**best_params)

        model.fit(X, y)

        pool.append(model)

    return pool
