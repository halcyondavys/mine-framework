def test_svr(data, regressor):
    return regressor.predict(data)


def test_cart(data, regressor):
    return regressor.predict(data)


def test_linear(data, regressor):
    return regressor.predict(data)


def test_knn(data, regressor):
    return regressor.predict(data)


def test_mlp(data, regressor):
    return regressor.predict(data)


def test_rbf(data, regressor):
    return regressor.predict(data)


def test_data_testing(data, regressor, tipo):
    """
    Executa a predição do modelo treinado.
    O parâmetro 'tipo' é mantido por compatibilidade com o MINE original.
    """

    if tipo in ['cart', 'linear', 'knn', 'mlp', 'rbf']:
        return regressor.predict(data)

    elif tipo in ['svr', 'svr-linear', 'svr-rbf', 'svr-poly1', 'svr-poly3']:
        return regressor.predict(data)

    elif tipo in ['ridge', 'huber', 'elm']:
        return regressor.predict(data)

    else:
        # Fallback seguro: qualquer modelo sklearn-like
        try:
            return regressor.predict(data)
        except Exception:
            raise ValueError(f"Unsupported regressor type: {tipo}")
