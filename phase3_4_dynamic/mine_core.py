# -*- coding: utf-8 -*-
"""
mine_core.py — Port fiel (Python/NumPy) das Fases 3-4 do MINE (Moura et al., 2021)
====================================================================================
Referência: mine-master/PHASE3-4/{extractAttValues, attribute, knn, simpleDS,
simpleDW, simpleDWS, process, objectiveFunction}.m

Convenções (idênticas ao MATLAB, porém com índices 0-based):
  - coluna 0 de `data` = alvo; colunas 1: = atributos
  - matrix_values: shape (n_measures, L, n_patterns)
  - ATTRIBUTES = [5, 7, 9, 11, 13, 15, 18, 19]  (todas >= 0 por construção)

Diferenças deliberadas em relação ao MATLAB:
  (a) Nenhum `denominador == 0 -> 1` silencioso: valores não finitos disparam erro
      (no MATLAB, virariam NaN e contaminariam o MSE de forma visível).
  (b) O limiar do DWS pode ser 'original' ((Emax-Emin)*p, como no código e no
      artigo) ou 'corrected' (Emin + (Emax-Emin)*p). Default = 'original' (fidelidade).
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors

ATTRIBUTES = [5, 7, 9, 11, 13, 15, 18, 19]
IDX_ATT9 = ATTRIBUTES.index(9)   # MATLAB: matrixValues(3,:,v) -> medida 9
EPS_ZERO_DIST = 0.0              # MATLAB knn.m remove vizinhos com distância == 0


# --------------------------------------------------------------------------- #
# Região de competência (knn.m) — exclui TODOS os vizinhos com distância 0
# --------------------------------------------------------------------------- #
def knn_nonzero(ref_X: np.ndarray, query_X: np.ndarray, K: int):
    """Retorna (dist, idx) dos K vizinhos mais próximos com distância > 0.
    Equivalente a knn.m: sort -> remove dist == 0 -> primeiros K."""
    n_ref = ref_X.shape[0]
    extra = min(n_ref, K + 16)
    nn = NearestNeighbors(n_neighbors=extra, algorithm="auto").fit(ref_X)
    dist, idx = nn.kneighbors(query_X)
    out_d = np.empty((query_X.shape[0], K))
    out_i = np.empty((query_X.shape[0], K), dtype=int)
    for r in range(query_X.shape[0]):
        d, i = dist[r], idx[r]
        keep = d > EPS_ZERO_DIST
        if keep.sum() < K:                      # muitos duplicados: busca ampliada
            d2, i2 = nn.kneighbors(query_X[r:r + 1], n_neighbors=n_ref)
            d, i = d2[0], i2[0]
            keep = d > EPS_ZERO_DIST
            if keep.sum() < K:
                raise ValueError("Menos de K vizinhos com distância > 0.")
        out_d[r], out_i[r] = d[keep][:K], i[keep][:K]
    return out_d, out_i


# --------------------------------------------------------------------------- #
# extractAttValues.m + attribute.m (somente as 8 medidas do artigo)
# --------------------------------------------------------------------------- #
def extract_att_values(eval_set, ref_set, pred_eval, pred_ref, K=10,
                       attributes=ATTRIBUTES, column_target=0):
    """
    eval_set : (n, 1+d)  padrões avaliados (teste, ou o próprio treino na otimização)
    ref_set  : (m, 1+d)  conjunto de referência da região de competência (train+valid)
    pred_eval: (n, L)    previsões do pool sobre eval_set
    pred_ref : (m, L)    previsões do pool sobre ref_set
    retorna  : (n_att, L, n)
    """
    y_ref = ref_set[:, column_target]
    X_ref = np.delete(ref_set, column_target, axis=1)
    X_eval = np.delete(eval_set, column_target, axis=1)

    dist, idx = knn_nonzero(X_ref, X_eval, K)             # (n, K)
    w = (1.0 / dist) / np.sum(1.0 / dist, axis=1, keepdims=True)   # pesos
    w3 = w[:, :, None]                                    # (n, K, 1)

    y_nb = y_ref[idx][:, :, None]                         # (n, K, 1)
    p_nb = pred_ref[idx]                                  # (n, K, L)
    e2 = (y_nb - p_nb) ** 2                               # (n, K, L)

    out = {}
    out[5] = np.var(p_nb, axis=1, ddof=1)                          # variancia
    out[7] = np.sum(np.abs(y_nb - p_nb) * w3, axis=1)              # erroTotalAbsoluto
    out[9] = np.sum(e2 * w3, axis=1)                               # erroTotalMSEDistancia
    out[11] = np.min(e2 * w3, axis=1)                              # erroMinimoDistancia
    out[13] = np.max(e2 * w3, axis=1)                              # erroMaximoDistancia
    out[15] = np.sum((pred_eval[:, None, :] - y_nb) ** 2 * w3, axis=1)  # somatorioDiferencasDistanciaReal
    out[18] = np.sqrt(out[9])                                      # erroTotalRMSEDistancia
    out[19] = e2[:, 0, :]                                          # erroVizinhoMaisProximo

    M = np.stack([out[a] for a in attributes], axis=0)   # (n_att, n, L)
    M = np.transpose(M, (0, 2, 1))                       # (n_att, L, n)
    if not np.all(np.isfinite(M)) or np.any(M < 0):
        raise ValueError("Medidas de competência não finitas ou negativas.")
    return M


def _C(matrix_values, attributes, measure):
    """(n_att, L, n) -> (n, L) para uma única medida."""
    k = list(attributes).index(measure)
    return matrix_values[k].T.copy()


def _check(W):
    if not np.all(np.isfinite(W)):
        raise FloatingPointError("Pesos não finitos na combinação.")
    return W


# --------------------------------------------------------------------------- #
# DRS da literatura com uma única medida (simpleDS/DW/DWS.m)
# --------------------------------------------------------------------------- #
def simple_DS(test_set, pred, column_target, matrix_values, attributes, measure):
    C = _C(matrix_values, attributes, measure)
    p = pred[np.arange(pred.shape[0]), np.argmin(C, axis=1)]
    return (test_set[:, column_target] - p) ** 2, p


def _inv_weights_with_zero_rule(C, mask):
    """1/C restrito a mask; linhas com algum C==0 selecionado -> indicador desses zeros
    (reproduz o tratamento de Inf/NaN de simpleDW.m / simpleDWS.m)."""
    zero = (C == 0) & mask
    with np.errstate(divide="ignore"):
        num = np.where(mask & ~zero, 1.0 / np.where(C == 0, 1.0, C), 0.0)
    rows = zero.any(axis=1)
    num[rows] = zero[rows].astype(float)
    return _check(num / num.sum(axis=1, keepdims=True))


def simple_DW(test_set, pred, column_target, matrix_values, attributes, measure):
    C = _C(matrix_values, attributes, measure)
    W = _inv_weights_with_zero_rule(C, np.ones_like(C, dtype=bool))
    p = np.sum(W * pred, axis=1)
    return (test_set[:, column_target] - p) ** 2, p


def _dws_mask(C, percentual, threshold):
    Emax, Emin = C.max(axis=1), C.min(axis=1)
    E = (Emax - Emin) * (percentual / 100.0)
    if threshold == "corrected":
        E = Emin + E
    mask = C <= E[:, None]
    mask[mask.sum(axis=1) == 0] = True            # iPred(sum==0,:) = 1
    return mask


def simple_DWS(test_set, pred, column_target, matrix_values, attributes,
               percentual, measure, threshold="original"):
    C = _C(matrix_values, attributes, measure)
    W = _inv_weights_with_zero_rule(C, _dws_mask(C, percentual, threshold))
    p = np.sum(W * pred, axis=1)
    return (test_set[:, column_target] - p) ** 2, p


# --------------------------------------------------------------------------- #
# MINE-S / MINE-W / MINE-WS (process.m)
# --------------------------------------------------------------------------- #
def _combined(weights, matrix_values):
    """values = sum_p w_p * m_p  -> (n, L)"""
    return np.tensordot(np.asarray(weights, float), matrix_values, axes=(0, 0)).T


def _replace_zero_rows(values, matrix_values):
    """process.m: padrões com algum valor 0 recebem a medida 9 (linha 3 no MATLAB)."""
    rows = np.any(values == 0, axis=1)
    if rows.any():
        values = values.copy()
        values[rows] = matrix_values[IDX_ATT9].T[rows]
    return values


def mine_predict(weights, matrix_values, pred, option, threshold="original"):
    """option: 1 = MINE-S, 2 = MINE-W, 3 = MINE-WS"""
    values = _combined(weights, matrix_values)
    if option == 1:
        return pred[np.arange(pred.shape[0]), np.argmin(values, axis=1)]
    values = _replace_zero_rows(values, matrix_values)
    if option == 2:
        with np.errstate(divide="ignore"):
            num = 1.0 / values
        # MATLAB: se ainda houver 0 -> Inf/Inf = NaN. Aqui: regra do indicador.
        if np.isinf(num).any():
            return np.sum(_inv_weights_with_zero_rule(values, np.ones_like(values, bool)) * pred, axis=1)
        return np.sum(_check(num / num.sum(axis=1, keepdims=True)) * pred, axis=1)
    if option == 3:
        W = _inv_weights_with_zero_rule(values, _dws_mask(values, 50, threshold))
        return np.sum(W * pred, axis=1)
    raise ValueError(option)


def mine_process(weights, data_set, pred, column_target, matrix_values, option,
                 threshold="original"):
    """process.m -> vetor de erros quadráticos."""
    p = mine_predict(weights, matrix_values, pred, option, threshold)
    return (data_set[:, column_target] - p) ** 2


# --------------------------------------------------------------------------- #
# GA (aproximação do `ga` do MATLAB com as opções de configs.m)
# --------------------------------------------------------------------------- #
def ga_minimize(fitness, n_var, rng, pop_size=80, elite=None, crossover_fraction=0.8,
                max_generations=None, stall_generations=40, function_tolerance=1e-6,
                initial_population=None):
    """
    Opções espelhadas de configs.m:
      PopulationSize=80, PopInitRange=[0,1], lb=0, ub=1, EliteCount=nvars,
      StallGenLimit=40, FitnessLimit=0, InitialPopulation=[ones; eye(nvars)].
    Operadores (defaults do MATLAB, aproximados): escala por rank (1/sqrt(rank)),
    seleção estocástica uniforme, crossover disperso, mutação gaussiana com escala
    decrescente, projetada em [0,1]. MaxGenerations default = 100*nvars.
    """
    elite = n_var if elite is None else elite
    max_generations = 100 * n_var if max_generations is None else max_generations
    P = rng.uniform(0, 1, (pop_size, n_var))
    if initial_population is None:
        initial_population = np.vstack([np.ones(n_var), np.eye(n_var)])
    P[:len(initial_population)] = initial_population
    F = np.array([fitness(x) for x in P])
    best_hist = [F.min()]

    n_x = int(round(crossover_fraction * (pop_size - elite)))
    n_m = pop_size - elite - n_x

    for g in range(1, max_generations + 1):
        order = np.argsort(F)
        ranks = np.empty(pop_size); ranks[order] = np.arange(1, pop_size + 1)
        expct = 1.0 / np.sqrt(ranks)
        expct = expct / expct.sum()

        def select(n):          # stochastic uniform
            cum = np.cumsum(expct)
            step = 1.0 / n
            ptr = rng.uniform(0, step) + step * np.arange(n)
            return np.searchsorted(cum, ptr)

        par = P[select(2 * n_x + n_m)]
        p1, p2, pm = par[:n_x], par[n_x:2 * n_x], par[2 * n_x:]
        mask = rng.random((n_x, n_var)) < 0.5
        kids_x = np.where(mask, p1, p2)
        sigma = max(1.0 * (1.0 - g / max_generations), 1e-3)
        kids_m = np.clip(pm + rng.normal(0, sigma, pm.shape), 0.0, 1.0)

        newP = np.vstack([P[order[:elite]], kids_x, kids_m])
        newF = np.concatenate([F[order[:elite]], [fitness(x) for x in newP[elite:]]])
        P, F = newP, newF
        best_hist.append(F.min())

        if F.min() <= 0:                                        # FitnessLimit = 0
            break
        if g >= stall_generations:
            old, new = best_hist[-1 - stall_generations], best_hist[-1]
            if (old - new) / max(1.0, abs(new)) <= function_tolerance:
                break
    return P[np.argmin(F)], float(F.min()), g
