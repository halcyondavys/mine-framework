# -*- coding: utf-8 -*-
"""Testes de sanidade do port do MINE. Uso: python test_mine_core.py"""
import numpy as np
from mine_core import (ATTRIBUTES, extract_att_values, simple_DS, simple_DW, simple_DWS,
                       mine_process, ga_minimize)
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(1)
n = 800
X = rng.uniform(0, 1, (n, 5)); y = np.sin(4 * X[:, 0]) + X[:, 1] ** 2
y = (y - y.min()) / np.ptp(y)
D = np.c_[y, X]; test, plus = D[:100], D[100:]
plus = np.vstack([plus, plus[:30]])            # duplicados (distância 0)
test[:10, 1:] = plus[:10, 1:]                  # teste com vizinho idêntico no treino

pool = []
for j in range(30):
    b = rng.integers(0, len(plus), len(plus))
    pool.append(DecisionTreeRegressor(random_state=j).fit(plus[b, 1:], plus[b, 0]))
Pt = np.column_stack([m.predict(test[:, 1:]) for m in pool])
Pp = np.column_stack([m.predict(plus[:, 1:]) for m in pool])

Mt = extract_att_values(test, plus, Pt, Pp, 10)
Mp = extract_att_values(plus, plus, Pp, Pp, 10)
assert Mt.shape == (len(ATTRIBUTES), 30, 100) and np.isfinite(Mt).all() and (Mt >= 0).all()

worst = np.mean(np.max((Pt - test[:, [0]]) ** 2, axis=1))
for f, args in ((simple_DW, ()), (simple_DWS, (50,))):
    e, _ = f(test, Pt, 0, Mt, ATTRIBUTES, *args, 18)
    assert np.isfinite(e).all() and e.mean() <= worst + 1e-12, f.__name__

# DWS: modelos rejeitados devem ter peso 0
C = np.array([[1.0, 1.1, 5.0, 6.0]]).T[None]
_, p = simple_DWS(np.zeros((1, 2)), np.array([[0., 0., 10., 10.]]), 0, C, [18], 50, 18)
assert p[0] == 0.0, "DWS está ponderando os modelos rejeitados"

# GA melhora (ou iguala) a solução inicial [1,...,1] no conjunto de otimização
for opt in (1, 2, 3):
    fit = lambda w: float(np.mean(mine_process(w, plus, Pp, 0, Mp, opt)))
    w, fbest, g = ga_minimize(fit, len(ATTRIBUTES), np.random.default_rng(opt))
    assert fbest <= fit(np.ones(len(ATTRIBUTES))) + 1e-12 and ((w >= 0) & (w <= 1)).all()

print("OK — todos os testes passaram.")
