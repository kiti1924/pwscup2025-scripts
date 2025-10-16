# language: python
# 使い方: python3 - <<'PY'
import numpy as np

def norm_weight_from_l1(p, q):
    # L1 距離を [0,1] に正規化（最大値は2なので /2）
    d = np.sum(np.abs(p-q))
    nd = d / 2.0
    return 1.0 - nd  # 差が大きい -> 小さい重み

# テスト例
p = np.array([0.5,0.5])
q_close = np.array([0.49,0.51])
q_far   = np.array([1.0,0.0])

print("close diff -> weight:", norm_weight_from_l1(p,q_close))
print("far   diff -> weight:", norm_weight_from_l1(p,q_far))
# PY
