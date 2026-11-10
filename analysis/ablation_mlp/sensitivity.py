import pickle, numpy as np
def lock(e, lo, hi, dv_th, dur):
    g, dv = e['gap'], np.abs(e['v1'] - e['v2']); fol = (g > lo) & (g < hi) & (dv < dv_th); run = 0
    for f in fol:
        run = run + 1 if f else 0
        if run >= dur: return True
    return False
for sc in ['std', 'nar']:
    data = {m: pickle.load(open(f'extract/{sc}_{m}.pkl', 'rb')) for m in ['ego_p09', 'ego_mlp5', 'ego_mlp1', 'lattice_p03' if sc == 'std' else 'lattice']}
    for lo, hi, dv, dur in [(1.5, 3.0, 0.15, 100), (1.0, 3.5, 0.15, 100), (1.5, 3.0, 0.10, 100), (1.5, 3.0, 0.25, 100), (1.5, 3.0, 0.15, 50), (1.5, 3.0, 0.15, 200)]:
        print(sc, f'gap {lo}-{hi} dv<{dv} {dur/100}s:', ' '.join(f"{m.split('_')[-1]}={100*np.mean([lock(e, lo, hi, dv, dur) for e in eps]):.0f}" for m, eps in data.items()))
