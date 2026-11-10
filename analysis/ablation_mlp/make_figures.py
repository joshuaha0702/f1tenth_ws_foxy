"""패널용 그림·표: 시간 맥락(GRU, p09) vs 맥락 없는 MLP 어블레이션 (컨테이너에서 실행).

    python3 make_figures.py      (extract_episodes.py 결과 extract/*.pkl 필요)

출력: fig*.png / fig*.svg, metrics.csv, metrics.json
지표 정의는 README.md 와 같다.
  - 무충돌 추월 성공률 : 추월(진행 간격 g < -0.6 m)하고 그 에피소드에 충돌이 없음
  - 1 m 이내 접근     : g < 1.0 m 에 도달한 에피소드 비율
  - 나란히 도달       : g < 0 (ego 중심이 앞차 중심을 넘음)
  - 추종 고착         : 1.5 < g < 3.0 m 에서 |v_ego - v_leader| < 0.15 m/s 가 1 s 이상 연속
"""
import csv
import glob
import json
import pickle
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

ROOT = '/root/f1tenth_ws/data/ablation_mlp/panel'
fm.fontManager.ttflist.append(fm.FontEntry(fname=f'{ROOT}/fonts/NotoSansKR.ttf', name='Noto Sans KR'))
plt.rcParams.update({
    'font.family': 'Noto Sans KR', 'font.size': 11, 'axes.unicode_minus': False,
    'axes.edgecolor': '#c9c8c3', 'axes.linewidth': 1.0, 'axes.labelcolor': '#52514e',
    'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b',
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff', 'savefig.facecolor': '#ffffff', 'legend.frameon': False,
})
TEXT2, MUTED, GRID = '#52514e', '#8a8984', '#e6e5e1'

# 모델: (키, 표시 이름, 짧은 이름, 색) — 색은 검증된 categorical 슬롯 1·3·2, lattice 는 참조용 회색
MODELS = [
    ('ego_p09', 'GRU · p09 (시간 맥락)', 'GRU', '#2a78d6'),
    ('ego_mlp5', 'MLP-stack (0.5 s 창)', 'MLP-stack', '#1baf7a'),
    ('ego_mlp1', 'MLP (단일 프레임 · 맥락 없음)', 'MLP', '#eb6834'),
    ('lattice', 'Lattice (전문가 플래너)', 'Lattice', '#a3a29d'),
]
SCEN = [('std', '표준 트랙 10종 (n=200)'), ('nar', '좁은 트랙 5종 (폭 1.6–2.4 m, n=100)')]


def load(sc, key):
    name = {'std': {'lattice': 'lattice_p03'}}.get(sc, {}).get(key, key)
    with open(f'{ROOT}/extract/{sc}_{name}.pkl', 'rb') as f:
        return pickle.load(f)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * (c - h), 100 * (c + h)


def follow_state(e):
    g, dv = e['gap'], np.abs(e['v1'] - e['v2'])
    fol = (g > 1.5) & (g < 3.0) & (dv < 0.15)
    run = 0
    for f in fol:
        run = run + 1 if f else 0
        if run >= 100:          # 100 Hz → 1 s
            return True
    return False


def episode_metrics(eps):
    k = {'n': len(eps), 'reach1': 0, 'along': 0, 'overtake': 0, 'cf_overtake': 0, 'car_coll': 0,
         'any_coll': 0, 'follow': 0}
    t_ot, final_gap = [], []
    for e in eps:
        g = e['gap']
        k['reach1'] += bool((g < 1.0).any())
        k['along'] += bool((g < 0.0).any())
        k['overtake'] += e['overtake']
        k['cf_overtake'] += e['overtake'] and not e['collision']
        k['car_coll'] += e['collision_with'] == 'car'
        k['any_coll'] += e['collision']
        k['follow'] += follow_state(e)
        if e['overtake']:
            t_ot.append(float(e['t'][np.argmax(g < -0.6)]))
        else:
            final_gap.append(float(np.median(g[e['t'] > e['t'][-1] - 5])))
    k['t_overtake_med'] = float(np.median(t_ot)) if t_ot else None
    k['final_gap_nonot_med'] = float(np.median(final_gap)) if final_gap else None
    return k


def parse_solo():
    """analyze_solo.py 출력의 '== ALL' 행 → {조건: (eps, coll, mean v, v@front<1.5)}"""
    out = {}
    for path in ['/root/f1tenth_ws/data/ablation_mlp/solo_ablation_summary.txt',
                 '/root/f1tenth_ws/data/finetune_p09/solo_p09_summary.txt']:
        for line in open(path, errors='replace'):
            m = re.match(r'(\S+)\s+== ALL (\d+) eps, (\d+) coll .*mean v ([\d.]+) \| v@front<1.5 ([\d.]+)', line)
            if m:
                out[m.group(1)] = (int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5)))
    names = {'ego_p09': ['p09_base', 'n04_p09_base'], 'ego_mlp5': ['circ_ego_mlp5_base', 'n04_ego_mlp5_base'],
             'ego_mlp1': ['circ_ego_mlp1_base', 'n04_ego_mlp1_base'], 'lattice': ['circ_lattice_base', 'n04_lattice_base']}
    solo = {}
    for key, conds in names.items():
        rows = [out[c] for c in conds]
        n = sum(r[0] for r in rows)
        solo[key] = {'eps': n, 'coll': sum(r[1] for r in rows),
                     'mean_v': sum(r[0] * r[2] for r in rows) / n,
                     'v_wall_close': sum(r[0] * r[3] for r in rows) / n}
    return solo


def bar_labels(ax, xs, vals, fmt, dy=1.5):
    for x, v in zip(xs, vals):
        ax.text(x, v + dy, fmt.format(v), ha='center', va='bottom', fontsize=10, color='#0b0b0b')


def end_labels(ax, x, items, min_sep):
    """끝점 라벨: 겹치면 세로로 벌리고 끝점까지 가는 가는 연결선(leader line)을 그린다."""
    items = sorted(items, key=lambda it: it[0])
    ys = [it[0] for it in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_sep)
    for (y0, text), y in zip(items, ys):
        if abs(y - y0) > 1e-6:
            ax.plot([x + 0.03, x + 0.1], [y0, y], color='#c9c8c3', linewidth=1)
        ax.text(x + 0.12, y, text, va='center', fontsize=9.5)


def save(fig, name):
    fig.savefig(f'{ROOT}/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{ROOT}/{name}.svg', bbox_inches='tight')
    plt.close(fig)


def main():
    data = {sc: {key: load(sc, key) for key, *_ in MODELS} for sc, _ in SCEN}
    met = {sc: {key: episode_metrics(data[sc][key]) for key, *_ in MODELS} for sc, _ in SCEN}
    solo = parse_solo()

    # 통계: GRU vs 각 모델, 무충돌 추월 · 차량 충돌 · 추종 고착 (Fisher exact, 양측)
    tests = {}
    for sc, _ in SCEN:
        g = met[sc]['ego_p09']
        for key, *_ in MODELS[1:]:
            o = met[sc][key]
            for metric in ['cf_overtake', 'car_coll', 'follow']:
                _, p = fisher_exact([[g[metric], g['n'] - g[metric]], [o[metric], o['n'] - o[metric]]])
                tests[f'{sc}:{key}:{metric}'] = p

    # ---- 표 ----
    with open(f'{ROOT}/metrics.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'model', 'n', 'cf_overtake_%', 'cf_overtake_ci95', 'overtake_%', 'car_collision_%',
                    'car_collision_ci95', 'reach_1m_%', 'alongside_%', 'follow_lock_%', 'follow_lock_ci95',
                    'p_cf_vs_gru', 'p_follow_vs_gru', 'final_gap_nonovertake_m', 't_overtake_med_s'])
        for sc, _ in SCEN:
            for key, *_ in MODELS:
                m = met[sc][key]
                n = m['n']
                cf, cfl, cfh = wilson(m['cf_overtake'], n)
                cc, ccl, cch = wilson(m['car_coll'], n)
                fl, fll, flh = wilson(m['follow'], n)
                w.writerow([sc, key, n, f'{cf:.1f}', f'{cfl:.1f}-{cfh:.1f}', f"{100 * m['overtake'] / n:.1f}",
                            f'{cc:.1f}', f'{ccl:.1f}-{cch:.1f}', f"{100 * m['reach1'] / n:.1f}",
                            f"{100 * m['along'] / n:.1f}", f'{fl:.1f}', f'{fll:.1f}-{flh:.1f}',
                            '' if key == 'ego_p09' else f"{tests[f'{sc}:{key}:cf_overtake']:.1e}",
                            '' if key == 'ego_p09' else f"{tests[f'{sc}:{key}:follow']:.1e}",
                            m['final_gap_nonot_med'], m['t_overtake_med']])
    with open(f'{ROOT}/metrics.json', 'w') as f:
        json.dump({'episodes': met, 'solo': solo, 'fisher_p_vs_gru': tests}, f, indent=1, ensure_ascii=False)

    # ---- 그림 1: 같은 운전 실력, 다른 추월 판단 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={'width_ratios': [1, 1, 1]})
    xs = np.arange(len(MODELS))
    ax = axes[0]
    vals = [solo[k]['mean_v'] for k, *_ in MODELS]
    ax.bar(xs, vals, width=0.6, color=[c for *_, c in MODELS])
    bar_labels(ax, xs, vals, '{:.2f}', dy=0.03)
    ax.set_ylim(0, 2.5)
    ax.set_ylabel('평균 속도 [m/s]')
    ax.set_title('앞차 없음 · 단독 주행\n(좁은 트랙 + 서킷, 각 50 ep, 충돌 0)', fontsize=11.5, loc='left')
    for i, (sc, title) in enumerate(SCEN):
        ax = axes[i + 1]
        ci = [wilson(met[sc][k]['cf_overtake'], met[sc][k]['n']) for k, *_ in MODELS]
        vals = [c[0] for c in ci]
        err = np.array([[c[0] - c[1] for c in ci], [c[2] - c[0] for c in ci]])
        ax.bar(xs, vals, width=0.6, color=[c for *_, c in MODELS])
        ax.errorbar(xs, vals, yerr=err, fmt='none', ecolor='#52514e', elinewidth=1, capsize=3)
        for x, v, c in zip(xs, vals, ci):
            ax.text(x, c[2] + 2, f'{v:.0f}%', ha='center', va='bottom', fontsize=10.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('무충돌 추월 성공률 [%]')
        ax.set_title(f'앞차 있음 · {title}', fontsize=11.5, loc='left')
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([s for _, _, s, _ in MODELS])
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    fig.text(0.01, -0.04, '오차 막대: Wilson 95% 신뢰구간. 모든 모델은 같은 맵 · 같은 seed · 같은 선두차(lattice)로 평가. '
             'MLP 두 모델은 GRU 와 파라미터 수(11.3 M) · 학습 데이터 · 학습 절차가 같다.', fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, 'fig1_outcome')

    # ---- 그림 2: 추월 의사결정 단계 ----
    stages = [('reach1', '1 m 이내\n접근'), ('along', '나란히\n도달'), ('overtake', '추월\n완료'), ('cf_overtake', '무충돌\n추월')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, (sc, title) in zip(axes, SCEN):
        ends = []
        for key, label, short, color in MODELS:
            m = met[sc][key]
            ys = [100 * m[s] / m['n'] for s, _ in stages]
            ax.plot(range(len(stages)), ys, color=color, linewidth=2, marker='o', markersize=7,
                    markeredgecolor='#ffffff', markeredgewidth=1.5, label=label, solid_capstyle='round')
            ends.append((ys[-1], f'{short} {ys[-1]:.0f}%'))
        end_labels(ax, len(stages) - 1, ends, min_sep=5)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([s for _, s in stages])
        ax.set_xlim(-0.2, len(stages) - 1 + 0.9)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11.5, loc='left')
    axes[0].set_ylabel('해당 단계에 도달한 에피소드 [%]')
    axes[0].legend(loc='lower left', fontsize=9)
    fig.text(0.01, -0.04, '나란히 도달한 뒤의 추월 완료율은 모든 모델이 97–100% — 차이는 "실행"이 아니라 "다가가 추월하겠다는 결정"에서 생긴다.',
             fontsize=9.5, color=TEXT2)
    fig.tight_layout()
    save(fig, 'fig2_decision_funnel')

    # ---- 그림 3: 앞차와의 간격에 따른 상대 속도 (추월 전 구간) ----
    bins = np.arange(0.0, 4.01, 0.25)
    mids = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, (sc, title) in zip(axes, SCEN):
        ax.axvspan(1.5, 3.0, color='#f1f0ec', zorder=0)
        ax.text(2.25, 1.32, '추종 구간', ha='center', fontsize=9.5, color=TEXT2)
        ax.axhline(0, color='#c9c8c3', linewidth=1)
        for key, label, short, color in MODELS:
            G, R = [], []
            for e in data[sc][key]:
                g = e['gap']
                end = np.argmax(g < 0) if (g < 0).any() else len(g)
                G.append(g[:end])
                R.append((e['v1'] - e['v2'])[:end])
            G, R = np.concatenate(G), np.concatenate(R)
            idx = np.digitize(G, bins) - 1
            med = np.array([np.median(R[idx == k]) if (idx == k).sum() >= 300 else np.nan for k in range(len(mids))])
            q1 = np.array([np.percentile(R[idx == k], 25) if (idx == k).sum() >= 300 else np.nan for k in range(len(mids))])
            q3 = np.array([np.percentile(R[idx == k], 75) if (idx == k).sum() >= 300 else np.nan for k in range(len(mids))])
            if key in ('ego_p09', 'ego_mlp1'):
                ax.fill_between(mids, q1, q3, color=color, alpha=0.10, linewidth=0)
            ax.plot(mids, med, color=color, linewidth=2, label=label, solid_capstyle='round')
        ax.set_xlim(4.0, 0.0)       # 왼쪽 = 멀리, 오른쪽 = 앞차 바로 뒤 (접근 방향)
        ax.set_ylim(-0.4, 1.45)
        ax.set_xlabel('앞차까지의 진행 간격 [m]  (← 멀리 · 가까이 →)')
        ax.set_title(title, fontsize=11.5, loc='left')
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('상대 속도 v_ego − v_앞차 [m/s]\n(중앙값, 음영 = GRU · MLP 사분위)')
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.01, -0.04, '추월 전 구간만 집계. 상대 속도 0 = 앞차 속도에 맞춰 뒤따름. '
             '맥락 없는 MLP 는 간격 ~3 m 에서 속도를 맞추고 머무르지만, GRU 는 좁은 트랙에서도 속도를 유지하며 다가간다.',
             fontsize=9, color=MUTED)
    save(fig, 'fig3_speed_vs_gap')

    # ---- 그림 4: 추종 고착률 ----
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.18
    for i, (key, label, short, color) in enumerate(MODELS):
        for j, (sc, _) in enumerate(SCEN):
            p, lo, hi = wilson(met[sc][key]['follow'], met[sc][key]['n'])
            x = j + (i - 1.5) * (w + 0.02)
            ax.bar(x, p, width=w, color=color, label=label if j == 0 else None)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt='none', ecolor='#52514e', elinewidth=1, capsize=2.5)
            ax.text(x, hi + 2, f'{p:.0f}', ha='center', va='bottom', fontsize=9.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['표준 트랙 (n=200)', '좁은 트랙 (n=100)'])
    ax.set_ylim(0, 110)
    ax.set_ylabel('추종 고착 에피소드 [%]')
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.set_title('앞차 뒤 1.5–3 m 에서 같은 속도로 1 s 이상 머문 비율', fontsize=11.5, loc='left')
    fig.tight_layout()
    save(fig, 'fig4_follow_lock')

    # ---- 그림 5: 같은 출발 조건(같은 seed)의 대표 에피소드 ----
    pick = None
    for mp in ['Spline_n04', 'Freeform_n04', 'Simple_n04', 'Spline_w24_04']:
        by = {key: {e['episode'][:6]: e for e in data['nar'][key] if e['map'] == mp} for key, *_ in MODELS[:3]}
        for ep in sorted(by['ego_p09']):
            es = [by[k].get(ep) for k, *_ in MODELS[:3]]
            if any(x is None for x in es):
                continue
            if es[0]['overtake'] and not es[0]['collision'] and not es[1]['overtake'] and not es[2]['overtake'] \
                    and not es[1]['collision'] and not es[2]['collision']:
                pick = (mp, ep, es)
                break
        if pick:
            break
    mp, ep, es = pick
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1.1, 1]})
    ax1.axhspan(1.5, 3.0, color='#f1f0ec', zorder=0)
    ax1.axhline(0, color='#c9c8c3', linewidth=1)
    ax1.axhline(-0.6, color='#c9c8c3', linewidth=1, linestyle=(0, (1, 2)))
    ax1.text(15.3, -0.75, '추월 판정\n(한 차 길이 앞)', fontsize=9, color=TEXT2, va='top')
    ax1.text(15.3, 2.25, '추종 구간', fontsize=9, color=TEXT2, va='center')
    k = np.ones(25) / 25      # 표시용 0.25 s 이동평균 (raceline 투영의 계단 제거)
    for (key, label, short, color), e in zip(MODELS[:3], es):
        g = np.convolve(np.pad(e['gap'], 12, mode='edge'), k, mode='valid')
        ax1.plot(e['t'], g, color=color, linewidth=2, label=label, solid_capstyle='round')
    t_ot = float(es[0]['t'][np.argmax(es[0]['gap'] < -0.6)])
    ax1.text(5.6, -1.3, f'GRU: {t_ot:.1f} s 에 추월,\n15 s 후 {-es[0]["gap"][-1]:.0f} m 앞', fontsize=9.5, color='#0b0b0b',
             va='center')
    ax1.text(6.0, 2.55, 'MLP · MLP-stack: 앞차 뒤 ~2 m 에서 15 s 내내 추종', fontsize=9.5, color='#0b0b0b')
    ax1.set_xlabel('시간 [s]')
    ax1.set_ylabel('앞차까지의 진행 간격 [m]')
    ax1.set_xlim(0, 15.2)
    ax1.set_ylim(-3.0, 4.0)
    ax1.yaxis.grid(True, color=GRID, linewidth=1)
    ax1.set_axisbelow(True)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_title(f'같은 출발 조건(같은 seed) · 좁은 트랙 {mp} {ep}', fontsize=11.5, loc='left')
    # 위에서 본 궤적
    import yaml
    from PIL import Image
    meta = yaml.safe_load(open(f'/root/f1tenth_ws/src/f1tenth_lattice_ros2/maps/{mp}/{mp}_map.yaml'))
    img = np.asarray(Image.open(f'/root/f1tenth_ws/src/f1tenth_lattice_ros2/maps/{mp}/{meta["image"]}').convert('L'))
    res, (ox, oy, _) = meta['resolution'], meta['origin']
    h, wpx = img.shape
    walls = np.where(img < 128, 0.0, 1.0)
    ax2.imshow(walls, cmap='gray', vmin=-0.6, vmax=1.0, extent=[ox, ox + wpx * res, oy, oy + h * res], origin='upper')
    ax2.plot(es[0]['xy2'][:, 0], es[0]['xy2'][:, 1], color='#52514e', linewidth=1.5, linestyle=(0, (3, 2)),
             label='앞차 (lattice)')
    for (key, label, short, color), e in zip(MODELS[:3], es):
        ax2.plot(e['xy1'][:, 0], e['xy1'][:, 1], color=color, linewidth=2, label=short)
        ax2.plot(*e['xy1'][-1], 'o', color=color, markersize=7, markeredgecolor='#ffffff', markeredgewidth=1.5)
    ax2.plot(*es[0]['xy1'][0], 's', color='#0b0b0b', markersize=6)
    ax2.text(*(es[0]['xy1'][0] + [0, -0.5]), '출발', fontsize=9, ha='center', va='top')
    allxy = np.vstack([e['xy1'] for e in es] + [es[0]['xy2']])
    pad = 1.0
    ax2.set_xlim(allxy[:, 0].min() - pad, allxy[:, 0].max() + pad)
    ax2.set_ylim(allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9)
    ax2.set_title('위에서 본 궤적 (점 = 15 s 후 위치,\nMLP · MLP-stack 궤적은 거의 겹침)', fontsize=11.5, loc='left')
    fig.tight_layout()
    save(fig, 'fig5_example_episode')
    print('example episode:', mp, ep, [(e['overtake'], round(float(e['gap'][-1]), 2)) for e in es])

    for sc, _ in SCEN:
        for key, *_ in MODELS:
            m = met[sc][key]
            print(sc, key, {k: (round(100 * v / m['n'], 1) if isinstance(v, (int, np.integer)) and k != 'n' else v)
                            for k, v in m.items()})
    print('solo', solo)
    print('fisher', {k: f'{v:.1e}' for k, v in tests.items()})


if __name__ == '__main__':
    main()
