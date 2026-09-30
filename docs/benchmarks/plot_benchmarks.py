#!/usr/bin/env python3
"""Render the README's four benchmark panels from verified aggregate data.

Usage: python docs/benchmarks/plot_benchmarks.py --data docs/benchmarks/qwen3.json --out docs/loop_vs_dense_tokens
Only the pre-existing 4B series can be previewed before the new results are verified.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BENCHES = [('aime24', 'AIME 2024'), ('aime25', 'AIME 2025'), ('math500', 'MATH-500'), ('gpqa', 'GPQA-Diamond')]
SERIES = [
    ('qwen3_4b_loop', '#E0301E', 'o', '-', '4B · looped'),
    ('qwen3_4b_dense', '#1F5FAD', 's', '-', '4B · dense'),
    ('qwen3_1p7b_loop_short8k', '#9A6700', 'D', '--', '1.7B · looped (short traces)'),
]


def render(data, out, historical_preview=False):
    historical_only = data['status'] == 'verified_historical'
    assert data['status'] == ('preflight_passed' if historical_preview else ('verified_historical' if historical_only else 'verified'))
    historical = historical_preview or historical_only
    assert data['protocol'] == dict(benchmarks='aime24,aime25,math500,gpqa', aime_samples=8,
                                    max_tokens=16384, temperature=.6, top_p=.95)
    assert data['question_count'] == 758 and data['samples_per_checkpoint'] == 1178
    rows = data['points']
    assert len(rows) == (11 if historical else 13)
    expected = dict(qwen3_4b_base=[0], qwen3_4b_loop=[750, 1500, 2250, 3000],
                    qwen3_4b_dense=[750, 1500, 2250, 3000, 3750, 4500])
    if not historical:
        expected['qwen3_1p7b_loop_short8k'] = [2500, 3000]
    assert set(r['series'] for r in rows) == set(expected)
    for series, steps in expected.items():
        assert sorted(r['step'] for r in rows if r['series'] == series) == steps
    for row in rows:
        assert row['tokens_b'] == row['step'] * 524288 / 1e9
        for bench, (questions, n) in dict(aime24=(30, 8), aime25=(30, 8), math500=(500, 1), gpqa=(198, 1)).items():
            v = row['benchmarks'][bench]
            assert (v['questions'], v['samples'], v['samples_per_question']) == (questions, questions * n, n)
            assert abs(v['accuracy'] - 100 * v['correct'] / v['samples']) < 1e-10
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.weight': 'bold', 'axes.labelweight': 'bold',
        'axes.titleweight': 'bold', 'axes.labelsize': 14, 'axes.titlesize': 18,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'axes.linewidth': 1.4,
        'axes.edgecolor': '#222222', 'text.color': '#222222', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), dpi=200)
    base = [r for r in rows if r['series'] == 'qwen3_4b_base']
    for i, (ax, (key, title)) in enumerate(zip(axes.flat, BENCHES)):
        for series, color, marker, style, label in SERIES:
            pts = sorted([r for r in rows if r['series'] == series], key=lambda r: r['step'])
            if not pts:
                continue
            if series in ('qwen3_4b_loop', 'qwen3_4b_dense'):
                pts = base + pts
            ax.plot([r['tokens_b'] for r in pts], [r['benchmarks'][key]['accuracy'] for r in pts],
                color=color, linewidth=2.4, marker=marker, markersize=6.5, markeredgecolor='white',
                markeredgewidth=1.4, linestyle=style, label=label, zorder=5 if '1p7b' in series else (4 if series.endswith('_loop') else 3))
        ax.set_title(title, pad=12)
        ax.set_xlim(-.06, 2.48)
        ax.set_xticks([0, .5, 1., 1.5, 2.])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=True))
        ax.grid(axis='y', color='#D5D5D5', linestyle='--', linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Training tokens (B)', labelpad=7)
        if i == 0:
            ax.set_ylabel('Accuracy (%)', labelpad=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.015), fontsize=15, handlelength=2.3, columnspacing=1.5)
    fig.subplots_adjust(left=.052, right=.99, bottom=.24, top=.76, wspace=.30)
    note = '16k output limit · AIME avg@8 · MATH-500 / GPQA: 1 sample · Temperature 0.6 · Top-p 0.95'
    if historical_preview:
        note = 'Historical 4B data only — 1.7B evaluation pending. Preview; not for publication.'
    elif not historical_only:
        note = '1.7B uses shorter training traces; only steps 2500 and 3000 retained. 4B loop/dense share training data.'
    fig.text(.5, .035, note, ha='center', fontsize=11, fontweight='normal')
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['svg', 'png']:
        target = out.with_suffix('.' + ext)
        fig.savefig(target, facecolor='white')
        if ext == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--historical-preview', action='store_true')
    a = p.parse_args()
    render(json.loads(a.data.read_text()), a.out, a.historical_preview)
