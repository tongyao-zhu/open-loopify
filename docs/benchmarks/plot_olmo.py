#!/usr/bin/env python3
"""Reproduce the OLMo GSM8K figure from verified aggregate counts.

python docs/benchmarks/plot_olmo.py --data docs/benchmarks/olmo2_1b.json --out docs/olmo2_1b_gsm8k
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(data, out):
    assert data['status'] == 'verified'
    points = data['points']
    assert [p['step'] for p in points] == [1000, 1500, 2000, 2500, 3000]
    assert data['missing_checkpoint_steps'] == [500]
    for p in points + [data['baseline']]:
        assert p['n'] == 1319 and abs(p['accuracy'] - p['correct'] / p['n']) < 1e-12
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
    x = [p['step'] for p in points]
    y = [100*p['accuracy'] for p in points]
    err = [[100*(p['accuracy']-p['accuracy_ci95_wilson'][0]) for p in points],
           [100*(p['accuracy_ci95_wilson'][1]-p['accuracy']) for p in points]]
    ax.errorbar(x, y, yerr=err, color='#1F5FAD', marker='o', markersize=6,
                linewidth=2, capsize=4, elinewidth=1.2, label='Looped · layers [7, 11) × 3')
    base = 100*data['baseline']['accuracy']
    ax.axhline(base, color='#555555', linestyle='--', linewidth=1.4,
               label=f'Original stage-1 base: {base:.2f}% (reference)')
    for p in points:
        ax.text(p['step'], 100*p['accuracy_ci95_wilson'][1]+.55,
                f"{100*p['accuracy']:.2f}%", ha='center', fontsize=10, fontweight='bold')
    ax.set(xlim=(850,3150), ylim=(0,24), xticks=x, yticks=[0,5,10,15,20],
           xlabel='Training steps', ylabel='GSM8K accuracy (%)')
    ax.set_title('OLMo-2-1B · performance during looped training', fontweight='bold', pad=12)
    ax.grid(axis='y', color='#DDDDDD', linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    fig.subplots_adjust(left=.09, right=.98, bottom=.24, top=.85)
    fig.text(.5,.09,'1,319 test questions · 8-shot · greedy · 512-token limit · bars: 95% Wilson intervals',
             ha='center',fontsize=9)
    fig.text(.5,.035,'Step 500 unavailable · one training seed · no matched dense-training control',
             ha='center',fontsize=9,color='#555555')
    for ext in ['png','svg']:
        f = out.with_suffix('.'+ext)
        fig.savefig(f, facecolor='white')
        if ext == 'svg':
            f.write_text('\n'.join(l.rstrip() for l in f.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    render(json.loads(a.data.read_text()), a.out)
