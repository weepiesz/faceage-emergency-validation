"""Reformat the metric tables out of the ml_final.py log.

Parses results/ml_modeling/ml_final.log and reprints, in a fixed-width layout, the six metrics
plus the calibration intercept and slope for every combination of scheme (MICE), variable mode
(raw, corrected), classifier (RF, LR) and feature set (A-I). Purely a readability aid — it
computes nothing.

Input  : results/ml_modeling/ml_final.log     (run ml_final.py first)
Output : printed to stdout"""
from pathlib import Path

# 仓库根目录（scripts/<sub>/x.py -> parents[2]）
STUDY = Path(__file__).resolve().parents[2]

import re
PAT = re.compile(r'^\s*(MICE|CC)\s+(raw|corrected)\s+(RF|LR)\s+([A-I])\s')
CAL = re.compile(r'([-+]?[\d.]+) \(([-+]?[\d.]+), ([-+]?[\d.]+)\)\s+'
                 r'([-+]?[\d.]+) \(([-+]?[\d.]+), ([-+]?[\d.]+)\)')
res = {}
st = False
for ln in open(str(STUDY / 'results' / 'ml_modeling' / 'ml_final.log'), encoding='utf-8'):
    if '固定 90%' in ln:
        st = True; continue
    if not st: continue
    m = PAT.match(ln)
    if not m: continue
    toks = [t for t in ln[m.end():].split() if '±' in t][:6]
    c = CAL.search(ln)
    if len(toks) < 6 or not c: continue
    res[(m.group(1), m.group(2), m.group(3), m.group(4))] = \
        [t.replace('±', ' ± ') for t in toks] + list(c.groups())
print('解析 %d 行' % len(res))
for sch in ['MICE']:
    for mode in ['raw', 'corrected']:
        print('\n' + '=' * 118)
        print('  %s / %s' % (sch, mode))
        print('=' * 118)
        for clf in ['RF', 'LR']:
            for s in 'ABCDEFGHI':
                k = (sch, mode, clf, s)
                if k not in res: continue
                v = res[k]
                print('%-3s %-2s | %-13s | %-13s | %-13s | %-13s | %-13s | %-13s | %-22s | %-22s'
                      % (clf, s, v[0], v[1], v[2], v[3], v[4], v[5],
                         '%s (%s, %s)' % (v[6], v[7], v[8]),
                         '%s (%s, %s)' % (v[9], v[10], v[11])))
