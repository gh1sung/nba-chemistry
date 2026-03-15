"""
Phase 2: Year-over-Year Chemistry Stability Analysis
=====================================================
Tests whether chemistry scores persist across seasons.
If they do, chemistry is measuring a real signal — not noise.

Input:  results/chemistry_v3.csv
Output: results/chemistry_yoy.csv, console output with statistics
"""

import csv
import math
import random
import unicodedata

random.seed(42)
N_BOOT = 2000

def load_csv(path):
    with open(path, 'r') as f:
        return list(csv.DictReader(f))

def norm(name):
    nfkd = unicodedata.normalize('NFKD', name)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()

# Load all chemistry results (not just filtered)
# We use a lower minutes threshold for YOY since we need overlap
all_results = load_csv('../results/chemistry_v3.csv')  # adjust path as needed
print(f"Loaded {len(all_results)} chemistry pairs")

# Build lookup: canonical pair key -> {season: data}
pair_data = {}
for r in all_results:
    pair_key = tuple(sorted([norm(r['player_a']), norm(r['player_b'])]))
    season = r['season']
    if pair_key not in pair_data:
        pair_data[pair_key] = {}
    pair_data[pair_key][season] = r

# Find pairs in both seasons with >= 300 shared minutes each
MIN_MIN = 300
yoy_pairs = []

for pair_key, seasons in pair_data.items():
    if '2023-24' in seasons and '2024-25' in seasons:
        d1 = seasons['2023-24']
        d2 = seasons['2024-25']
        if float(d1['shared_min']) >= MIN_MIN and float(d2['shared_min']) >= MIN_MIN:
            yoy_pairs.append({
                'player_a': d1['player_a'],
                'player_b': d1['player_b'],
                'team_2324': d1['team'],
                'team_2425': d2['team'],
                'min_2324': float(d1['shared_min']),
                'min_2425': float(d2['shared_min']),
                'chem_2324': float(d1['chemistry']),
                'chem_2425': float(d2['chemistry']),
            })

print(f"Pairs in both seasons (>={MIN_MIN} min each): {len(yoy_pairs)}")

# Pearson correlation
x = [p['chem_2324'] for p in yoy_pairs]
y = [p['chem_2425'] for p in yoy_pairs]
n = len(x)

mx = sum(x)/n; my = sum(y)/n
cov = sum((x[i]-mx)*(y[i]-my) for i in range(n))/n
sx = (sum((xi-mx)**2 for xi in x)/n)**0.5
sy = (sum((yi-my)**2 for yi in y)/n)**0.5
r = cov/(sx*sy) if sx > 0 and sy > 0 else 0

# Bootstrap CI
boot_rs = []
for _ in range(N_BOOT):
    idx = [random.randint(0, n-1) for _ in range(n)]
    bx = [x[i] for i in idx]; by = [y[i] for i in idx]
    bmx = sum(bx)/n; bmy = sum(by)/n
    bc = sum((bx[i]-bmx)*(by[i]-bmy) for i in range(n))/n
    bsx = (sum((xi-bmx)**2 for xi in bx)/n)**0.5
    bsy = (sum((yi-bmy)**2 for yi in by)/n)**0.5
    boot_rs.append(bc/(bsx*bsy) if bsx > 0 and bsy > 0 else 0)

boot_rs.sort()
ci_lo = boot_rs[int(0.025 * N_BOOT)]
ci_hi = boot_rs[int(0.975 * N_BOOT)]

print(f"\nYear-over-Year Stability (all pairs, n={n}):")
print(f"  Pearson r = {r:.3f}  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

# High-minute subset
high_min = [p for p in yoy_pairs if p['min_2324'] >= 1000 and p['min_2425'] >= 1000]
if len(high_min) >= 10:
    hx = [p['chem_2324'] for p in high_min]
    hy = [p['chem_2425'] for p in high_min]
    hn = len(hx)
    hmx = sum(hx)/hn; hmy = sum(hy)/hn
    hcov = sum((hx[i]-hmx)*(hy[i]-hmy) for i in range(hn))/hn
    hsx = (sum((xi-hmx)**2 for xi in hx)/hn)**0.5
    hsy = (sum((yi-hmy)**2 for yi in hy)/hn)**0.5
    hr = hcov/(hsx*hsy) if hsx > 0 and hsy > 0 else 0
    
    hm_boot = []
    for _ in range(N_BOOT):
        idx = [random.randint(0, hn-1) for _ in range(hn)]
        bx = [hx[i] for i in idx]; by = [hy[i] for i in idx]
        bmx = sum(bx)/hn; bmy = sum(by)/hn
        bc = sum((bx[i]-bmx)*(by[i]-bmy) for i in range(hn))/hn
        bsx = (sum((xi-bmx)**2 for xi in bx)/hn)**0.5
        bsy = (sum((yi-bmy)**2 for yi in by)/hn)**0.5
        hm_boot.append(bc/(bsx*bsy) if bsx > 0 and bsy > 0 else 0)
    hm_boot.sort()
    
    print(f"\nHigh-minute pairs (>= 1000 min both seasons, n={hn}):")
    print(f"  Pearson r = {hr:.3f}  95% CI: [{hm_boot[int(0.025*N_BOOT)]:.3f}, {hm_boot[int(0.975*N_BOOT)]:.3f}]")

# Save
with open('../results/chemistry_yoy.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=yoy_pairs[0].keys())
    writer.writeheader()
    writer.writerows(yoy_pairs)

print(f"\nSaved chemistry_yoy.csv ({len(yoy_pairs)} pairs)")
