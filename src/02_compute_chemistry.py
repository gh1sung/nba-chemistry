import csv
from collections import defaultdict
import unicodedata

# ============================================================
# LOAD DATA
# ============================================================

def load_csv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

lineups = load_csv('/mnt/user-data/uploads/nba_2man_lineups.csv')
bpm_2324 = load_csv('/mnt/user-data/uploads/bpm_2023_24__1_.csv')
bpm_2425 = load_csv('/mnt/user-data/uploads/bpm_2024_25.csv')

# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_name(name):
    """Strip accents, normalize unicode, lowercase for matching."""
    # NFD decomposition splits accented chars into base + combining mark
    nfkd = unicodedata.normalize('NFKD', name)
    # Keep only non-combining characters (strips accents)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_name.strip()

def build_matching_keys(full_name, team, season):
    """Generate multiple lookup keys for a player to maximize matching."""
    name = normalize_name(full_name)
    parts = name.split()
    if len(parts) < 2:
        return []
    
    first_init = parts[0][0].upper()
    keys = []
    
    # Key 1: first initial + last word (handles most names)
    keys.append((first_init, parts[-1], team, season))
    
    # Key 2: first initial + everything after first name (handles multi-word last names)
    if len(parts) > 2:
        full_last = ' '.join(parts[1:])
        keys.append((first_init, full_last, team, season))
    
    # Key 3: same but with TOT team (for traded players)
    keys.append((first_init, parts[-1], 'TOT', season))
    if len(parts) > 2:
        keys.append((first_init, ' '.join(parts[1:]), 'TOT', season))
    
    return keys

# ============================================================
# BUILD BPM LOOKUP
# ============================================================

bpm_lookup = {}

for season_tag, bpm_data in [('2023-24', bpm_2324), ('2024-25', bpm_2425)]:
    for row in bpm_data:
        name = row['PLAYER_NAME'].strip()
        team = row['TEAM'].strip()
        try:
            bpm = float(row['BPM'])
            obpm = float(row['OBPM']) if row['OBPM'] else None
            dbpm = float(row['DBPM']) if row['DBPM'] else None
            mp = int(row['MP']) if row['MP'] else 0
            gp = int(row['GP']) if row['GP'] else 0
            vorp = float(row['VORP']) if row['VORP'] else None
        except (ValueError, TypeError):
            continue
        
        info = {'bpm': bpm, 'obpm': obpm, 'dbpm': dbpm, 'mp': mp, 'gp': gp, 
                'vorp': vorp, 'name': name, 'team': team}
        
        for key in build_matching_keys(name, team, season_tag):
            # Prefer team-specific over TOT, and higher-minute entries
            existing = bpm_lookup.get(key)
            if existing is None or (existing['team'] == 'TOT' and team != 'TOT') or mp > existing['mp']:
                bpm_lookup[key] = info

print(f"BPM lookup: {len(bpm_lookup)} entries")

# ============================================================
# PARSE LINEUPS
# ============================================================

def parse_abbrev(abbrev_name):
    """Parse 'N. Jokic' or 'S. Gilgeous-Alexander' -> (initial, last_name)"""
    name = normalize_name(abbrev_name.strip())
    # Split on ". " to get "N" and "Jokic"
    parts = name.split('. ', 1)
    if len(parts) == 2:
        return parts[0][0].upper(), parts[1].strip()
    # Fallback
    parts2 = name.split()
    if len(parts2) >= 2:
        return parts2[0][0].upper(), parts2[-1]
    return None, None

def find_bpm(init, last, team, season):
    """Try multiple matching strategies."""
    # Direct match
    result = bpm_lookup.get((init, last, team, season))
    if result:
        return result
    # TOT match
    result = bpm_lookup.get((init, last, 'TOT', season))
    if result:
        return result
    return None

results = []
matched = 0
unmatched_names = defaultdict(int)

for row in lineups:
    group = row['GROUP_NAME'].strip('"')
    parts = group.split(' - ')
    if len(parts) != 2:
        continue
    
    team = row['TEAM']
    season = row['SEASON']
    
    try:
        pair_min = float(row['MIN'])
        pair_pm = float(row['PLUS_MINUS'])
        gp = int(row['GP'])
        pair_w = int(row['W'])
        pair_l = int(row['L'])
    except (ValueError, TypeError):
        continue
    
    if pair_min <= 0:
        continue
    
    init_a, last_a = parse_abbrev(parts[0].strip())
    init_b, last_b = parse_abbrev(parts[1].strip())
    
    if not init_a or not init_b:
        continue
    
    stats_a = find_bpm(init_a, last_a, team, season)
    stats_b = find_bpm(init_b, last_b, team, season)
    
    if not stats_a:
        unmatched_names[f"{parts[0].strip()} ({team}, {season})"] += 1
        continue
    if not stats_b:
        unmatched_names[f"{parts[1].strip()} ({team}, {season})"] += 1
        continue
    
    matched += 1
    
    pair_pm_per48 = (pair_pm / pair_min) * 48
    a_bpm = stats_a['bpm']
    b_bpm = stats_b['bpm']
    expected = a_bpm + b_bpm
    chemistry = pair_pm_per48 - expected
    
    results.append({
        'player_a': stats_a['name'],
        'player_b': stats_b['name'],
        'team': team,
        'season': season,
        'gp': gp,
        'w': pair_w,
        'l': pair_l,
        'shared_min': round(pair_min, 1),
        'pair_pm': pair_pm,
        'pair_pm_per48': round(pair_pm_per48, 2),
        'a_bpm': a_bpm,
        'a_obpm': stats_a['obpm'],
        'a_dbpm': stats_a['dbpm'],
        'b_bpm': b_bpm,
        'b_obpm': stats_b['obpm'],
        'b_dbpm': stats_b['dbpm'],
        'expected': round(expected, 2),
        'chemistry': round(chemistry, 2),
    })

print(f"Matched: {matched}")
print(f"Unmatched players (top 15):")
for name, count in sorted(unmatched_names.items(), key=lambda x: -x[1])[:15]:
    print(f"  {name}: {count} pairs missed")

# ============================================================
# FILTER & KEY RESULTS
# ============================================================

MIN_MINUTES = 500
filtered = [r for r in results if r['shared_min'] >= MIN_MINUTES]
filtered.sort(key=lambda x: x['chemistry'], reverse=True)

print(f"\nFiltered pairs (>={MIN_MINUTES} min): {len(filtered)}")

# === CASE STUDIES (the ones that were missing before) ===
target_pairs = [
    ("Jokic", "Murray"),
    ("Tatum", "Brown"),
    ("Embiid", "Maxey"),
    ("James", "Davis"),
    ("Curry", "Green"),  # Draymond, not Josh
    ("Gilgeous-Alexander", "Holmgren"),
    ("Doncic", "Irving"),
    ("Edwards", "Gobert"),
    ("LaVine", "DeRozan"),
    ("Brunson", "Towns"),
    ("Haliburton", "Siakam"),
    ("Mitchell", "Garland"),
    ("Booker", "Durant"),
    ("Morant", "Jackson"),
    ("Young", "Murray"),  # Trae + Dejounte
    ("Wembanyama", "Paul"),
    ("Fox", "Sabonis"),
    ("Lillard", "Antetokounmpo"),
]

def normalize_for_search(s):
    return normalize_name(s).lower()

print("\n" + "="*90)
print("CASE STUDY PAIRS (v3 — fixed name matching)")
print("="*90)

for last_a, last_b in target_pairs:
    la = normalize_for_search(last_a)
    lb = normalize_for_search(last_b)
    
    found = [r for r in results
             if (la in normalize_for_search(r['player_a']) and lb in normalize_for_search(r['player_b'])) or
                (lb in normalize_for_search(r['player_a']) and la in normalize_for_search(r['player_b']))]
    
    if found:
        for r in found:
            if r['shared_min'] >= MIN_MINUTES:
                rank = sum(1 for x in filtered if x['chemistry'] > r['chemistry']) + 1
                pct = (1 - rank / len(filtered)) * 100
                pct_str = f"(#{rank}/{len(filtered)}, {pct:.0f}th pctl)"
            else:
                pct_str = f"(below {MIN_MINUTES}min threshold)"
            
            print(f"\n  {r['player_a']} + {r['player_b']} ({r['team']}, {r['season']})")
            print(f"    Min: {r['shared_min']:.0f} | GP: {r['gp']} | W-L: {r['w']}-{r['l']}")
            print(f"    Pair ±/48: {r['pair_pm_per48']:+.1f}  |  A BPM: {r['a_bpm']:+.1f}  |  B BPM: {r['b_bpm']:+.1f}")
            print(f"    Expected: {r['expected']:+.1f}  |  CHEMISTRY: {r['chemistry']:+.1f}  {pct_str}")
    else:
        print(f"\n  {last_a} + {last_b}: NOT FOUND")

# ============================================================
# PHASE 2: SPLIT-HALF RELIABILITY
# ============================================================
# 
# The core question: is chemistry REAL or NOISE?
# Test: do pairs that show high chemistry early in the season
# continue to show high chemistry later?
#
# We can't split by game date with this data (we have season totals).
# But we CAN test YEAR-OVER-YEAR stability for pairs that appear
# in both 2023-24 and 2024-25.
#
# If chemistry(pair, 2023-24) correlates with chemistry(pair, 2024-25),
# then we're measuring something persistent — not just noise.

print("\n" + "="*90)
print("PHASE 2: YEAR-OVER-YEAR CHEMISTRY STABILITY")
print("="*90)

# Build lookup: (player_a, player_b, season) -> chemistry
# Normalize pair order so (A,B) and (B,A) map to same key
pair_chem = {}
for r in results:
    # Sort names alphabetically to create canonical pair key
    pair_key = tuple(sorted([normalize_for_search(r['player_a']), 
                              normalize_for_search(r['player_b'])]))
    season_key = (pair_key, r['season'])
    pair_chem[season_key] = r

# Find pairs that appear in BOTH seasons with >= 300 min each
MIN_MIN_YOY = 300  # lower threshold for year-over-year since we need overlap
yoy_pairs = []

seen_2324 = {k: v for k, v in pair_chem.items() if k[1] == '2023-24' and v['shared_min'] >= MIN_MIN_YOY}
seen_2425 = {k: v for k, v in pair_chem.items() if k[1] == '2024-25' and v['shared_min'] >= MIN_MIN_YOY}

for (pair_key, _), data_2324 in seen_2324.items():
    match_key = (pair_key, '2024-25')
    if match_key in seen_2425:
        data_2425 = seen_2425[match_key]
        yoy_pairs.append({
            'player_a': data_2324['player_a'],
            'player_b': data_2324['player_b'],
            'team_2324': data_2324['team'],
            'team_2425': data_2425['team'],
            'min_2324': data_2324['shared_min'],
            'min_2425': data_2425['shared_min'],
            'chem_2324': data_2324['chemistry'],
            'chem_2425': data_2425['chemistry'],
        })

print(f"\nPairs appearing in both seasons (>={MIN_MIN_YOY} min each): {len(yoy_pairs)}")

if len(yoy_pairs) >= 10:
    # Compute Pearson correlation
    x = [p['chem_2324'] for p in yoy_pairs]
    y = [p['chem_2425'] for p in yoy_pairs]
    n = len(x)
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    cov_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    std_x = (sum((xi - mean_x)**2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y)**2 for yi in y) / n) ** 0.5
    
    if std_x > 0 and std_y > 0:
        pearson_r = cov_xy / (std_x * std_y)
    else:
        pearson_r = 0
    
    # Spearman rank correlation
    def rank_values(vals):
        sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
        return ranks
    
    ranks_x = rank_values(x)
    ranks_y = rank_values(y)
    
    mean_rx = sum(ranks_x) / n
    mean_ry = sum(ranks_y) / n
    cov_rxy = sum((ranks_x[i] - mean_rx) * (ranks_y[i] - mean_ry) for i in range(n)) / n
    std_rx = (sum((r - mean_rx)**2 for r in ranks_x) / n) ** 0.5
    std_ry = (sum((r - mean_ry)**2 for r in ranks_y) / n) ** 0.5
    
    if std_rx > 0 and std_ry > 0:
        spearman_rho = cov_rxy / (std_rx * std_ry)
    else:
        spearman_rho = 0
    
    print(f"\n  Pearson r:   {pearson_r:.3f}")
    print(f"  Spearman ρ:  {spearman_rho:.3f}")
    print(f"  n pairs:     {n}")
    
    # Interpretation
    if abs(pearson_r) < 0.1:
        interp = "Very weak — chemistry appears mostly noise at this level"
    elif abs(pearson_r) < 0.25:
        interp = "Weak but present — some persistent signal exists"
    elif abs(pearson_r) < 0.4:
        interp = "Moderate — meaningful persistent chemistry signal"
    elif abs(pearson_r) < 0.6:
        interp = "Strong — chemistry is substantially persistent year-over-year"
    else:
        interp = "Very strong — highly stable signal"
    
    print(f"  Interpretation: {interp}")
    
    # Show pairs with biggest chemistry CHANGES (positive to negative or vice versa)
    yoy_pairs.sort(key=lambda p: abs(p['chem_2324'] - p['chem_2425']), reverse=True)
    
    print(f"\n  MOST CHANGED CHEMISTRY (year-over-year):")
    print(f"  {'Player A':<22} {'Player B':<22} {'2324':>6} {'2425':>6} {'Δ':>6}")
    print(f"  {'-'*64}")
    for p in yoy_pairs[:15]:
        delta = p['chem_2425'] - p['chem_2324']
        print(f"  {p['player_a']:<22} {p['player_b']:<22} {p['chem_2324']:>+6.1f} {p['chem_2425']:>+6.1f} {delta:>+6.1f}")
    
    # Show most STABLE chemistry pairs
    yoy_pairs.sort(key=lambda p: abs(p['chem_2324'] - p['chem_2425']))
    
    print(f"\n  MOST STABLE CHEMISTRY (year-over-year):")
    print(f"  {'Player A':<22} {'Player B':<22} {'2324':>6} {'2425':>6} {'Δ':>6}")
    print(f"  {'-'*64}")
    for p in yoy_pairs[:15]:
        delta = p['chem_2425'] - p['chem_2324']
        print(f"  {p['player_a']:<22} {p['player_b']:<22} {p['chem_2324']:>+6.1f} {p['chem_2425']:>+6.1f} {delta:>+6.1f}")
    
    # Stratify by minutes: do high-minute pairs show more stability?
    high_min = [p for p in yoy_pairs if p['min_2324'] >= 1000 and p['min_2425'] >= 1000]
    if len(high_min) >= 10:
        hx = [p['chem_2324'] for p in high_min]
        hy = [p['chem_2425'] for p in high_min]
        hn = len(hx)
        hmx = sum(hx)/hn; hmy = sum(hy)/hn
        hcov = sum((hx[i]-hmx)*(hy[i]-hmy) for i in range(hn))/hn
        hsx = (sum((xi-hmx)**2 for xi in hx)/hn)**0.5
        hsy = (sum((yi-hmy)**2 for yi in hy)/hn)**0.5
        hr = hcov/(hsx*hsy) if hsx>0 and hsy>0 else 0
        print(f"\n  High-minute pairs (>= 1000 min both seasons): n={hn}")
        print(f"  Pearson r (high-min only): {hr:.3f}")

# ============================================================
# SAVE EVERYTHING
# ============================================================

filtered.sort(key=lambda x: x['chemistry'], reverse=True)
with open('/home/claude/chemistry_v3.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=filtered[0].keys())
    writer.writeheader()
    writer.writerows(filtered)

if yoy_pairs:
    with open('/home/claude/chemistry_yoy.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=yoy_pairs[0].keys())
        writer.writeheader()
        writer.writerows(yoy_pairs)

print(f"\n✅ Saved chemistry_v3.csv ({len(filtered)} pairs)")
print(f"✅ Saved chemistry_yoy.csv ({len(yoy_pairs)} year-over-year pairs)")
