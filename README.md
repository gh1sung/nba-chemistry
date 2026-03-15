# Quantifying NBA Player Chemistry: Measuring Interaction Effects Beyond Individual Talent

**Author:** Jihwan Sung (NYU, Data Science & Mathematics)  
**Date:** March 2026  
**Status:** Work in Progress — Phases 1–3 complete, Phase 4 (write-up) in progress

---

## Abstract

Basketball "chemistry" — the interaction effect between player pairs that goes beyond the sum of their individual contributions — is widely discussed but rarely quantified. This project introduces a chemistry metric for NBA two-player combinations, validates its persistence across seasons, and models what predicts it.

Using BPM (Box Plus-Minus) as an individual baseline and NBA.com lineup data from the 2023-24 and 2024-25 seasons, we define chemistry as the residual between a pair's actual plus-minus and the sum of their individual BPMs:

```
Chemistry(i,j) = Pair_PM_per48(i,j) − [BPM(i) + BPM(j)]
```

### Key Findings

1. **Chemistry is real and persistent.** Year-over-year correlation: r = 0.33 [0.24, 0.42] (95% CI) for all pairs (n=433), and r = 0.50 [0.27, 0.68] for high-minute pairs (n=62).

2. **Ball-dominance overlap is the strongest chemistry killer.** Combined usage rate correlates with chemistry at r = −0.30 (p < 10⁻²¹) after controlling for team quality. Two high-usage players together produce worse-than-expected results.

3. **Talent gaps hurt more with more shared minutes.** Forcing an imbalanced pair to play heavy minutes amplifies negative chemistry (interaction magnitude: −1.08).

4. **A gradient-boosted model explains ~13% of chemistry variance** (5-fold CV R² = 0.126 ± 0.029, permutation test p < 0.005), confirming that pair features contain real predictive signal.

---

## Motivation

Joel Embiid publicly stated the 2023-24 Philadelphia 76ers lacked "chemistry" despite Daryl Morey's analytics-driven talent acquisition. This tension — between accumulating individual talent and building a cohesive unit — motivates the central question: **Can we measure chemistry, and what drives it?**

This project builds on:
- **Bransen & Van Haaren (2020)**, "Player Chemistry: Striving for a Perfectly Balanced Soccer Team" — SSAC 2020 winner (Other Sports track). Defined chemistry as the residual of joint performance after accounting for individual quality in soccer.
- **Lutz (~2012)**, "A Cluster Analysis of NBA Players" — showed player-type combinations predict winning better than individual quality alone.
- **DRL-Shapley (2026)**, "Deep RL for NBA Player Valuation" — identified 127 significant player synergies using Shapley attribution, finding the Jokić-Murray pairing at +2.87 pts/100 possessions.

No prior work has rigorously defined and validated a pairwise chemistry metric specifically for basketball using publicly available data.

---

## Project Structure

```
nba-chemistry/
├── README.md                    # This file
├── src/
│   ├── 01_collect_data.py       # Phase 1: Pull lineup + player data from NBA API
│   ├── 02_compute_chemistry.py  # Phase 1: Compute raw chemistry scores (BPM baseline)
│   ├── 03_validate_stability.py # Phase 2: Year-over-year stability analysis
│   └── 04_model_predictors.py   # Phase 3: Feature engineering + modeling
├── data/
│   ├── nba_2man_lineups.csv     # Raw 2-man lineup data from NBA.com (2023-24, 2024-25)
│   ├── nba_player_advanced.csv  # NBA.com individual advanced stats
│   ├── bpm_2023_24.csv          # Basketball-Reference BPM data (2023-24)
│   └── bpm_2024_25.csv          # Basketball-Reference BPM data (2024-25)
├── results/
│   ├── chemistry_v3.csv         # Final chemistry scores (≥500 shared minutes)
│   ├── chemistry_yoy.csv        # Year-over-year stability pairs
│   └── chemistry_features.csv   # Feature-enriched dataset for modeling
└── figures/                     # Visualizations (TBD)
```

---

## Methodology

### Phase 1: Data Collection & Chemistry Computation

**Data Sources:**
- NBA.com Stats API: 2-man lineup combinations (GroupQuantity=2), regular season, totals
- Basketball-Reference: BPM (Box Plus-Minus) for individual baselines

**Why BPM, not NET_RATING?**
NBA.com's NET_RATING measures a player's team's net rating *while they're on court*. This double-counts teammate effects — Tatum's NET_RATING includes the benefit of playing with Brown, and vice versa. Summing them inflates the expected value and makes every star pair look like anti-chemistry.

BPM is estimated from box-score statistics via regression. It approximates a player's *individual* contribution without directly encoding who they play with, making it a cleaner baseline for the additive null hypothesis.

**Chemistry Formula:**
```
Chemistry(i,j) = (Plus_Minus / Minutes × 48) − [BPM(i) + BPM(j)]
```

**Filtering:** Minimum 500 shared minutes to reduce noise. Final dataset: 1,452 pairs across two seasons.

### Phase 2: Validation

**Year-over-year stability:** For 433 pairs appearing in both seasons (≥300 shared minutes each), we compute the Pearson correlation between 2023-24 and 2024-25 chemistry scores.

- All pairs: r = 0.330, 95% CI [0.239, 0.418]
- High-minute pairs (≥1000 min both seasons, n=62): r = 0.498, 95% CI [0.268, 0.679]

Chemistry is persistent — not just noise.

### Phase 3: Modeling Predictors

**Features tested (13 total):**
- Usage overlap and sum (ball-dominance)
- Offensive/defensive BPM differences and sums
- Two-way complementarity score
- Assist rate difference and maximum
- True shooting difference
- Age difference
- BPM max, min, difference, sum
- Shared minutes

**Significance testing:** Two-sided t-tests on Pearson correlations with Bonferroni-aware interpretation. Bootstrap 95% CIs (2,000 resamples) for all key estimates.

**Team-quality control:** Subtracted team-season mean chemistry to remove confound that pairs on good teams show inflated chemistry.

**Model:** Manual gradient-boosted regression trees (50 trees, max depth 3, learning rate 0.1). Evaluated via 5-fold cross-validation and permutation test (200 shuffles).

**Results:**
| Metric | Value |
|---|---|
| 5-Fold CV R² | 0.126 ± 0.029 |
| Permutation p-value | < 0.005 |
| RMSE improvement | 6.7% over baseline |

**Top predictors after team adjustment:**
| Feature | Adjusted r | p-value |
|---|---|---|
| Usage sum | −0.297 | < 10⁻²¹ |
| BPM max | −0.284 | < 10⁻²¹ |
| BPM difference | −0.193 | < 10⁻¹⁰ |
| Offensive BPM diff | −0.186 | < 10⁻⁹ |
| BPM min | −0.158 | < 10⁻⁷ |

---

## Selected Case Studies

| Pair | Season | Chemistry | Percentile | Interpretation |
|---|---|---|---|---|
| Ja Morant + JJJ | 2024-25 | +6.9 | 89th | Elite complementary pairing |
| Donovan Mitchell + Darius Garland | 2024-25 | +4.0 | 75th | Growing backcourt chemistry |
| Anthony Edwards + Rudy Gobert | 2023-24 | +3.4 | 70th | Driving + rim protection synergy |
| Jayson Tatum + Jaylen Brown | 2024-25 | +2.9 | 67th | Steady positive, championship-caliber |
| Joel Embiid + Tyrese Maxey | 2023-24 | −1.7 | 37th | Below average despite winning record |
| LeBron James + Anthony Davis | 2024-25 | −12.5 | 2nd | Severe decline, near bottom of league |
| DeRozan + LaVine (CHI) | 2023-24 | −12.6 | 2nd | Ball-dominance redundancy |
| Dejounte Murray + Trae Young | 2023-24 | −11.2 | 3rd | Dual-PG anti-chemistry |

---

## Limitations

1. **BPM is an imperfect individual baseline.** It's derived from box-score stats and doesn't capture off-ball value, spacing, or screening. A RAPM-based baseline would be more principled but requires play-by-play data processing.

2. **Confounders remain.** Coaches deploy pairs strategically — high-chemistry pairs may get favorable matchups. Our team-adjusted analysis mitigates but doesn't eliminate this.

3. **Two seasons of data.** Stability analysis uses only one year-over-year transition. More seasons would strengthen the validation.

4. **No tracking data.** Spatial complementarity (e.g., one player spaces the floor while the other drives) is not captured. This is a known limitation shared with most publicly available analytics.

5. **Correlation, not causation.** Chemistry scores are associational. A player trade that appears chemistry-positive based on our model may not replicate the estimated effect in a new team context.

---

## Future Work

- Compute split-half reliability (first 41 games vs. last 41) using game-level play-by-play data
- Build a RAPM-based individual baseline for cleaner chemistry estimation
- Extend to 3-player and 5-player lineup chemistry
- Incorporate player-tracking data for spatial complementarity features
- Predict chemistry for hypothetical pairings (trade evaluation tool)

---

## References

- Bransen, L. & Van Haaren, J. (2020). Player Chemistry: Striving for a Perfectly Balanced Soccer Team. *Proceedings of the 14th MIT Sloan Sports Analytics Conference.*
- Lutz, D. (~2012). A Cluster Analysis of NBA Players. *MIT Sloan Sports Analytics Conference.*
- Paper ID 137 (2026). Deep Reinforcement Learning for NBA Player Valuation: A Temporal Difference Approach with Shapley Attribution. *SSAC 2026 Basketball Track.*
- Myers, D. (2011). About Box Plus/Minus. *Basketball-Reference.com.*
- Rosenbaum, D. T. (2004). Measuring How NBA Players Help Their Teams Win. *82games.com.*

---

## Data Sources

- [NBA.com Stats API](https://www.nba.com/stats/) — Lineup data, player advanced stats
- [Basketball-Reference](https://www.basketball-reference.com/) — BPM, OBPM, DBPM, VORP
- All data from the 2023-24 and 2024-25 NBA regular seasons

---

## License

MIT License. Data sourced from NBA.com and Basketball-Reference under their respective terms of use.
