# HybridScope

混合属性数据聚类的诊断工具箱：不只给出特征权重，还给出**这个权重能不能信**。

混合属性数据聚类里有一条被广泛采用的流水线：先对数值块做小波去噪，再用信息增益加权的
kNN 聚类，最后用 PSO 在单纯形上搜特征权重。本项目把这条流水线完整实现（`knn_hybrid/`），
然后追问它的每个环节**在什么条件下可信**，并把结论做成可直接用在别人表上的工具
（`hybridscope/`）：一个 sklearn 兼容的估计器、一个命令行工具，和一个 Streamlit 工作台
（支持一次上传多个 CSV／Excel、按行追加或按键对齐、把簇编号翻译成可读的簇画像、把诊断
结论直接画成图），已部署为可远程访问的 Web 服务（`deploy/`）。
`cases/learning_analytics/` 给出学情分析实例（仅脚本与报告，学生原始成绩不公开）。

- **去噪能不能用**：沿样本轴的一维小波去噪**不是置换等变的**——它把行号当成有意义的
  位置，而表格的行序只是文件写入顺序，因此去噪可能"看起来有道理地"损坏数据。项目给出
  一个不需要标签、只需几行代码的预检规则（沿样本轴自相关剖面 + 排序键检验），
  明确划分"可以去噪／禁止去噪／无法判定"三种结论。
- **权重能不能信**：在复合似然下用 Godambe 三明治方差给出每个权重的标准误，以 z 值相对
  均匀权重零假设做绿／黄／红判定，并由 CRB ∝ 1/N 外推"这个权重要多少样本才可信"。
- **批次之间有没有变**：把每次分析封成带数据指纹的快照，再按"Δw 是否超过它自己的标准误"
  判定漂移，把"真漂移 / 只是噪声 / 数据先变了 / 口径不可比"四种结论分开。

> **一句话（GitHub About / 汇报用）**：给每个学到的特征权重配一个标准误和一句人话结论，
> 附 sklearn 估计器、CLI 与已部署的 Streamlit 诊断工作台。

### Project summary (EN)

Mixed-attribute clustering is commonly run as a three-stage pipeline: wavelet-denoise the numeric
block, cluster with an information-gain-weighted kNN, then search feature weights on the simplex
with PSO. This repository implements that pipeline end to end (`knn_hybrid/`) and then asks, for
each of its stages, **under what conditions the result can be trusted** — and ships the answer as
a tool you can point at someone else's table.

Three questions are answered per dataset, not per method paper:

* *May I denoise?* 1-D wavelet denoising applied down the sample axis is **not
  permutation-equivariant**: it treats row index as a meaningful coordinate, while a table's row
  order is merely the order in which the file was written, so denoising can damage the data while
  looking justified. A label-free, few-lines pre-check — the autocorrelation profile along the
  sample axis plus a sort-key test — returns authorise / forbid / indeterminate.
* *May I trust this weight?* A Godambe sandwich variance gives a standard error per weight; the z
  value against the equal-weight null is rendered green/amber/red; the sample size each weight
  would need follows from CRB ∝ 1/N.
* *Did anything change?* Every run can be sealed into a snapshot carrying a data fingerprint, and
  two snapshots are compared by asking whether each Δw exceeds its own standard error.

On top of the estimator sits a product layer, `hybridscope/` — an sklearn-compatible estimator, a
CLI, and a Streamlit workbench (multi-file upload, row-append or key-align merging, readable
cluster profiles, charts for every diagnosis), deployed as a remote web service (`deploy/`).
`cases/learning_analytics/` is a worked example on two semesters of course grades (scripts and
report only; the grade rosters are not public).

## Repository layout

```
knn_hybrid/         pipeline implementation (S1 denoise, S2 weighted kNN, S3 PSO)
  data.py denoise.py model.py pso.py pipeline.py baselines.py metrics.py
  run_experiments.py
hybridscope/        product layer: SDK + CLI + Streamlit workbench + HTTP service
  io.py prep.py advisor.py diagnostics.py stability.py estimator.py
  schema.py monitor.py report.py speed.py serve.py cli.py app.py
  templates/        domain templates: grades (学情分型), survey (量表), ops (设备工况)
deploy/             server deployment kit (Docker / systemd + nginx), see DEPLOY.md
cases/              worked examples (scripts and reports only; raw data not public)
tests/              self-checks, run each file directly: python tests/test_*.py
```

## Quick start

```bash
git clone https://github.com/adhoccn/hybridscope.git
cd hybridscope
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS
```

Tested with Python 3.13, numpy 2.5.0, pandas 3.0.3, scipy 1.18.0,
scikit-learn 1.9.0, PyWavelets 1.8.0, matplotlib 3.11.0.

```bash
streamlit run hybridscope/app.py        # web workbench
python -m hybridscope.cli --help        # command line
python -m hybridscope.serve             # HTTP API (needs fastapi + uvicorn)
```

## Workbench

`hybridscope/` fits the hybrid-attribute model and then reports, for every learned feature weight,
**whether that weight can be trusted**: a Godambe sandwich standard error gives a z value against
the "all features equally important" null, which is rendered as green / amber / red. It also
advises whether the denoising front-end helps *on this data* and how many samples each weight
would need to reach a target precision.

The workbench takes CSV/Excel uploads — several files at once, either appended by row, aligned on a
key column, or analysed one by one — draws the full report as charts, labels each cluster with a
profile instead of a bare index, and exports the result as JSON. `deploy/` holds the scripts used
to run it on a self-hosted server (see `DEPLOY.md`); `cases/learning_analytics/` is a worked
example on two semesters of course grades (scripts and report only — the grade rosters are not
public).

### Data health check and the τ̂ pre-test

Two things stood between v1 and a table someone else actually owns.

**Data health check** (`hybridscope/prep.py`). v1 called `df.dropna()`, which throws away a whole
row because one cell is empty — on the shipped grade files that costs 40 of 172 students. `Prep`
replaces it with per-column repair and reports every decision: identifier columns (学号/编号,
nearly unique per row) are dropped, columns that are constant or more than 40% missing are
dropped, numeric columns are median-imputed and clipped at the 1st/99th percentile, categorical
missing values become their own level and rare levels (fewer than 3 cases) are merged into 其他,
duplicate rows are removed (except on discrete tables, where identical rows are several people
answering alike), and columns whose scales differ by more than 10× are standardised (the weighted
distance is scale-sensitive). The fitted object is reusable: `transform()` applies the same rules
to new data, so prediction does not silently drift.

**The τ̂ pre-test** (`hybridscope/advisor.py`). v1 decided whether to denoise from the lag-1
autocorrelation alone (≥0.3 → denoise). That is exactly the statistic shown to be insufficient: a
genuinely smooth series and a table stored in class-sorted order both have large lag-1 values. v2
uses the *shape* of the autocorrelation profile instead, τ̂ = acf(8)/acf(1), with τ̂ ≥ 0.8 →
denoise, τ̂ ≤ 0.5 → do not, and the band 0.5 < τ̂ < 0.8 reported honestly as indeterminate and
resolved conservatively — plus a sort-key test that flags tables whose row order comes from a
grouping key (adjacent rows share the group far more often than chance). When labels are available
the cross-validated accuracy comparison still decides, and any disagreement with τ̂ is surfaced
as a caveat rather than hidden.

```bash
python -m hybridscope.cli grades.csv --template grades     # 学情分型模板
python -m hybridscope.cli data.csv --prep off              # 退回旧行为
python tests/test_prep_advisor.py                          # 17 项自检
python tests/test_fdr_stability.py                         # 29 项自检
python tests/test_report_monitor.py                        # 29 项自检
```

### Multiple-testing control and credibility without labels

**BH-FDR over the weights** (`hybridscope/diagnostics.py`). *m* weights are tested at once, so
reading each |z| ≥ 2 on its own inflates the false-positive rate to roughly 1−(1−α)^*m*. Every row
now carries a two-sided p value and a Benjamini–Hochberg q value; the traffic light follows the
corrected verdict (green needs q ≤ α and |z| ≥ 2). On the shipped autumn grade file this changes
the answer: 平时作业 has z = 2.39 and looks "trustworthy" uncorrected, but q = 0.068 after
correction — not established. The Hessian is also checked before any of it is believed: materially
negative curvature or a condition number above 10⁶ is reported as *not identifiable* and every
row is forced to red, because a sandwich variance computed at a non-stationary point is
meaningless. An optional gradient bootstrap (`--boot 200`) re-samples the per-sample score
contributions to confirm the sandwich SE is not driven by a handful of rows.

**Credibility without labels** (`hybridscope/stability.py`). The sandwich route needs `y`, so an
unlabelled table used to get clustering and nothing else. It now fits weights from the silhouette
objective and then asks whether the resulting partition is real, by **split-half transfer**: split
the sample in two, cluster each half, carry the B-half points over to the nearest A-half cluster
label using the weighted distance, and score with the chance-corrected ARI. Two null distributions
are computed by repeating the same procedure on surrogates — a *strict* one that permutes each
column independently (keeps the marginals, removes only the dependence between columns) and a
*lenient* one that replaces each column by a normal of the same mean and variance. Green requires
passing the strict null; passing only the lenient one is reported as amber with the reason spelled
out, because multimodal marginals (grades that come in natural bands, machines that run in
distinct operating modes) produce clumping on their own and raise the strict baseline. A
leave-one-feature-out pass then reports how much stability each feature carries.

```bash
python -m hybridscope.cli data.csv                # 无标签：轮廓系数拟合 + 稳定性
python -m hybridscope.cli data.csv --fdr-alpha 0.10 --boot 200
python -m hybridscope.cli data.csv --no-stability # 大表提速
```

**Reports and drift** (`hybridscope/report.py`, `hybridscope/monitor.py`). A single analysis is
not worth much; what is worth something is a comparison. Every run can be sealed into a
**snapshot** carrying a data fingerprint (`hash_rows` is order-sensitive, `hash_pool` is not, so
"same people, new row order" is distinguishable from "different people"), the full reproduction
config, the weights and their standard errors. Two snapshots are then compared by asking whether
each Δw exceeds *its own* standard error: `se(Δwᵢ) = sqrt(se_A,ᵢ² + se_B,ᵢ²)`, per-feature z with
BH-FDR, plus a global χ² with `m − 1` degrees of freedom (weights live on a simplex, so ΣΔw = 0).
Three outcomes are kept apart on purpose: identical fingerprints mean Δw can only come from the
optimiser and is **not** drift; changed data with insignificant Δw means the conclusion survived;
significant Δw is real drift. Columns whose value domains barely overlap (an ID column, or a
"major" column that holds major names in one batch and class names in the next) are reported as
*not comparable* rather than as drift — same name, different meaning.

```bash
python -m hybridscope.cli data.csv --name 2026春 \
       --snapshot snap.json --report report.html
python -m hybridscope.cli next.csv --name 2026秋 --compare snap.json
python tests/test_report_monitor.py                       # 29 项自检
```

The HTML report is self-contained (charts are embedded PNGs, no network needed) and ends with an
action per weight — keep it, collect more samples, or drop it — rather than a bare z value.

### Field mapping, compute budgets, a service, and two more templates

**Field mapping** (`hybridscope/schema.py`). Cross-batch comparisons mostly fail on bookkeeping,
not on statistics: the same course exports "平时作业 / 大作业" in one term and "测验1–4 /
报告1–4" in the next. Feeding both to one model mixes "the columns changed" with "the world
changed", and the two cannot be separated afterwards. A `Schema` states what counts as the same
thing (canonical fields), how each batch folds onto them (`mean` / `wmean` / `sum` / `first`,
optional `scale`, optional `zero_as_missing` for "0 means absent"), and — crucially — which fields
are **not** comparable across batches. Before folding, `consistency()` checks the group:
standard-deviation ratio for scale, Cronbach α and mean inter-item correlation for whether the
items measure one thing. Folding four quizzes into "process score" when α = 0.25 is legal but
lossy, and the audit says so instead of averaging silently.

On the shipped grade files the mapping is confirmed by the data, not assumed: regressing 总评 on
the canonical scores gives 0.599·process + 0.401·project in 2025 autumn and 0.600·process +
0.398·project in 2026 spring (residual sd ≈ 0.29). The 60/40 structure is identical, so 平时作业 ↔
测验1–4 and 大作业 ↔ 报告1–4 are the same two constructs. The "专业" column is *not*: it holds
major names in spring (an elective taken by many majors) and class names in autumn (计算机25-x班),
overlap 0.00, so it is marked `comparable=False` and excluded from Δw. Comparing only the two
comparable fields, process falls 0.562 → 0.372 and project rises 0.438 → 0.628 (z = ∓3.31,
q = 0.001): real drift, and consistent with project scores rising 66.9 → 76.7 (SMD = +0.57).
Running the same comparison with the group column included gives a different number, because a
column going from weight 0 to 0.64 must take its share from the others on the simplex — the
comparison therefore runs on comparable fields only, and the report states why.

```bash
python cases/learning_analytics/make_unified.py          # 对齐两学期 + 审计
python -m hybridscope.cli a.csv b.csv --schema sch.json --unify long.csv
python -m hybridscope.cli a.csv --schema sch.json --profile 2026spring \
       --template grades --snapshot snap.json
python tests/test_schema.py                              # 38 项自检
```

**Compute budgets** (`hybridscope/speed.py`). Every PSO evaluation rebuilds the full n×n weighted
distance matrix, so cost grows roughly with n²·⁵: measured 1.2 s at n = 200 and 7.8 s at n = 400,
before optimisation. Two changes are exact and need no opt-in: the uniform-weight distance matrix
(recomputed on every evaluation although it never changes) is cached, and `knn_indices` no longer
copies the matrix to mask the diagonal. Together, PSO time drops 36–42%. Beyond that,
`--speed auto|fast|thorough` scales particles, iterations and stability replicates to the sample
size, and `--max-n` sub-samples the stability check — with the price stated: a smaller budget means
a coarser random search, and sub-sampling costs precision (the null distribution uses the same
sub-sample size, so the test stays calibrated). Actual budgets are written into the snapshot, and
`/compare` warns when the two sides used different ones. `--dry-run` estimates cost and memory
first and suggests a mode.

```bash
python -m hybridscope.cli big.csv --dry-run               # 先估再跑
python -m hybridscope.cli big.csv --speed fast --max-n 800
```

**Service layer** (`hybridscope/serve.py`). `python -m hybridscope.serve` exposes `/analyze`,
`/analyze/text`, `/snapshot`, `/compare`, `/unify` and `/advice` over HTTP, so a registry or MES
can call it directly. Two rules are enforced: NaN/Inf become `null` (a degenerate SE would
otherwise turn the whole response into a 500), and a continuous target (总评 with 43 distinct
values) is either banded on request (`band=3`, with the source column dropped — leaving it in
would leak the label and hand it a 0.41 weight) or at least reported as a warning.

**Two more templates** (`hybridscope/templates/`). `survey` handles Likert data: reverse-scored
items are flipped before averaging (skipping this makes the dimension cancel itself out — α drops
by more than 0.2 in the test), straight-lining is detected and optionally dropped, and dimension
scores carry their α. `ops` handles sensor tables: it detects repeated measures on the same unit
(rows are then not independent, so the sandwich SE is optimistic), flags severe class imbalance
(accuracy is meaningless when the minority class is 8%), and reports scale spread (1375× across
channels in the test). Related: `Prep` no longer drops duplicate rows on discrete tables, where
identical rows are several people answering alike — on a 150-response questionnaire this kept 134
rows instead of 97.

```bash
python -m hybridscope.cli survey.csv --template survey --dims dims.json \
       --drop-invalid --target 档位
python -m hybridscope.cli sensors.csv --template ops
python tests/test_templates.py                            # 32 项自检
python tests/test_serve_speed.py                          # 27 项自检
```

## Notes

* Random seeds are fixed per run and recorded in the snapshot, so re-running reproduces the
  reported numbers to the reported precision.
* Data fingerprints distinguish "same rows, new order" from "different rows"; two snapshots with
  identical fingerprints cannot be drifting, whatever Δw says.
* `DEPLOY.md` covers the self-hosted web service; `PRODUCT_DESIGN.md` records the design
  rationale behind the product layer.

## License

MIT (see `LICENSE`).
