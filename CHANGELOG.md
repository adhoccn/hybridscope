# Changelog

## v2.3.0（2026-10-10）

> 混合属性聚类的诊断工作台：不只给出特征权重，还给出**这个权重能不能信**。
> A diagnostic workbench for mixed-attribute clustering: a standard error and a plain-language
> verdict for every learned feature weight.

### 安装

```bash
# 从本 Release 附件安装（wheel 已在下方 Assets）
pip install hybridscope-2.3.0-py3-none-any.whl

# 需要 Streamlit 工作台 / HTTP 服务时
pip install "hybridscope-2.3.0-py3-none-any.whl[web]"
pip install "hybridscope-2.3.0-py3-none-any.whl[api]"

# 或从源码
pip install .
```

装完后命令行入口 `hybridscope` 可用：

```bash
hybridscope data.csv --template grades
hybridscope --help
```

要求 Python ≥ 3.10；在 Python 3.13 / numpy 2.5 / pandas 3.0 / scikit-learn 1.9 下测试通过。

### 它回答三个问题

| 问题 | 做法 |
|---|---|
| **能不能去噪？** | 沿样本轴的一维小波去噪不是置换等变的——行号只是文件写入顺序。τ̂ = acf(8)/acf(1) 的**形状**判据 + 排序键检验给出「可以去噪／禁止去噪／无法判定」，无需标签 |
| **权重能不能信？** | 复合似然下的 Godambe 三明治方差给出每个权重的标准误，对「等权」零假设做 z 检验，BH-FDR 校正后给出绿／黄／红，并按 CRB ∝ 1/N 外推所需样本量 |
| **批次之间有没有变？** | 每次分析封成带数据指纹的快照；Δw 与**它自己的**标准误比，区分「真漂移 / 只是噪声 / 数据先变了 / 口径不可比」四种结论 |

### v2.3 新增

- **字段口径统一** `schema.py`：跨批次对比常常不是输在统计上，而是输在口径上——同一门课一学期导出「平时作业／大作业」，下一学期导出「测验1–4／报告1–4」。规范字段 + 折叠规则（mean/wmean/sum/first）+ **可比性声明**，把「列变了」和「世界变了」分开。
- **算力预算** `speed.py`：两处无损加速（缓存均匀权重距离矩阵、kNN 索引不再复制矩阵）使 PSO 耗时下降 36–42%；`--speed auto|fast|thorough` 按样本量缩放，`--dry-run` 先估成本再跑。
- **HTTP 服务** `serve.py`：`/analyze`、`/analyze/text`、`/snapshot`、`/compare`、`/unify`、`/advice`，供脚本与业务系统直接调用；NaN/Inf 输出为 `null`，连续目标必须切档并剔除原列（防标签泄漏）。
- **两个新模板**：`survey`（李克特量表：反向计分、直线作答检测、维度 α）、`ops`（传感器表：重复测量、类别不平衡、量纲跨度）。
- **数据体检修正**：离散量表不再剔重（150 份问卷保留 134 行而非 97 行）。

### 一个真实结论

用两学期成绩数据验证：回归确认两学期都是 60/40 结构（2025 秋 0.599·过程 + 0.401·项目；2026 春 0.600·过程 + 0.398·项目），因此「平时作业 ↔ 测验1–4」「大作业 ↔ 报告1–4」是同一对构念，可以比；而「专业」列一学期存专业名、一学期存行政班名，取值域重合 **0.00**，判定为**口径不可比**并排除。

只在可比字段上比较：过程考核权重 0.562 → 0.372，项目考核 0.438 → 0.628（z = ∓3.31，q = 0.001），是真漂移，与项目考核均分 66.9 → 76.7（SMD = +0.57）自洽。把不可比的分组列放进去会得出另一个数——因为单纯形上一个权重从 0 涨到 0.64 必然挤占其他权重，所以跨期对比只跑可比字段，报告里写明原因。

### 自检

```bash
python tests/test_schema.py          # 38 项
python tests/test_serve_speed.py     # 27 项
python tests/test_templates.py       # 32 项
python tests/test_fdr_stability.py   # 29 项
python tests/test_report_monitor.py  # 29 项
python tests/test_prep_advisor.py    # 17 项
```

共 **172 项，全部通过**。

### 链接

- 文档：`README.md`（方法与用法）、`PRODUCT_DESIGN.md`（设计取舍）、`DEPLOY.md`（自建 Web 服务）
- 案例：`cases/learning_analytics/`（脚本与报告；成绩明细不公开）

### License

MIT
