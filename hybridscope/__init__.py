# -*- coding: utf-8 -*-

from hybridscope.diagnostics import diagnose
from hybridscope.advisor import recommend
from hybridscope.estimator import HybridScopeEstimator
from hybridscope.io import load_table, split_columns

__all__ = [
    "HybridScopeEstimator", "diagnose", "recommend",
    "load_table", "split_columns",
]

# 版本随能力走：0.1 基线 → 2.0 数据体检+τ̂ → 2.1 FDR+无标签可信度
# → 2.2 报告导出 + 跨批次漂移追踪 → 2.3 字段映射 + 算力预算 + 服务层 + 新模板
__version__ = "0.2.3"
