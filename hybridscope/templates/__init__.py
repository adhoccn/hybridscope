# -*- coding: utf-8 -*-
"""领域模板：把某类业务的字段语义、默认处理与解释话术固化下来。

每个模板提供: 角色识别(id/cat/num/target)、特征切分、结果解释、注意事项。
  grades  学情分型（成绩表）
  survey  问卷/量表（李克特题目 → 维度分）
  ops     设备运维（传感器通道 → 设备状态）
"""

from . import grades   # noqa: F401
from . import survey   # noqa: F401
from . import ops      # noqa: F401

TEMPLATES = {"grades": grades, "survey": survey, "ops": ops}


def get(name: str):
    return TEMPLATES.get(name)


def names() -> list[str]:
    return sorted(TEMPLATES)
