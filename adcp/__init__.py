"""ADCP 波束速度 / 姿态旋转 / 质量层级证据工具。

核心包划分:
- units:      领域量携带单位与坐标基准, 原始表示保留, 无法转换的值留在原位
- geometry:   波束几何 (beam <-> instrument), 四束唯一解与三束多解族
- attitude:   姿态插值 (360 度跨越展开, 时间缺口不跨)
- transforms: ship / earth 逐级旋转链, 坐标转换顺序进入指纹
- fingerprint: 派生指纹 (输入哈希 + 波束几何 + 磁偏角版本 + 转换顺序 ...)
- store:      SQLite 版本化存储, 原子指针切换, 导出导入身份保持
- engine:     依赖图传播重算
- cases:      离线领域案例包与可重放脚本
"""

__version__ = "1.0.0"
