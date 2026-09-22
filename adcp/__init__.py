"""ADCP 坐标变换与流速证据工具。

分层结构：
  units       原始量/单位/坐标基准，未转换值保留诊断
  fingerprint 稳定身份与派生指纹（内容寻址）
  geometry    波束几何（Janus 配置，进入派生指纹）
  transforms  姿态旋转与逐级坐标变换
  fixture     仓库内离线领域案例包（非通用表格）
  storage     SQLite 存储模型：源数据/派生/事件/分支/指针，原子版本
  derivation  依赖图传播、质量层级、三/四波束候选
  replay      边界情形脚本化重放（机器可读事件）
  service     存储之上的应用服务（导入、裁定、导出导入、as-of 还原）
  web         FastAPI + SVG 复核界面
"""
__version__ = "1.0.0"
SCHEMA_VERSION = 1
