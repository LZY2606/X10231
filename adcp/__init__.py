"""ADCP 流束换算台：beam velocity -> instrument -> ship -> earth。

核心规则层为纯函数、无 I/O；存储与重放分别位于 ``storage`` 与 ``replay``。
"""

__version__ = "1.0.0"
SCHEMA_VERSION = 2
