"""ADCP 坐标变换与流速证据工具。

把 ADCP 原始 beam velocity、姿态、bottom track 和换能器配置重算为
instrument / ship / earth 坐标流速，并保留完整证据链。
"""

RULESET = "flowbench"
