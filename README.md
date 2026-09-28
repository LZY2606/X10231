# 流束换算台（ADCP Beam → Instrument → Ship → Earth 证据工具）

把 ADCP 原始 **beam velocity**、姿态（heading/pitch/roll）、**bottom track** 与换能器配置
重算为 **instrument / ship / earth(ENU)** 三级坐标流速；所有派生结果带稳定身份、单位、坐标基准、
来源链与候选解，并保存被采纳与未被采纳的全部结果。

## 领域语义（硬约束，自动化测试锁定）

- **三波束解 ≠ 四波束解**：单 beam 失效时保留全部合法三波束候选，不与四波束最小二乘解合并为同一值。
- **姿态插值不跨缺口**：时间缺口（gap）两侧不做插值，相关 ping 的 earth 级结果显式标记不可得。
- **派生指纹**：坐标变换顺序、波束几何（beam angles / 方位 / 深度单元几何）、磁偏角版本全部进入指纹；
  任何一项修订都会使下游 instrument/ship/earth 节点失效并通过依赖图传播。
- **质量层级**：坏 beam 标志、correlation、amplitude 分级判定；无效/被排除 depth cell 保留在原位并附诊断，
  不为画图丢弃。
- **多解不收敛**：无法唯一决定时保留全部合法候选，等待人工裁定；确认生成新事件，旧候选不被覆盖。
- **单位/基准**：原始层保留导入表示与原始单位，派生层才规范化；无法转换的值留在原位置并附 `UNCONVERTIBLE` 诊断。
- **版本化**：每次推导是独立 commit，只有完整成功才切换 current 指针；写入中断后重启只见旧版或完整新版。
- **并发**：两位使用者从同一版本编辑产生冲突事件，必须经解决动作才能推进，且保留两个分支。

## 安装

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 完整演示

```
.venv/bin/python -m pytest -q; .venv/bin/uvicorn app:app --host 127.0.0.1 --port 5347
```

然后打开 <http://127.0.0.1:5347>，页面内置 90 秒短样例路径（右上角“演示引导”）。

## 离线案例包

`cases/`（仓库自带，非通用表格，针对“流束换算台：ADCP 波束速度、姿态旋转与质量层级”）：
初始版本把 8 类来源分开记录，随后安排一次参数修订（磁偏角 v1→v2）、一次只影响局部依赖的源数据改动
（仅一个 ping 的 beam 修复）和一次互相冲突的人工决定（C2 vs C3 分支裁定）。反例档案实际经过：
单 beam 失效、姿态跨越 360°、时间缺口、三波束多解、安装角修订、坐标变换顺序、无效 depth cell。
可用 `python -m scripts.build_cases` 重新生成（确定性输出）。

## 布局

| 路径 | 职责 |
| --- | --- |
| `adcp/ids.py` | 稳定身份（内容哈希） |
| `adcp/units.py` | 带单位/坐标基准的量；导入表示保留；规范化与 UNCONVERTIBLE 诊断 |
| `adcp/geometry.py` | 换能器波束几何；beam→instrument 最小二乘；四波束/三波束解 |
| `adcp/transforms.py` | 姿态/安装角/磁偏角三级旋转；变换顺序版本化并进入指纹 |
| `adcp/attitude.py` | 角度规范化（359→1°）、不跨缺口的分段插值 |
| `adcp/quality.py` | correlation/amplitude/坏标志质量层级；cell 排除 |
| `adcp/events.py` | 事件溯源：8 类来源事件、修订、裁定；版本化视图 |
| `adcp/graph.py` | 依赖图传播：只重算受影响节点，说明 still-valid/stale |
| `adcp/replay.py` | 重算引擎：逐级向量、候选、指纹、来源链 |
| `adcp/store.py` | SQLite 原子提交、分支冲突、导出/导入、旧版本迁移 |
| `adcp/fixture.py` | 确定性领域案例包构建器 |
| `adcp/archive.py` | 反例档案：单来源改动 + 机器可读事件 + 采纳/未采纳结果 |
| `app.py` | FastAPI；SVG 复核界面 API |
