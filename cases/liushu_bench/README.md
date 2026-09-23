# 流束换算台：ADCP 波束速度、姿态旋转与质量层级

离线领域案例包。水文团队把 ADCP 原始 beam velocity、姿态、bottom track
和换能器配置重算为 instrument、ship 与 earth 坐标流速的完整叙事。

## 文件
- `baseline.json` — 初始导入：波束测量、correlation、amplitude、坏 beam
  标志、heading/pitch/roll、bottom track、换能器配置分开记录；磁偏角与
  安装角作为带版本参数。12 ping × 6 cell × 4 beam。
- `change_local.json` — 一次只影响局部依赖的源数据改动（p4/b1/c3 单条
  波束速度改正）。
- `revision_mounting_angle.json` — 一次参数修订（安装角 0.5°→1.2°）。
- `conflict_decisions.json` — 同一基版上的两次互相冲突的人工决定
  （ping10/cell3 三波束多解：A 采纳四波束解，B 采纳三波束解）。
- `scenarios.json` — 七个边界情形的机器可读期望：单 beam 失效、姿态跨越
  360°、时间缺口、三波束多解、安装角修订、坐标变换顺序、无效深度 cell。

## 重放顺序
1. 导入 baseline → 派生 v1（核对分开记录）。
2. 应用 change_local → v2（仅 p4/c3 与 profile/p4 重算，其余 still-valid）。
3. 应用安装角修订 → v3（仅 ship/earth/profile 重算）。
4. 坐标变换顺序变体 → v4（分支，指纹不同，主指针不动）。
5. 决定 A（v3 基版）→ v5 主线；决定 B（同一 v3 基版）→ 冲突 v6 分支；
   解决保留 A，B 分支保留不覆盖。
