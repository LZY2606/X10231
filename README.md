# 流束换算台：ADCP 波束速度、姿态旋转与质量层级

这是一个可离线重放的 ADCP 证据工具。原始 beam velocity、correlation、amplitude、坏 beam 标志、heading/pitch/roll、磁偏角、安装角、bottom track 与换能器几何分别入库；派生层以依赖图生成 instrument、ship、earth 坐标流速。

## 安装

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 完整演示

```bash
.venv/bin/python -m pytest -q; .venv/bin/uvicorn app:app --host 127.0.0.1 --port 5347
```

然后打开：

```text
http://127.0.0.1:5347
```

页面点击“运行内置重放脚本”，可在九十秒内依次看到来源分离、单 beam 失效、姿态跨 360°、时间缺口、三波束多解、安装角修订、坐标变换顺序、无效深度 cell、分支比较、冲突解决与历史读回。

## 语义红线

- 三波束结果和四波束结果使用不同 stage/result 身份，不合并成一个值。
- 姿态只在连续时间片段内插值；heading 先按最短角跨 360° 解绕，缺口两侧不插值。
- 坐标转换顺序、换能器几何版本和磁偏角版本都参与派生指纹。
- 无法唯一决定时保留全部合法候选；未采纳候选不会被删除。
- 原始量保留数值、单位、坐标基准和原始表示；不能转换的值留在原位置并带诊断。
- 派生版本完整写入后才切换 current；中断重启只能看到旧版本或完整新版本。
