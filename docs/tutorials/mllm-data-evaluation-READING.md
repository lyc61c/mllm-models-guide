# 数据工程与评估教程：阅读与复现

在线阅读：https://lyc61c.github.io/mllm-models-guide/tutorials/mllm-data-evaluation.html

先读第一部分的数据生命周期，再读第二部分的评测协议与指标，最后运行第三部分的练习。正文参考 Datawhale 两章的教学组织，具体多模态内容独立撰写，技术来源见正文链接与 `mllm-data-evaluation.sources.json`。

## 文件

- `mllm-data-evaluation.md`：可编辑正文，是源文件。
- `mllm-data-evaluation.html`：单文件阅读版，内嵌 5 张原创 SVG；公式使用在线 MathJax 3.2.2，离线保留 TeX。
- `assets/data-evaluation/*.svg`：可编辑图示，不是论文模型架构图。
- `mllm-data-evaluation-lab/`：Python 3.10+ 标准库练习；在该目录执行 `python lab.py --out lab-output` 和 `python -m unittest -v test_lab.py`。
- `.sources.json`、`.render.json`、`.checks.json`、`.html.review.json`：来源、生成、静态完整性和独立审阅记录。
- `tools/render_mllm_learning.py`、`tools/check_mllm_learning.py`：仓库中的渲染与检查工具；渲染需要 `markdown-it-py`，学习练习本身不需要第三方依赖。

学习包只包含本教程，不包含另外的模型综述。页面指向模型综述的入口在学习包中需要配合现有仓库或在线网页使用。

## 练习边界

所有媒体哈希由模拟标识字符串产生，没有真实图片；A/B 预测预先编写，没有模型推理。数据划分与评分是两套独立练习样例。教学版归一化、ANLS 空串边界和共识核心不能代替官方评测器。bootstrap 估计的是等父对象权重的配对分差，不能冒充官方题目加权总分。

本文不发布真实模型成绩或当前 SOTA 排名。讨论数据和基准时区分版本与输入条件，实际训练/评测请固定对应项目的官方版本。
