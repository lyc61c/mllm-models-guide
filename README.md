# MLLM Models Guide · 多模态大模型综述

中文多模态大语言模型学习材料，整理模型结构、训练方法、评测效果与架构图。原续篇资料核验日期：2026 年 10 月 2 日；InternVL2.5、InternVL3 补充核验日期：2026 年 10 月 6 日。

## 阅读与下载

- [在线阅读 HTML](https://lyc61c.github.io/mllm-models-guide/)：直接在浏览器中阅读，无需下载文件。
- [可编辑文章](docs/tutorials/mllm-expanded.md)：保留 Yue Shui 原文主体，记录少量技术校订，并在原文最后一个模型之后补充 27 个模型章节，含 InternVL2.5、InternVL3。
- [HTML 阅读版](docs/tutorials/mllm-expanded.html)：下载后使用浏览器打开；全部 58 张图片已内嵌，公式使用在线 MathJax。
- [完整压缩包](docs/tutorials/mllm-expanded.zip)：包含文章、配图、来源记录和审查结果。
- [阅读说明](docs/tutorials/READING-GUIDE.md)。

每个新增模型章节均介绍核心思想、模型结构、训练与效果。新增配图包括 17 张论文／官方图，以及 16 张依据公开资料生成、明确标为非官方示意图的架构重建图。闭源模型未披露的内部细节与公开信息分开说明。

## 数据工程与评估教程

- [在线学习：MLLM 数据工程与评估基准](https://lyc61c.github.io/mllm-models-guide/tutorials/mllm-data-evaluation.html)
- [可编辑 Markdown](docs/tutorials/mllm-data-evaluation.md) · [完整学习包](docs/tutorials/mllm-data-evaluation.zip) · [资料来源](docs/tutorials/mllm-data-evaluation.sources.json)

参考 Datawhale《动手学大模型》数据工程与评估章节的教学组织，面向 MLLM 独立编写。涵盖图像、文档、视频、音频和联合模态的数据获取、清洗、对齐、去重、评测污染控制、训练配比、基准选择、常见指标、评测协议与误差分析。附 5 张原创流程图、6 道自测题和 Python 标准库练习，无需 GPU 或模型权重；练习数据与预测均为人为构造，不对应真实模型成绩。资料核验日期：2026 年 10 月 6 日。

## 来源与校订

原文：[Yue Shui — 多模态大语言模型](https://syhya.github.io/zh/posts/2025-05-04-multimodal-llm/)。

- [原文源码](docs/tutorials/mllm-original.md)
- [原文 MIT 许可](docs/tutorials/SOURCE-LICENSE.txt)
- [技术校订记录](docs/tutorials/original-corrections.json)
- [论文／官方图片来源](docs/tutorials/figure-sources.json)
- [重建图依据与生成记录](docs/tutorials/reconstructed-figure-sources.json)
- [内容与配图审读](docs/tutorials/content-and-figure-review.md)
- [完整性检查](docs/tutorials/mllm-expanded.coverage.json)

原文的 MIT 许可按原作者声明保留。论文及官方图片的权利属于相应作者或机构；具体出处见图注与来源记录，不将原文许可扩展为第三方图片的统一许可。
