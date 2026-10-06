# MLLM expanded tutorial — independent review notes

Review status: preliminary content and figure review complete; final Markdown/HTML render review waits for the explicit final-file freeze.

## Figure provenance and integrity

- Inspected all 15 files listed in `docs/tutorials/reconstructed-figure-sources.json`.
- Every reconstructed PNG exists and its SHA-256 matches the recorded value.
- The generated figures consistently identify themselves as reconstructions rather than official paper figures.
- Closed-model internals are generally kept in gray dashed boxes and are not populated with invented layer or parameter counts.
- The GPT-6 Astra / GPT-6.1 Sol figure correctly keeps external tools outside the model, routes screenshot observations to visual input, and presents the two disclosed training elements in parallel rather than as a fabricated fixed stage sequence.

## Items to verify in the frozen tutorial

1. The Qwen3.6 figure labels tool feedback as “text or screenshots” and returns it to a combined `Text / reasoning history / tool results` box that feeds token embeddings. The Qwen3.8 figure similarly returns “Screenshots or results” to a text/history box. Their frozen captions or surrounding prose should state that screenshots remain visual observations and the return arrow is a system-level simplification; it must not imply screenshots are converted to text embeddings.
2. The Gemini 3.8 Flash figure labels a `Reasoning / Plan` box as “internal chain of thought” and maps effort to a reasoning-token budget. The frozen prose should retain the current limitation that this box is a behavioral abstraction, not a disclosed separate neural module or a claim that hidden reasoning is exposed.
3. The Gemini 3.1 Pro figure draws tool results both toward the model and toward multimodal context. The frozen caption should make clear that these are two views of the same feedback concept, not two independently disclosed internal routes.
4. The Kimi K3 QAT shorthand must be scoped in the frozen caption or prose to the disclosed quantized routed-expert weights/activations; shared experts and other components must not be implied to use the same low precision.
5. For every reconstructed figure in the frozen Markdown, verify that the caption says it is reconstructed/non-official and that any gray-box uncertainty described in the figure remains explicit in the text.

## Preliminary content assessment

- `closed_models.md` maintains a clear boundary between public interface/system behavior and undisclosed neural internals. It repeatedly avoids inferring encoders, expert counts, loss recipes, freeze schedules, or reward details from benchmark behavior.
- `omni_models.md` is technically detailed enough to explain structures, training stages, and evaluation conditions rather than listing product features. It distinguishes HTTP text/tool services from realtime speech branches and distinguishes paper-original figures from reconstructions.
- No preliminary blocking factual issue was found. Final verdict remains pending because the canonical Markdown and generated HTML have not yet been frozen and hash-checked.


---

# InternVL2.5 / InternVL3 独立核对记录

- **审查日期**：2026-10-06
- **最终 Markdown SHA-256**：`fd3084ac1ed691819b6a47ac0407b1a420785b78d89797d5beac9e4156dac6d7`
- **最终 HTML SHA-256**：`f5b6fb52e7f9eaf506f5140e0e29c1175acedf7d6012ca120a9ea8f2ec5f55ea`
- **审查者**：fresh cross-model collaboration reviewer
- **MCP 状态**：技能指定的 MCP 端点不可用，本次未调用，也不声称取得 MCP 结果；以下结论来自独立 reviewer 对冻结文件和一手资料的核对。
- **范围**：重点审查新增的 InternVL2.5 与 InternVL3 两章、三张图、公式、表格、评测条件和 HTML 忠实度。更早章节的事实审查继承 2026-10-02 的独立 PASS；本次只复核其保留状态与整篇静态完整性。

## 结论

**PASS。无阻塞问题，无剩余 warning。**

此前唯一问题是 InternVL3 教学重建图把图像与视频共同画入 `Dynamic 448×448 tiles`，可能让读者误以为每个视频帧都采用多 tile。最终图注已加入“图中 Dynamic 448×448 tiles 概括视觉预处理，不表示每个视频帧都采用多 tile。”该句把教学概括与论文明确披露的事实边界分开，歧义已消除。

## 一手资料

1. [InternVL 2.5 论文](https://arxiv.org/html/2412.05271v1)
2. [InternVL3 论文](https://arxiv.org/html/2504.10479v3)
3. [InternVL 2.5 官方博客](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)
4. [InternVL 3.0 官方博客](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/)

## InternVL2.5 事实核对

| 核对项 | 论文事实与文章判断 |
|---|---|
| 主结构 | ViT–MLP–LLM；视觉编码器为 InternViT-300M 或 InternViT-6B；连接器为随机初始化的两层 MLP。文章一致。 |
| 图像 token | 每个 `448×448` tile 经 ViT 得 1024 个视觉 token，再经 pixel unshuffle 压到 256 个。文章一致。 |
| 视频输入 | 报告描述的视频帧不做图像式切块；正文已明确这一条件。 |
| 训练冻结 | Stage 1 只训练 MLP，ViT 与 LLM 冻结；8B、26B 的可选 Stage 1.5 训练 ViT 与 MLP、冻结 LLM；Stage 2 全模型指令微调。文章一致。 |
| 平方平均 | 教学公式正确表达先聚合后平方与按样本平方再平均的区别，没有把教学推导写成论文原式。 |
| MMMU test-time scaling | InternVL2.5-78B 在 MMMU validation 上 direct 为 66.4，CoT 为 70.1，提升 3.7。文章一致。 |
| Majority vote 归属 | 62.7→65.3 是 InternVL2-Llama3-76B 的例子，不是 InternVL2.5-78B。文章归属正确。 |

两张论文图的编号和用途与报告一致：Figure 2 是总体架构，Figure 4 是三阶段训练和渐进扩展示意。

## InternVL3 事实核对

| 核对项 | 论文事实与文章判断 |
|---|---|
| 初始化 | 从 pretrained base LLM 和 pretrained ViT 出发，另配随机初始化的两层 MLP，并非从 instruction-tuned LLM 起步。文章一致。 |
| 原生多模态预训练 | 文本与多模态数据比例 `1:3`，分别约 50B 与 150B token，共 200B；loss 只计算在文本 token 上，但所有层联合更新。正文正确区分“不对视觉 token 计 loss”和“冻结视觉模块”。 |
| V2PE | 文本位置步长为 1；同一图像内部视觉 token 使用固定步长 `δ`；训练时每图从 `{1, 1/2, ..., 1/256}` 随机采样，推理时可灵活选择。除 Table 12 专门消融外，报告结果固定 `δ=1`。文章限定准确。 |
| MPO | SFT 后进行 MPO；目标为 DPO、BCO 与生成损失的加权和；约 300K preference samples，rollout 来自 SFT 8B/38B/78B。文章一致。14B、38B、78B 增益 `+3.7`、`+4.5`、`+4.1` 也与消融表一致。 |
| Test-time scaling | Best-of-N 由外部 VisualPRM-8B 打分；`VisualPRM-Bo8` 是 Best-of-8。文章没有把该 critic 写成主模型内部组件。 |
| Bo8 数值 | 78B：MMMU `72.2→72.2`，七任务均值 `54.6→56.5`，MathVerse `51.0→54.2`，MathVision `43.1→40.8`。数值和条件准确，也保留了并非每项都提升的事实。 |

新增的 V2PE 位置递推式与 MPO 加权目标式在 Markdown 和 HTML 中术语、下标和条件一致。

## 图片与来源核对

| 文件 | SHA-256 | 判断 |
|---|---|---|
| `assets/new/internvl25-architecture.png` | `883b8c92fa65b4862d4ca8ff03c9c1345af01852394a7b1080ef2adf5dd02ccc` | InternVL2.5 Figure 2 的完整可读 crop；页码、尺寸和哈希与 `figure-sources.json` 一致。 |
| `assets/new/internvl25-training.png` | `ad77c8842376fe49120127ca7e871bcc8f1d8b08269e6b22959098802398c6d0` | InternVL2.5 Figure 4 的完整可读 crop；冻结/训练状态和渐进扩展均保留，manifest 一致。 |
| `assets/reconstructed/internvl3-architecture.png` | `5d24967fea5d9f2feeba78650052a0187ee91c982c8a9984c3d3aa60bd0f4131` | 教学重建图，图内与 `reconstructed-figure-sources.json` 均明确标注非官方；结构、V2PE、SFT/MPO/VisualPRM 关系与报告一致，最终图注已消除视频切块歧义。 |

三张图在 HTML 中的 data URI 解码结果与本地文件逐字节一致，没有缺失来源记录或 manifest 哈希不一致。

## HTML 静态忠实度与限制

- 全文：52 个 heading、29 张表、285 个数学表达式、3 个 fenced code block、58 张图、2 个 details block。
- 新增审查范围：1 个 H2、2 个 H3、4 张表、3 个数学表达式、3 张图。
- `source-sha256` 元数据和页面可见来源哈希均等于最终 Markdown SHA-256。
- 553 个原文非空行仍按顺序保留；10 项既有修订各出现一次。
- 未发现占位符、替换字符、源文注入事件处理器、`javascript:`/`vbscript:` URL 或异常 active/form 元素。
- 脚本仅包括 renderer 生成的 MathJax 配置和 jsDelivr 上的 MathJax 3.2.2 入口。

本次是静态源文件与 HTML 解析审查。由于本地 `file://` 预览被浏览器协议策略拒绝，没有绕过策略，因此本记录不声称本次构建中 MathJax 已在浏览器实际执行，也不声称完成了交互式浏览器布局检查。早期 282 个公式的 2026-10-02 浏览器记录仅作为历史记录，不替代本次 285 个公式的静态一致性检查。评测数据已对照官方报告表格，但没有重新运行第三方 benchmark。
