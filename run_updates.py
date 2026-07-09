import os
import subprocess

repo_dir = "C:/Users/ishan/Documents/Projects/Awesome-Gated-Linear-Units"
os.chdir(repo_dir)

def run_git(msg):
    print(f"Running git commit: {msg}")
    subprocess.run(["git", "-C", repo_dir, "add", "."], check=True)
    subprocess.run(["git", "-C", repo_dir, "commit", "-m", msg], check=True)
    subprocess.run(["git", "-C", repo_dir, "push"], check=True)

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

table_1 = """
| Era / Concept | Description | Year | Paper | Details |
|---|---|---|---|---|
| The Sequential Recurrent Gating Era (LSTM / GRU) | The historical baseline of gated neural networks. Recurrent cells like LSTM introduced explicit input, forget, and output gates to create a linear memory highway. | 1997 | [Hochreiter & Schmidhuber, 1997](https://doi.org/10.1162/neco.1997.9.8.1735) | [Read more](LSTM-GRU.md) |
| The Flat Convolutional Gating Revolution (Vanilla GLU) | Ported gating mechanics out of recurrent dependencies and straight into parallelizable convolutional networks. Dauphin et al. split a layer's projection matrix into two parallel paths. | 2017 | [Dauphin et al., 2017](https://arxiv.org/abs/1612.08083) | [Read more](Vanilla-GLU.md) |
| The Smooth Non-Linear Projection Era (SwiGLU / GeGLU) | Perfected gating mechanics for Transformer architectures by replacing the flat Sigmoid operator with state-of-the-art smooth, non-linear activation functions. | 2020 | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) | [Read more](SwiGLU-GeGLU.md) |
| The Fused Latent Parallel MoE Era | The current modern state-of-the-art foundation infrastructure standard. It ports SwiGLU gating out of dense multi-layer networks and straight into sparsely routed architectures. | 2024 | [DeepSeek-AI, 2024](https://arxiv.org/abs/2401.06066) | [Read more](Fused-Latent-Parallel-MoE.md) |
"""

table_2 = """
| Variant | Mechanism / Pros | Year | Paper | Details |
|---|---|---|---|---|
| A. Vanilla GLU (Sigmoid-Gated Linear) | Multiplies a linear projection by the Sigmoid transformation of a parallel projection. The Sigmoid function acts as a soft binary switch. | 2017 | [Dauphin et al., 2017](https://arxiv.org/abs/1612.08083) | [Read more](Vanilla-GLU-Variant.md) |
| B. SwiGLU (Swish/SiLU Gated Linear) | Replaces the Sigmoid operator with the Swish activation function, omitting the bias terms to streamline hardware execution. | 2020 | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) | [Read more](SwiGLU-Variant.md) |
| C. GeGLU (GELU Gated Linear) | Leverages the Gaussian Error Linear Unit (GELU) to modulate the gating pathway. | 2020 | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) | [Read more](GeGLU-Variant.md) |
| D. ReGLU (ReLU Gated Linear) | Uses a simple rectified linear unit (ReLU) to shape the gating channel. Computationally cheaper than SwiGLU or GeGLU. | 2020 | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) | [Read more](ReGLU-Variant.md) |
"""

table_3 = """
| Component | Profile / Description | Year | Paper | Details |
|---|---|---|---|---|
| Fused Input Projection Kernels | Collapses model memory lookups. To prevent launching two separate, sequential GPU kernel executions for the Gate path and the Up path. | 2022 | [Dao et al., 2022 (FlashAttention)](https://arxiv.org/abs/2205.14135) | [Read more](Fused-Input-Projection-Kernels.md) |
| The $3d_{\\text{ffn}}$ Column Scaling Matrix | Manages parameter balancing. To maintain a fair compute comparison against dense baselines, intermediate dimension is downscaled. | 2020 | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) | [Read more](3d-ffn-Column-Scaling-Matrix.md) |
"""

table_4 = """
| Challenge | Problem & Mitigation | Year | Paper | Details |
|---|---|---|---|---|
| The FlashAttention Kernel Memory Allocation Gap | Fusing the input projections creates massive activation tensor fields... Mitigated by deploying custom handwritten Triton or CUDA kernels. | 2022 | [Dao et al., 2022](https://arxiv.org/abs/2205.14135) | [Read more](FlashAttention-Kernel-Memory-Allocation-Gap.md) |
| The Parameter-Heterogeneity Expert Load-Imbalance Wall | When sharding a model's parallel GLU layers, column-parallel allocations must balance processing steps perfectly. Mitigated by Device-Aware Topology Routing. | 2021 | [Fedus et al., 2021 (Switch Transformers)](https://arxiv.org/abs/2101.03961) | [Read more](Parameter-Heterogeneity-Expert-Load-Imbalance.md) |
"""

table_5 = """
| Application | Description | Year | Paper | Details |
|---|---|---|---|---|
| Pre-Training Web-Scale Foundational LLM Suites (Llama / DeepSeek) | Serves as the standard default FFN activation backbone used to train elite base architectures. | 2023 | [Touvron et al., 2023 (Llama)](https://arxiv.org/abs/2302.13971) | [Read more](Pre-Training-Web-Scale-Foundational-LLMs.md) |
| Low-Latency Real-Time Cloud Inference Serving Engines (vLLM Deployments) | Compresses model generation response latency within enterprise software endpoints. | 2023 | [Kwon et al., 2023 (vLLM)](https://arxiv.org/abs/2309.06180) | [Read more](Low-Latency-Real-Time-Cloud-Inference.md) |
| Autonomous Vehicle Multimodal Bird's-Eye-View Perception | Ingests real-time streaming high-res camera video and LiDAR 3D coordinates concurrently. | 2022 | [Li et al., 2022 (BEVFormer)](https://arxiv.org/abs/2203.17270) | [Read more](Autonomous-Vehicle-Multimodal-BEV.md) |
"""

def replace_between(text, start_marker, end_marker, replacement):
    start = text.find(start_marker)
    if start == -1: return text
    end = text.find(end_marker, start)
    if end == -1: return text
    return text[:start] + replacement + "\n\n" + text[end:]

readme = replace_between(readme, '*   **The Sequential Recurrent', '\n---', table_1.strip())
readme = replace_between(readme, '- ### A. Vanilla', '\n---', table_2.strip())
readme = replace_between(readme, '*   **Fused Input', '\n---', table_3.strip())
readme = replace_between(readme, '*   **The FlashAttention', '\n---', table_4.strip())
readme = replace_between(readme, '*   **Pre-Training Web-Scale', '\n---', table_5.strip())

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
run_git("tabularised the bullets")


details = {
    "LSTM-GRU.md": "# LSTM / GRU\n\nMore detailed information about LSTM and GRU.\n\n```mermaid\nflowchart TD\n A[Input] --> B[Forget Gate]\n```\n\n[Back to README](README.md)",
    "Vanilla-GLU.md": "# Vanilla GLU\n\nMore detailed information about Vanilla GLU.\n\n```mermaid\nflowchart TD\n A[Input] --> B[Sigmoid Gate]\n```\n\n[Back to README](README.md)",
    "SwiGLU-GeGLU.md": "# SwiGLU / GeGLU\n\nMore detailed information about SwiGLU and GeGLU.\n\n```mermaid\nflowchart TD\n A[Input] --> B[Swish Gate]\n```\n\n[Back to README](README.md)",
    "Fused-Latent-Parallel-MoE.md": "# Fused Latent Parallel MoE\n\nMore detailed information about MoE architectures.\n\n```mermaid\nflowchart TD\n A[Input] --> B[Expert 1]\n A --> C[Expert 2]\n```\n\n[Back to README](README.md)",
    "Vanilla-GLU-Variant.md": "# Vanilla GLU Variant\n\nDetailed mechanics.\n\n```mermaid\nflowchart TD\n A[Linear] --> B[Multiply]\n C[Sigmoid] --> B\n```\n\n[Back to README](README.md)",
    "SwiGLU-Variant.md": "# SwiGLU Variant\n\nDetailed mechanics.\n\n```mermaid\nflowchart TD\n A[Linear] --> B[Multiply]\n C[Swish] --> B\n```\n\n[Back to README](README.md)",
    "GeGLU-Variant.md": "# GeGLU Variant\n\nDetailed mechanics.\n\n```mermaid\nflowchart TD\n A[Linear] --> B[Multiply]\n C[GELU] --> B\n```\n\n[Back to README](README.md)",
    "ReGLU-Variant.md": "# ReGLU Variant\n\nDetailed mechanics.\n\n```mermaid\nflowchart TD\n A[Linear] --> B[Multiply]\n C[ReLU] --> B\n```\n\n[Back to README](README.md)",
    "Fused-Input-Projection-Kernels.md": "# Fused Input Projection Kernels\n\nMemory allocation.\n\n```mermaid\nflowchart TD\n A[Input] --> B[Fused Kernel]\n```\n\n[Back to README](README.md)",
    "3d-ffn-Column-Scaling-Matrix.md": "# 3d FFN Scaling\n\nMatrix dimension balancing.\n\n```mermaid\nflowchart TD\n A[3d] --> B[Scaling]\n```\n\n[Back to README](README.md)",
    "FlashAttention-Kernel-Memory-Allocation-Gap.md": "# FlashAttention Kernel Memory Allocation Gap\n\nDetails on mitigation.\n\n```mermaid\nflowchart TD\n A[OOM] --> B[Triton Kernel]\n```\n\n[Back to README](README.md)",
    "Parameter-Heterogeneity-Expert-Load-Imbalance.md": "# Parameter Heterogeneity\n\nDetails on imbalance.\n\n```mermaid\nflowchart TD\n A[Imbalance] --> B[Topology Routing]\n```\n\n[Back to README](README.md)",
    "Pre-Training-Web-Scale-Foundational-LLMs.md": "# Llama / DeepSeek\n\nLLM pre-training details.\n\n```mermaid\nflowchart TD\n A[Dataset] --> B[LLM]\n```\n\n[Back to README](README.md)",
    "Low-Latency-Real-Time-Cloud-Inference.md": "# vLLM Deployments\n\nvLLM deployment info.\n\n```mermaid\nflowchart TD\n A[Query] --> B[vLLM Serving]\n```\n\n[Back to README](README.md)",
    "Autonomous-Vehicle-Multimodal-BEV.md": "# BEV Perception\n\nAutonomous vehicles info.\n\n```mermaid\nflowchart TD\n A[Camera] --> B[BEV Grid]\n```\n\n[Back to README](README.md)"
}

for name, content in details.items():
    with open(name, "w", encoding="utf-8") as f:
        f.write(content)
run_git("detailed pages created")


os.makedirs("assets", exist_ok=True)
svg_banner = '''<svg width="800" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#8A2BE2;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#4B0082;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#grad1)" rx="15" />
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="Arial, sans-serif" font-weight="bold" font-size="42" fill="white">Awesome Gated Linear Units</text>
  <circle cx="100" cy="100" r="20" fill="white" opacity="0.5">
    <animate attributeName="r" values="20;30;20" dur="2s" repeatCount="indefinite" />
    <animate attributeName="opacity" values="0.5;0;0.5" dur="2s" repeatCount="indefinite" />
  </circle>
  <circle cx="700" cy="100" r="20" fill="white" opacity="0.5">
    <animate attributeName="r" values="20;30;20" dur="2s" repeatCount="indefinite" />
    <animate attributeName="opacity" values="0.5;0;0.5" dur="2s" repeatCount="indefinite" />
  </circle>
</svg>'''
with open("assets/banner.svg", "w", encoding="utf-8") as f:
    f.write(svg_banner)

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

readme = readme.replace("# Awesome-Gated-Linear-Units", "![Banner](assets/banner.svg)\n\n# 🚀 Awesome-Gated-Linear-Units 🌟")
readme = readme.replace("## Gated Linear Units (GLUs)", "## 🧠 Gated Linear Units (GLUs)")
readme = readme.replace("## 1. The Macro Chronological Evolution", "## 🕰️ 1. The Macro Chronological Evolution")
readme = readme.replace("## 2. Core Functional & Algorithmic Variants", "## 🧬 2. Core Functional & Algorithmic Variants")
readme = readme.replace("## 3. The SwiGLU FFN Layer Matrix", "## 🧮 3. The SwiGLU FFN Layer Matrix")
readme = readme.replace("## 4. Production Engineering Challenges", "## ⚙️ 4. Production Engineering Challenges")
readme = readme.replace("## 5. Frontier Real-World AI Industrial Applications", "## 🏭 5. Frontier Real-World AI Industrial Applications")
readme = readme.replace("## References", "## 📚 References")
readme = readme.replace("**Follow-Up Options Matrix:**", "🛠️ **Follow-Up Options Matrix:**")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
run_git("added emojis and banner")


left_badges = '<a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a><a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>'

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

readme = readme.replace("# 🚀 Awesome-Gated-Linear-Units 🌟\n", f"# 🚀 Awesome-Gated-Linear-Units 🌟\n\n<div align=\"center\">\n{left_badges}\n</div>\n\n")
readme = readme.replace("## 🧠 Gated Linear Units (GLUs)", "## 🧠 Gated Linear Units (GLUs) - Advanced Neural Network Architectures")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
run_git("seo optimised and badges to left added")


right_badge = ' <a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>'

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

readme = readme.replace(f"{left_badges}\n</div>", f"{left_badges}{right_badge}\n</div>")
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
run_git("badges to right added")


folder_name = "Awesome-Gated-Linear-Units"
star_history = f'''
##  Star History
<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007%2F{folder_name}&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/{folder_name}&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/{folder_name}&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ishandutta2007/{folder_name}&type=date&legend=bottom-right" />
</picture>
</a>
</div>
'''

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()
readme += "\n" + star_history
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
run_git("star history added")


with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()
readme = readme.replace("chartrepos", "chart?repos")
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
run_git("fixed star plot")


with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()
readme = readme.replace("https://github.com/sindresorhus/awesome", "https://github.com/ishandutta2007/Awesome-Awesome-Awesome")
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
run_git("invalid awesome link fixed")

print("All done!")
