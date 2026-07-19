# Cairn

**AI 代理的记忆、协调与学习，全部在你的机器上运行。** 你的代理的大脑不应该存储在别人的服务器上。

[![PyPI version](https://img.shields.io/pypi/v/cairn.svg)](https://pypi.org/project/cairn/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![#1 on LongMemEval](https://img.shields.io/badge/LongMemEval-95.4%25_%231_Overall-gold.svg)](https://tracqi.com/benchmarks)

[🇺🇸 English](../README.md) | [🇨🇳 中文](README_zh-CN.md) | [🇯🇵 日本語](README_ja.md) | [🇰🇷 한국어](README_ko.md) | [🇧🇷 Português](README_pt-BR.md) | [🇪🇸 Español](README_es.md) | [🇫🇷 Français](README_fr.md) | [🇩🇪 Deutsch](README_de.md) | [🇷🇺 Русский](README_ru.md)

## 快速开始

```bash
pip3 install cairn[server]
cairn setup
```

支持 **Claude Code** | **Cursor** | **Windsurf** | **Zed** | 任何 MCP 客户端

---

## 为什么不用 CLAUDE.md？

Claude Code 内置的 `CLAUDE.md` 是一个纯文本文件。记几条笔记没问题，但它在以下场景中会崩溃：

- **无法搜索。** 超过 200 行后，你只能用 grep 碰运气。Cairn 使用语义搜索（bge-small-en-v1.5 嵌入 + sqlite-vec），即使用词不同也能找到相关记忆。
- **不会自动捕获。** 每条经验都要手动写入。Cairn 自动检测决策和调试结果。
- **无限膨胀。** 没有去重、没有衰减、没有矛盾检测。Cairn 自动解决冲突，语义去重，衰减过时记忆。
- **每个项目一个文件。** 无法跨项目学习。Cairn 的记忆图谱覆盖你的整个开发历史。
- **无法存档恢复。** 重构到一半停下来就没法接着做。Cairn 保存任务状态，下次从中断处精确恢复。

CLAUDE.md 适合记录「始终使用 tabs」这类规则。Cairn 适合让你的代理真正学会东西。

## 基准测试

**[LongMemEval](https://github.com/xiaowu0162/LongMemEval) 排名第一** (ICLR 2025) — 学术界长期记忆系统评测基准，500 道题目测试提取、推理、时间理解和偏好追踪。

| 系统 | 得分 | 备注 |
|------|-----:|------|
| **Cairn** | **95.4%** | **第一名** |
| Mastra | 94.87% | 第二名 |
| Zep/Graphiti | 71.2% | -- |

## 核心特性

- **12 个 MCP 工具** — 存储、查询、搜索、检查点、恢复等。
- **语义搜索** — bge-small-en-v1.5 + sqlite-vec，快速精确检索。
- **自动捕获与唤起** — Hooks 自动检测决策，在工作时唤起相关记忆。
- **检查点与恢复** — 中途停止任务，下次从中断处继续。
- **智能遗忘** — 时间衰减、冲突解决、语义去重。
- **完全本地，无需 API Key** — 所有数据和计算都在你的机器上。

## 安装

```bash
pip3 install cairn[server]   # 从 PyPI 安装（包含 MCP 服务器）
cairn setup                         # 自动配置编辑器
cairn doctor                        # 验证安装是否正常
```

使用 Cursor、Windsurf 或 Zed？

```bash
cairn setup --client cursor
cairn setup --client windsurf
cairn setup --client zed
```

## 对比

| 特性 | Cairn | CLAUDE.md | Mem0 |
|------|:-----:|:---------:|:----:|
| 跨会话持久化 | ✅ | ✅ | ✅ |
| 语义搜索 | ✅ | ❌ | ✅ |
| 自动捕获 | ✅ | ❌ | ✅ |
| 矛盾检测 | ✅ | ❌ | ❌ |
| 纯本地（无需 API Key） | ✅ | ✅ | ❌ |

---

完整文档请参阅 [English README](../README.md)。

网站：[tracqi.com](https://tracqi.com) | 文档：[tracqi.com/docs](https://tracqi.com/docs) | 基准测试：[tracqi.com/benchmarks](https://tracqi.com/benchmarks)

## 许可证

Apache-2.0
