# Cairn

**AIエージェントの記憶・連携・学習を、すべてあなたのマシンで。** エージェントの頭脳を他人のサーバーに預ける必要はありません。

[![PyPI version](https://img.shields.io/pypi/v/cairn.svg)](https://pypi.org/project/cairn/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![#1 on LongMemEval](https://img.shields.io/badge/LongMemEval-95.4%25_%231_Overall-gold.svg)](https://tracqi.com/benchmarks)

[🇺🇸 English](../README.md) | [🇨🇳 中文](README_zh-CN.md) | [🇯🇵 日本語](README_ja.md) | [🇰🇷 한국어](README_ko.md) | [🇧🇷 Português](README_pt-BR.md) | [🇪🇸 Español](README_es.md) | [🇫🇷 Français](README_fr.md) | [🇩🇪 Deutsch](README_de.md) | [🇷🇺 Русский](README_ru.md)

## クイックスタート

```bash
pip3 install cairn[server]
cairn setup
```

対応エディタ: **Claude Code** | **Cursor** | **Windsurf** | **Zed** | 任意の MCP クライアント

---

## なぜ CLAUDE.md ではダメなのか？

Claude Code 組み込みの `CLAUDE.md` はただのテキストファイルです。メモ程度なら十分ですが、以下の場面で破綻します：

- **検索できない。** 200行を超えると grep 頼みになる。Cairn はセマンティック検索（bge-small-en-v1.5 埋め込み + sqlite-vec）を使い、表現が違っても関連する記憶を見つけます。
- **自動キャプチャがない。** 学んだことを毎回手動で書く必要がある。Cairn は意思決定やデバッグ結果を自動検出します。
- **際限なく肥大化する。** 重複排除もなく、劣化もなく、矛盾検出もない。Cairn は自動的に競合を解決し、意味的に類似したエントリを統合し、古い記憶を減衰させます。
- **プロジェクトごとに1ファイル。** プロジェクト間の学習ができない。Cairn のメモリグラフは開発履歴全体をカバーします。
- **チェックポイントができない。** リファクタリングの途中で止めたら再開できない。Cairn はタスク状態を保存し、次回正確に中断箇所から再開します。

CLAUDE.md は「常にタブを使う」のような規則の記録には十分です。Cairn はエージェントに本当に学習させたい場合のためのツールです。

## ベンチマーク

**[LongMemEval](https://github.com/xiaowu0162/LongMemEval) 第1位** (ICLR 2025) — 長期記憶システムの学術ベンチマーク。抽出、推論、時間理解、嗜好追跡を評価する500問。

| システム | スコア | 備考 |
|---------|------:|------|
| **Cairn** | **95.4%** | **1位** |
| Mastra | 94.87% | 2位 |
| Zep/Graphiti | 71.2% | -- |

## 主な機能

- **12個の MCP ツール** — 保存、クエリ、検索、チェックポイント、再開など。
- **セマンティック検索** — bge-small-en-v1.5 + sqlite-vec による高速・高精度な検索。
- **自動キャプチャ＆サーフェシング** — Hooks が意思決定を自動検出し、作業中に関連記憶を提示。
- **チェックポイント＆再開** — タスクを中断しても、次のセッションで続行可能。
- **インテリジェントな忘却** — 時間減衰、競合解決、重複排除。
- **完全ローカル、APIキー不要** — すべてのデータと処理はローカルマシン上。

## インストール

```bash
pip3 install cairn[server]   # PyPI からインストール（MCP サーバー含む）
cairn setup                         # エディタを自動設定
cairn doctor                        # インストールの正常性を確認
```

Cursor、Windsurf、Zed をお使いですか？

```bash
cairn setup --client cursor
cairn setup --client windsurf
cairn setup --client zed
```

## 比較

| 機能 | Cairn | CLAUDE.md | Mem0 |
|------|:-----:|:---------:|:----:|
| セッション間の永続化 | ✅ | ✅ | ✅ |
| セマンティック検索 | ✅ | ❌ | ✅ |
| 自動キャプチャ | ✅ | ❌ | ✅ |
| 矛盾検出 | ✅ | ❌ | ❌ |
| ローカル完結（APIキー不要） | ✅ | ✅ | ❌ |

---

完全なドキュメントは [English README](../README.md) をご覧ください。

ウェブサイト：[tracqi.com](https://tracqi.com) | ドキュメント：[tracqi.com/docs](https://tracqi.com/docs) | ベンチマーク：[tracqi.com/benchmarks](https://tracqi.com/benchmarks)

## ライセンス

Apache-2.0
