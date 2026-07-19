# Cairn

**AI 에이전트의 기억, 협업, 학습을 당신의 머신에서.** 에이전트의 두뇌가 다른 사람의 서버에 있을 필요는 없습니다.

[![PyPI version](https://img.shields.io/pypi/v/cairn.svg)](https://pypi.org/project/cairn/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![#1 on LongMemEval](https://img.shields.io/badge/LongMemEval-95.4%25_%231_Overall-gold.svg)](https://tracqi.com/benchmarks)

[🇺🇸 English](../README.md) | [🇨🇳 中文](README_zh-CN.md) | [🇯🇵 日本語](README_ja.md) | [🇰🇷 한국어](README_ko.md) | [🇧🇷 Português](README_pt-BR.md) | [🇪🇸 Español](README_es.md) | [🇫🇷 Français](README_fr.md) | [🇩🇪 Deutsch](README_de.md) | [🇷🇺 Русский](README_ru.md)

## 빠른 시작

```bash
pip3 install cairn[server]
cairn setup
```

지원: **Claude Code** | **Cursor** | **Windsurf** | **Zed** | 모든 MCP 클라이언트

---

## 왜 CLAUDE.md로는 부족한가?

Claude Code 내장 `CLAUDE.md`는 단순한 텍스트 파일입니다. 간단한 메모 정도는 괜찮지만, 다음과 같은 상황에서 한계가 드러납니다:

- **검색이 안 된다.** 200줄이 넘으면 grep에 의존해야 한다. CAIRN는 시맨틱 검색(bge-small-en-v1.5 임베딩 + sqlite-vec)을 사용해 표현이 달라도 관련 기억을 찾아냅니다.
- **자동 캡처가 없다.** 배운 것을 매번 직접 작성해야 한다. CAIRN는 결정과 디버깅 결과를 자동으로 감지합니다.
- **끝없이 비대해진다.** 중복 제거, 감쇠, 모순 감지가 없다. CAIRN는 충돌을 자동 해결하고, 의미적으로 유사한 항목을 통합하며, 오래된 기억을 감쇠시킵니다.
- **프로젝트당 하나의 파일.** 프로젝트 간 학습이 불가능하다. CAIRN의 메모리 그래프는 전체 개발 이력을 아우릅니다.
- **체크포인트가 안 된다.** 리팩토링 중간에 멈추면 다시 시작할 방법이 없다. CAIRN는 작업 상태를 저장하고, 다음에 정확히 중단된 지점에서 재개합니다.

CLAUDE.md는 "항상 탭을 사용할 것" 같은 규칙 기록에는 충분합니다. CAIRN는 에이전트가 진짜 학습해야 할 때를 위한 도구입니다.

## 벤치마크

**[LongMemEval](https://github.com/xiaowu0162/LongMemEval) 1위** (ICLR 2025) — 장기 메모리 시스템 학술 벤치마크. 추출, 추론, 시간 이해, 선호도 추적을 평가하는 500문항.

| 시스템 | 점수 | 비고 |
|--------|-----:|------|
| **Cairn** | **95.4%** | **1위** |
| Mastra | 94.87% | 2위 |
| Zep/Graphiti | 71.2% | -- |

## 주요 기능

- **12개 MCP 도구** — 저장, 쿼리, 검색, 체크포인트, 재개 등.
- **시맨틱 검색** — bge-small-en-v1.5 + sqlite-vec 기반의 빠르고 정확한 검색.
- **자동 캡처 및 서피싱** — Hooks가 결정을 자동 감지하고, 작업 중 관련 기억을 표시.
- **체크포인트 및 재개** — 작업 중단 후 다음 세션에서 이어서 진행.
- **지능적 망각** — 시간 감쇠, 충돌 해결, 중복 제거.
- **완전 로컬, API 키 불필요** — 모든 데이터와 처리가 로컬 머신에서 실행.

## 설치

```bash
pip3 install cairn[server]   # PyPI에서 설치 (MCP 서버 포함)
cairn setup                         # 에디터 자동 설정
cairn doctor                        # 설치 상태 확인
```

Cursor, Windsurf, Zed를 사용하시나요?

```bash
cairn setup --client cursor
cairn setup --client windsurf
cairn setup --client zed
```

## 비교

| 기능 | Cairn | CLAUDE.md | Mem0 |
|------|:-----:|:---------:|:----:|
| 세션 간 영속성 | ✅ | ✅ | ✅ |
| 시맨틱 검색 | ✅ | ❌ | ✅ |
| 자동 캡처 | ✅ | ❌ | ✅ |
| 모순 감지 | ✅ | ❌ | ❌ |
| 로컬 전용 (API 키 불필요) | ✅ | ✅ | ❌ |

---

전체 문서는 [English README](../README.md)를 참조하세요.

웹사이트: [tracqi.com](https://tracqi.com) | 문서: [tracqi.com/docs](https://tracqi.com/docs) | 벤치마크: [tracqi.com/benchmarks](https://tracqi.com/benchmarks)

## 라이선스

Apache-2.0
