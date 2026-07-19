# Cairn

**KI-Agenten, die sich erinnern, koordinieren und lernen. Alles auf deinem Rechner.** Das Gehirn deines Agenten sollte nicht auf dem Server eines anderen liegen.

[![PyPI version](https://img.shields.io/pypi/v/cairn.svg)](https://pypi.org/project/cairn/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![#1 on LongMemEval](https://img.shields.io/badge/LongMemEval-95.4%25_%231_Overall-gold.svg)](https://tracqi.com/benchmarks)

[🇺🇸 English](../README.md) | [🇨🇳 中文](README_zh-CN.md) | [🇯🇵 日本語](README_ja.md) | [🇰🇷 한국어](README_ko.md) | [🇧🇷 Português](README_pt-BR.md) | [🇪🇸 Español](README_es.md) | [🇫🇷 Français](README_fr.md) | [🇩🇪 Deutsch](README_de.md) | [🇷🇺 Русский](README_ru.md)

## Schnellstart

```bash
pip3 install cairn[server]
cairn setup
```

Kompatibel mit **Claude Code** | **Cursor** | **Windsurf** | **Zed** | jedem MCP-Client

---

## Warum reicht CLAUDE.md nicht aus?

Die eingebaute `CLAUDE.md` von Claude Code ist eine einfache Textdatei. Für ein paar Notizen reicht sie, aber sie stößt an ihre Grenzen, wenn:

- **Keine Suche möglich.** Ab 200 Zeilen bist du auf grep angewiesen. Cairn nutzt semantische Suche (bge-small-en-v1.5 Embeddings + sqlite-vec), um relevante Erinnerungen zu finden, auch wenn die Formulierung anders ist.
- **Kein automatisches Erfassen.** Jede Lektion muss manuell eingetragen werden. Cairn erkennt Entscheidungen und Debugging-Ergebnisse automatisch.
- **Wächst endlos.** Keine Deduplizierung, kein Verfall, keine Widerspruchserkennung. Cairn löst Konflikte automatisch, dedupliziert semantisch ähnliche Einträge und lässt veraltete Erinnerungen verfallen.
- **Eine Datei pro Projekt.** Kein projektübergreifendes Lernen. CAIRNs Gedächtnisgraph umfasst deine gesamte Entwicklungshistorie.
- **Kein Checkpoint.** Mitten im Refactoring aufgehört? Kein Weg, weiterzumachen. Cairn speichert den Aufgabenstatus und setzt exakt an der Unterbrechungsstelle fort.

CLAUDE.md taugt für „immer Tabs verwenden." Cairn ist für den Fall, dass dein Agent wirklich lernen soll.

## Benchmark

**Platz 1 bei [LongMemEval](https://github.com/xiaowu0162/LongMemEval)** (ICLR 2025) — der akademische Benchmark für Langzeitgedächtnissysteme. 500 Fragen zu Extraktion, Schlussfolgerung, Zeitverständnis und Präferenzverfolgung.

| System | Ergebnis | Anmerkung |
|--------|--------:|----------|
| **Cairn** | **95.4%** | **Platz 1** |
| Mastra | 94.87% | Platz 2 |
| Zep/Graphiti | 71.2% | -- |

## Hauptfunktionen

- **12 MCP-Tools** — Speichern, Abfragen, Suchen, Checkpoint, Fortsetzen und mehr.
- **Semantische Suche** — bge-small-en-v1.5 + sqlite-vec für schnelles, präzises Retrieval.
- **Automatisches Erfassen & Einblenden** — Hooks erkennen Entscheidungen und zeigen relevante Erinnerungen während der Arbeit.
- **Checkpoint & Fortsetzen** — Aufgabe mittendrin unterbrechen, in der nächsten Sitzung weitermachen.
- **Intelligentes Vergessen** — Zeitverfall, Konfliktlösung, Deduplizierung.
- **100% Lokal, keine API-Keys** — Alle Daten und Verarbeitung bleiben auf deinem Rechner.

## Installation

```bash
pip3 install cairn[server]   # von PyPI installieren (inkl. MCP-Server)
cairn setup                         # Editor automatisch konfigurieren
cairn doctor                        # prüfen, ob alles funktioniert
```

Du nutzt Cursor, Windsurf oder Zed?

```bash
cairn setup --client cursor
cairn setup --client windsurf
cairn setup --client zed
```

## Vergleich

| Funktion | Cairn | CLAUDE.md | Mem0 |
|----------|:-----:|:---------:|:----:|
| Persistent über Sitzungen | ✅ | ✅ | ✅ |
| Semantische Suche | ✅ | ❌ | ✅ |
| Automatisches Erfassen | ✅ | ❌ | ✅ |
| Widerspruchserkennung | ✅ | ❌ | ❌ |
| 100% lokal (keine API-Keys) | ✅ | ✅ | ❌ |

---

Die vollständige Dokumentation findest du im [englischen README](../README.md).

Website: [tracqi.com](https://tracqi.com) | Docs: [tracqi.com/docs](https://tracqi.com/docs) | Benchmarks: [tracqi.com/benchmarks](https://tracqi.com/benchmarks)

## Lizenz

Apache-2.0
