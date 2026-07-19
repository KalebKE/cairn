# Cairn

**Des agents IA qui mémorisent, coordonnent et apprennent. Tout sur votre machine.** Le cerveau de votre agent ne devrait pas être sur le serveur de quelqu'un d'autre.

[![PyPI version](https://img.shields.io/pypi/v/cairn.svg)](https://pypi.org/project/cairn/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![#1 on LongMemEval](https://img.shields.io/badge/LongMemEval-95.4%25_%231_Overall-gold.svg)](https://tracqi.com/benchmarks)

[🇺🇸 English](../README.md) | [🇨🇳 中文](README_zh-CN.md) | [🇯🇵 日本語](README_ja.md) | [🇰🇷 한국어](README_ko.md) | [🇧🇷 Português](README_pt-BR.md) | [🇪🇸 Español](README_es.md) | [🇫🇷 Français](README_fr.md) | [🇩🇪 Deutsch](README_de.md) | [🇷🇺 Русский](README_ru.md)

## Démarrage Rapide

```bash
pip3 install cairn[server]
cairn setup
```

Compatible avec **Claude Code** | **Cursor** | **Windsurf** | **Zed** | tout client MCP

---

## Pourquoi ne pas simplement utiliser CLAUDE.md ?

Le `CLAUDE.md` intégré à Claude Code est un fichier texte brut. Il fonctionne pour quelques notes, mais il atteint ses limites quand :

- **Impossible de chercher.** Au-delà de 200 lignes, vous dépendez de grep. Cairn utilise la recherche sémantique (embeddings bge-small-en-v1.5 + sqlite-vec) pour trouver les souvenirs pertinents même quand la formulation est différente.
- **Pas de capture automatique.** Chaque leçon doit être écrite manuellement. Cairn détecte les décisions et les résultats de débogage automatiquement.
- **Grossit indéfiniment.** Pas de déduplication, pas de décroissance, pas de détection de contradictions. Cairn résout les conflits automatiquement, déduplique les entrées sémantiquement similaires et fait décroître les souvenirs obsolètes.
- **Un fichier par projet.** Pas d'apprentissage inter-projets. Le graphe de mémoire d'Cairn couvre tout votre historique de développement.
- **Pas de checkpoint.** Si vous vous arrêtez au milieu d'un refactoring, impossible de reprendre. Cairn sauvegarde l'état de la tâche et reprend exactement là où vous vous êtes arrêté.

CLAUDE.md suffit pour noter « toujours utiliser les tabs ». Cairn, c'est pour quand votre agent doit vraiment apprendre.

## Benchmark

**1er au [LongMemEval](https://github.com/xiaowu0162/LongMemEval)** (ICLR 2025) — le benchmark académique de référence pour les systèmes de mémoire à long terme. 500 questions évaluant l'extraction, le raisonnement, la compréhension temporelle et le suivi des préférences.

| Système | Score | Note |
|---------|------:|------|
| **Cairn** | **95.4%** | **1er** |
| Mastra | 94.87% | 2e |
| Zep/Graphiti | 71.2% | -- |

## Fonctionnalités Principales

- **12 Outils MCP** — Stocker, interroger, chercher, checkpoint, reprendre et plus.
- **Recherche Sémantique** — bge-small-en-v1.5 + sqlite-vec pour une récupération rapide et précise.
- **Capture et Affichage Automatiques** — Les hooks détectent les décisions et affichent les souvenirs pertinents pendant le travail.
- **Checkpoint et Reprise** — Arrêtez une tâche en cours, reprenez dans la session suivante.
- **Oubli Intelligent** — Décroissance temporelle, résolution de conflits, déduplication.
- **100% Local, sans clés API** — Toutes les données et le traitement restent sur votre machine.

## Installation

```bash
pip3 install cairn[server]   # installer depuis PyPI (serveur MCP inclus)
cairn setup                         # configure l'éditeur automatiquement
cairn doctor                        # vérifie que tout fonctionne
```

Vous utilisez Cursor, Windsurf ou Zed ?

```bash
cairn setup --client cursor
cairn setup --client windsurf
cairn setup --client zed
```

## Comparaison

| Fonctionnalité | Cairn | CLAUDE.md | Mem0 |
|----------------|:-----:|:---------:|:----:|
| Persistant entre sessions | ✅ | ✅ | ✅ |
| Recherche sémantique | ✅ | ❌ | ✅ |
| Capture automatique | ✅ | ❌ | ✅ |
| Détection de contradictions | ✅ | ❌ | ❌ |
| 100% local (sans clés API) | ✅ | ✅ | ❌ |

---

Pour la documentation complète, consultez le [README en anglais](../README.md).

Site web : [tracqi.com](https://tracqi.com) | Docs : [tracqi.com/docs](https://tracqi.com/docs) | Benchmarks : [tracqi.com/benchmarks](https://tracqi.com/benchmarks)

## Licence

Apache-2.0
