# NJUPT Winter Battle — AMD ROCm

[中文说明](./README_ZH.md)

## Activity Overview

| Field | Details |
|-------|---------|
| **Activity** | NJUPT Winter Battle — AMD ROCm (南京邮电大学寒假大作战) |
| **Theme** | LLM Application Development based on ROCm |
| **Submission Deadline** | March 19, 2026 |
| **Defense** | March 22, 2026 (on-site) |
| **Venue** | Nanjing University of Posts and Telecommunications |
| **Platform** | aup-learning-cloud — AMD GPU Cluster (JupyterHub) |

## Background

AMD ROCm is an open-source GPU computing platform that enables high-performance AI workloads on AMD hardware. This competition challenges students to build interactive LLM-powered applications using the ROCm-based cluster environment.

**References:**
- ROCm Official Docs: https://rocm.docs.amd.com/
- ROCm GitHub: https://github.com/ROCm/ROCm
- AMD at CES 2025: https://www.amd.com/zh-cn/newsroom/press-releases/2025-1-6-amd-announces-expanded-consumer-and-commercial-ai-.html
- AMD at CES 2026: https://www.amd.com/zh-cn/newsroom/press-releases/2026-1-5-amd-expands-ai-leadership-across-client-graphics-.html

## Competition Theme: LLM Application Development

**Goal:** Build an interactive intelligent application (e.g. Q&A system, chatbot) based on the ROCm cluster.

### Development Environment

| Component | Details |
|-----------|---------|
| **Platform** | Remote JupyterHub (Python 3.12+) via aup-learning-cloud |
| **Ollama API Endpoint** | `open-webui-ollama.open-webui:11434` |
| **Available Models** | `qwen3-coder:30b`, `gpt-oss:20b` |
| **Context Window** | 32K tokens |
| **Recommended UI** | `ipywidgets` for in-notebook interaction |
| **API Reference** | https://ollama.readthedocs.io/api/ |

### Recommended Topics

#### Topic 1: Domain Knowledge Q&A Assistant (RAG)
Build a Q&A system for a specific domain (e.g. campus regulations, course materials, programming docs).
- **Required:** Multi-turn dialogue, memory mechanism, async/streaming API, Chinese & English support
- **Advanced:** Full RAG pipeline (document parsing → vector retrieval → augmented generation), source citation

#### Topic 2: Text Intelligence Analysis & Report Assistant (Data Processing)
Use LLMs for structured text extraction and analysis.
- **Required:** Summarization, sentiment analysis, keyword extraction; formatted Markdown output
- **Advanced:** Batch processing pipeline (multi-thread/async), cross-document comparison report

#### Topic 3: Code Assistant Programming Expert (Engineering)
Build a code explanation and debugging assistant for beginners.
- **Required:** Code explanation (step-by-step), error analysis, Python support (Java/C++ optional)
- **Advanced:** Auto-generate project scaffolding and test cases, code style review

#### Topic 4: Interactive Narrative & Logic Analysis (Creative)
Use LLMs to generate story continuations with logic consistency monitoring.
- **Required:** Multi-turn story continuation, genre selection (sci-fi / mystery / fantasy), basic structure check
- **Advanced:** Consistency detection (character/scene state tracking), structural scoring (three-act analysis)

### Technical Requirements

- **Prompt Engineering:** Document your prompt iteration process (constraints, few-shot examples, Chain-of-Thought)
- **Cluster API Usage:** Code must demonstrate calls to the cluster's internal Ollama API
- **Error Handling:** Must handle API timeouts, empty inputs, and context overflow (32K limit) via at least one strategy:
  - Chunking & Sliding Window
  - Recursive Summarization
  - Truncation & Prompt Compression

## Awards

| Award | Quantity | Prize |
|-------|----------|-------|
| 1st Place | 1 team | PYNQ Z2 Development Board × 1 + AMD Custom T-shirt × 2 + AMD Custom Backpack × 1 |
| 2nd Place | 2 teams | Spartan Edge FPGA Board × 1 + AMD Custom Hat × 2 + AMD Custom Backpack × 1 |
| 3rd Place | 4 teams | AMD Custom Mug × 2 + AMD Custom Backpack × 1 |
| Excellence Award | 10 teams | AMD Custom Backpack × 2 |
| Participation Award | All participants | *Reconfigurable Computing* book × 1 (first-come-first-served) |

## Learning Resources

- DataWhale: https://www.datawhale.cn/
- DataWhale GitHub: https://github.com/datawhalechina
- AMD ModelScope Community: https://modelscope.cn/brand/view/AMDCommunity

## Submissions

> Student projects are submitted via Pull Request. See [CONTRIBUTING.md](../../CONTRIBUTING.md) for instructions.

| Folder | Team | Project |
|--------|------|---------|
| _(submissions will appear here after PR merge)_ | — | — |

## How to Experience All Cases

1. Open aup-learning-cloud → select **Basic GPU Environment**
2. Set Git URL: `https://github.com/amdjiahangpan/aup-learning-cloud-case-hub`
3. After startup, navigate to `cases/2026-03-njupt-winter-battle/`
4. Open any submission folder and run `main.ipynb`
