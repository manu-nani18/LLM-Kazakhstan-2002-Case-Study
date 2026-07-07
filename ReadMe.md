# Kazakhstan NHDR 2002: Local LLM Extraction Pipeline

This project analyses the Kazakhstan National Human Development Report (NHDR) 2002 using a fully local large language model (LLM) pipeline. The workflow extracts structured insights from the report, compares multiple local LLMs, evaluates their outputs, and presents the results through an offline dashboard.

The project was built around the report *Rural Development in Kazakhstan: Challenges and Prospects* and focuses on extracting development indicators, chapter summaries, themes, model comparisons and dashboard-ready values.

## Features

- PDF text extraction and cleaning
- Chapter-wise document chunking
- Structured information extraction using local LLMs
- Theme distribution analysis for education, health, inequality, economy, gender, climate/environment and employment
- Numerical indicator extraction into machine-readable JSON
- Cross-model evaluation using an independent evaluator model
- Offline dashboard with six visualisations
- Fully local workflow with no cloud LLM API calls

## Technologies

- Python
- Ollama
- Qwen2.5 7B
- Llama 3.1 8B
- Gemma2 9B
- pypdf
- HTML/CSS

## Why Three Models?

Two models are used for extraction so their outputs can be compared rather than relying on a single LLM response. `qwen2.5:7b` is used for structured extraction, while `llama3.1:8b` provides an independent comparison output.

A third model, `gemma2:9b`, is used as an evaluator. This separates extraction from evaluation and reduces the chance of judging a model only by its own output style.

## Repository Structure

```text
.
+-- data/
+-- dashboard/
+-- outputs/
|   +-- llm/
|   +-- prompts/
+-- reports/
+-- src/
+-- requirements.txt
+-- ReadMe.md
```

## Pipeline

```text
PDF report
  -> raw text extraction
  -> text cleaning
  -> chapter chunking
  -> theme and evidence extraction
  -> local LLM extraction
  -> independent model evaluation
  -> dashboard and report
```

## Models

| Model | Role |
|---|---|
| `qwen2.5:7b` | Main structured extraction model |
| `llama3.1:8b` | Comparison extraction model |
| `gemma2:9b` | Independent evaluator model |

## Tested Environment

- Windows 11
- Python 3.11 / 3.12
- Ollama
- Local model storage on D drive

GPU availability depends on the machine running Ollama. The pipeline can run on CPU, but local model inference is much slower without GPU acceleration.

## Setup

Install the Python dependency:

```powershell
pip install -r requirements.txt
```

Pull the local Ollama models:

```powershell
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull gemma2:9b
```

## Run

Extract and clean the PDF:

```powershell
python src/extract_clean_text.py
```

Run extraction and evaluation:

```powershell
python src/llm_extraction.py
```

Rerun only the evaluator if extractor outputs already exist:

```powershell
python src/llm_extraction.py --evaluator-only
```

## Main Outputs

```text
outputs/cleaned_text.txt
outputs/chapter_chunks.json
outputs/theme_counts.csv
outputs/indicator_evidence_lines.json
outputs/llm/qwen2_5_7b_extraction.json
outputs/llm/llama3_1_8b_extraction.json
outputs/llm/gemma2_9b_evaluation.json
outputs/prompts/
```

The prompt logs are kept because the assignment requires the extraction and evaluation prompts to be available.

## Dashboard

Open the dashboard locally:

```text
dashboard/index.html
```

The dashboard includes six visualisations:

- theme distribution
- regional rural poverty
- agriculture share of GDP over time
- model evaluation scores
- barriers to entrepreneurship
- public social spending

## Report

The final report is available at:

```text
reports/Report.pdf
```

An editable Word version is also included:

```text
reports/Report.docx
```

## Notes

The original PDF contains table formatting and encoding noise, so table-derived numerical values should be treated with normal caution. The pipeline keeps intermediate files and prompt logs to make the extraction process reproducible and easier to inspect.
