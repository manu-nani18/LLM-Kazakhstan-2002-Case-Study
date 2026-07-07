"""local LLM extraction and evaluation with Ollama.

Models used for the assignment:
- qwen2.5:7b as the main structured extractor
- llama3.1:8b as the comparison extractor
- gemma2:9b as the independent evaluator


"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LLM_OUTPUT_DIR = OUTPUT_DIR / "llm"
PROMPT_DIR = OUTPUT_DIR / "prompts"
CHAPTERS_PATH = OUTPUT_DIR / "chapter_chunks.json"
CLEANED_TEXT_PATH = OUTPUT_DIR / "cleaned_text.txt"
THEME_COUNTS_PATH = OUTPUT_DIR / "theme_counts.csv"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"

EXTRACTOR_MODELS = ["qwen2.5:7b", "llama3.1:8b"]
EVALUATOR_MODEL = "gemma2:9b"

THEME_KEYWORDS = {
    "education": ["education", "school", "schools", "enrol", "literacy", "teacher", "pre-school"],
    "health": ["health", "medical", "morbidity", "mortality", "disease", "life expectancy"],
    "inequality": ["inequality", "poverty", "poor", "low-income", "vulnerable", "urban", "rural"],
    "economy": ["economy", "economic", "agriculture", "income", "GDP", "wages", "market", "farm"],
    "gender": ["gender", "women", "men", "female", "male"],
    "climate_environment": ["environment", "climate", "water", "land", "soil", "pasture", "irrigation"],
    "employment": ["employment", "unemployment", "jobs", "labour", "work", "migration"],
}

INDICATOR_KEYWORDS = [
    "HDI",
    "Human Development Index",
    "life expectancy",
    "GDP",
    "income",
    "poverty",
    "population",
    "rural population",
    "urban",
    "education",
    "enrol",
    "literacy",
    "health",
    "mortality",
    "wages",
    "employment",
    "unemployment",
]


def slug_model(model: str) -> str:
    return model.replace(":", "_").replace(".", "_")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def check_ollama() -> None:
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=5) as response:
            response.read()
    except Exception as exc:
        print("Ollama is not running or not installed.")
        print("Install Ollama, start it, then run:")
        print("  ollama pull qwen2.5:7b")
        print("  ollama pull llama3.1:8b")
        print("  ollama pull gemma2:9b")
        print("Then rerun: python src/llm_extraction.py")
        raise SystemExit(1) from exc


def ollama_json(model: str, prompt: str, temperature: float = 0.1, retries: int = 2) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful data science assistant. Return only valid JSON. "
                    "Do not invent facts. Use null when the source text does not provide a value."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "")
            return parse_json_response(content)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(3)
    raise RuntimeError(f"Ollama call failed for {model}: {last_error}")


def parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content, flags=re.I).strip()
        content = re.sub(r"```$", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def count_themes(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chapter in chapters:
        text = chapter["text"]
        lower = text.lower()
        for theme, keywords in THEME_KEYWORDS.items():
            count = sum(len(re.findall(rf"\b{re.escape(keyword.lower())}\b", lower)) for keyword in keywords)
            rows.append(
                {
                    "chapter_number": chapter["chapter_number"],
                    "chapter_title": chapter["title"],
                    "theme": theme,
                    "keyword_count": count,
                }
            )
    return rows


def write_theme_counts(rows: list[dict[str, Any]]) -> None:
    with THEME_COUNTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chapter_number", "chapter_title", "theme", "keyword_count"])
        writer.writeheader()
        writer.writerows(rows)


def evidence_lines(text: str, max_lines: int = 90) -> list[str]:
    lines: list[str] = []
    for raw_line in re.split(r"(?<=[.!?])\s+|\n+", text):
        line = raw_line.strip()
        if len(line) < 35:
            continue
        lower = line.lower()
        has_keyword = any(keyword.lower() in lower for keyword in INDICATOR_KEYWORDS)
        has_number = bool(re.search(r"\d", line))
        if has_keyword and has_number:
            lines.append(line[:500])
        if len(lines) >= max_lines:
            break
    return lines


def compact_chapter_context(chapter: dict[str, Any], limit_chars: int = 9000) -> str:
    text = chapter["text"]
    if len(text) <= limit_chars:
        return text
    head = text[: int(limit_chars * 0.65)]
    tail = text[-int(limit_chars * 0.35) :]
    return head + "\n\n[...middle section removed for prompt length...]\n\n" + tail


def chapter_summary_prompt(chapter: dict[str, Any], model: str) -> str:
    return f"""
You are extracting information from the Kazakhstan National Human Development Report 2002.
Model role: {model} extractor.

Chapter {chapter['chapter_number']}: {chapter['title']}
Source pages: {chapter['start_page']}-{chapter['end_page']}

TASK:
Return JSON with this exact structure:
{{
  "chapter_number": {chapter['chapter_number']},
  "chapter_title": "{chapter['title']}",
  "summary_under_100_words": "chapter summary",
  "key_points": ["specific key point"],
  "important_indicators_or_numbers": [
    {{"indicator": "indicator name", "value": "reported value", "context": "source context", "source_page_range": "{chapter['start_page']}-{chapter['end_page']}"}}
  ],
  "themes_present": ["education", "health", "inequality", "economy", "gender", "climate_environment", "employment"],
  "limitations_or_uncertainties": ["specific limitation"]
}}

Rules:
- Summary must be fewer than 100 words.
- Use only the chapter text.
- If no numerical value is found, use an empty list.

CHAPTER TEXT:
{compact_chapter_context(chapter)}
""".strip()


def full_report_prompt(chapter_outputs: list[dict[str, Any]], evidence: list[str], theme_rows: list[dict[str, Any]], model: str) -> str:
    theme_totals: dict[str, int] = {}
    for row in theme_rows:
        theme_totals[row["theme"]] = theme_totals.get(row["theme"], 0) + int(row["keyword_count"])

    return f"""
You are extracting structured development intelligence from the Kazakhstan National Human Development Report 2002.
Model role: {model} extractor.

Use these chapter-level model outputs, indicator evidence lines from the PDF, and deterministic theme keyword counts.

CHAPTER OUTPUTS JSON:
{json.dumps(chapter_outputs, ensure_ascii=False, indent=2)}

INDICATOR EVIDENCE LINES FROM PDF:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

THEME KEYWORD TOTALS:
{json.dumps(theme_totals, ensure_ascii=False, indent=2)}

TASK:
Return JSON with this exact structure:
{{
  "model": "{model}",
  "report_title": "Rural Development in Kazakhstan: Challenges and Prospects",
  "country": "Kazakhstan",
  "report_year": 2002,
  "full_report_key_findings": ["5-8 concise bullets"],
  "key_strengths": ["max 5-8 items"],
  "key_challenges": ["max 5-8 items"],
  "core_indicators": [
    {{"indicator": "indicator name", "value": "reported value", "unit": "unit", "year_or_period": "year or period", "context": "source context", "source_basis": "chapter/evidence line"}}
  ],
  "theme_analysis": [
    {{"theme": "education", "importance": "high/medium/low", "evidence_summary": "theme evidence summary", "keyword_count": 0}}
  ],
  "time_or_demographic_trends_for_plots": [
    {{"trend_name": "trend name", "values_or_description": "values or description", "plot_idea": "chart suggestion"}}
  ],
  "dashboard_ready_plot_ideas": ["at least 4 plot ideas"],
  "extraction_notes": ["specific extraction note"]
}}

Rules:
- Use only the provided chapter outputs and evidence lines.
- Preserve numerical values exactly when possible.
- If a value is uncertain, state that uncertainty in context.
""".strip()


def compact_output_for_evaluation(result: dict[str, Any]) -> dict[str, Any]:
    full = result.get("full_report_extraction", {})
    return {
        "model": result.get("model"),
        "chapter_summaries": [
            {
                "chapter_number": chapter.get("chapter_number"),
                "summary": chapter.get("summary_under_100_words"),
                "key_points": chapter.get("key_points", [])[:4],
                "important_indicators_or_numbers": chapter.get("important_indicators_or_numbers", [])[:3],
                "limitations_or_uncertainties": chapter.get("limitations_or_uncertainties", [])[:2],
            }
            for chapter in result.get("chapter_summaries", [])
        ],
        "full_report_key_findings": full.get("full_report_key_findings", [])[:8],
        "core_indicators": full.get("core_indicators", [])[:8],
        "theme_analysis": full.get("theme_analysis", []),
        "dashboard_ready_plot_ideas": full.get("dashboard_ready_plot_ideas", [])[:6],
        "extraction_notes": full.get("extraction_notes", [])[:4],
    }


def evaluation_prompt(qwen_output: dict[str, Any], llama_output: dict[str, Any], evidence: list[str]) -> str:
    qwen_eval = compact_output_for_evaluation(qwen_output)
    llama_eval = compact_output_for_evaluation(llama_output)
    compact_evidence = evidence[:35]

    return f"""
You are the independent evaluator for a local LLM extraction assignment.
Source: Kazakhstan National Human Development Report 2002.

Evaluate the two extractor model outputs against the evidence lines. Score each model from 1 to 5 for consistency, completeness, factual_alignment, numerical_extraction, and dashboard_usefulness.
Do not return generic filler text. Every score must be an integer from 1 to 5.

QWEN OUTPUT:
{json.dumps(qwen_eval, ensure_ascii=False, indent=2)}

LLAMA OUTPUT:
{json.dumps(llama_eval, ensure_ascii=False, indent=2)}

SOURCE EVIDENCE LINES:
{json.dumps(compact_evidence, ensure_ascii=False, indent=2)}

Return JSON with this exact structure:
{{
  "evaluator_model": "{EVALUATOR_MODEL}",
  "model_scores": [
    {{
      "model": "qwen2.5:7b",
      "consistency": "integer 1-5",
      "completeness": "integer 1-5",
      "factual_alignment": "integer 1-5",
      "numerical_extraction": "integer 1-5",
      "dashboard_usefulness": "integer 1-5",
      "strengths": ["specific strength"],
      "weaknesses": ["specific weakness"],
      "possible_hallucinations_or_unsupported_claims": ["specific unsupported claim or none found"]
    }},
    {{
      "model": "llama3.1:8b",
      "consistency": "integer 1-5",
      "completeness": "integer 1-5",
      "factual_alignment": "integer 1-5",
      "numerical_extraction": "integer 1-5",
      "dashboard_usefulness": "integer 1-5",
      "strengths": ["specific strength"],
      "weaknesses": ["specific weakness"],
      "possible_hallucinations_or_unsupported_claims": ["specific unsupported claim or none found"]
    }}
  ],
  "agreement_between_models": ["specific agreement"],
  "important_disagreements": ["specific disagreement"],
  "recommended_final_extraction": {{
    "best_model_for_summaries": "model name",
    "best_model_for_indicators": "model name",
    "items_to_verify_manually": ["specific item"]
  }}
}}
""".strip()


def evaluation_has_placeholders(evaluation: dict[str, Any]) -> bool:
    text = json.dumps(evaluation, ensure_ascii=False)
    if '"..."' in text:
        return True
    for score in evaluation.get("model_scores", []):
        numeric_scores = [
            score.get("consistency"),
            score.get("completeness"),
            score.get("factual_alignment"),
            score.get("numerical_extraction"),
            score.get("dashboard_usefulness"),
        ]
        if any(not isinstance(value, int) or value < 1 or value > 5 for value in numeric_scores):
            return True
    return False


def run_evaluator(qwen_result: dict[str, Any], llama_result: dict[str, Any], evidence: list[str]) -> dict[str, Any]:
    prompt = evaluation_prompt(qwen_result, llama_result, evidence)
    (PROMPT_DIR / f"{slug_model(EVALUATOR_MODEL)}_evaluation.txt").write_text(prompt, encoding="utf-8")
    print(f"Running evaluator {EVALUATOR_MODEL}...")
    evaluation = ollama_json(EVALUATOR_MODEL, prompt)
    if evaluation_has_placeholders(evaluation):
        raise RuntimeError(
            "Evaluator returned placeholder or invalid scores. "
            "Try rerunning, or inspect outputs/prompts/gemma2_9b_evaluation.txt."
        )
    write_json(LLM_OUTPUT_DIR / f"{slug_model(EVALUATOR_MODEL)}_evaluation.json", evaluation)
    return evaluation


def run_extractor(model: str, chapters: list[dict[str, Any]], evidence: list[str], theme_rows: list[dict[str, Any]]) -> dict[str, Any]:
    chapter_outputs: list[dict[str, Any]] = []
    for chapter in chapters:
        prompt = chapter_summary_prompt(chapter, model)
        prompt_path = PROMPT_DIR / f"{slug_model(model)}_chapter_{chapter['chapter_number']}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"Running {model} on chapter {chapter['chapter_number']}...")
        chapter_outputs.append(ollama_json(model, prompt))

    full_prompt = full_report_prompt(chapter_outputs, evidence, theme_rows, model)
    (PROMPT_DIR / f"{slug_model(model)}_full_report.txt").write_text(full_prompt, encoding="utf-8")
    print(f"Running {model} full-report extraction...")
    full_output = ollama_json(model, full_prompt)

    result = {
        "model": model,
        "chapter_summaries": chapter_outputs,
        "full_report_extraction": full_output,
    }
    write_json(LLM_OUTPUT_DIR / f"{slug_model(model)}_extraction.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Ollama LLM extraction/evaluation.")
    parser.add_argument("--skip-evaluator", action="store_true", help="Run only extractor models.")
    parser.add_argument("--evaluator-only", action="store_true", help="Reuse existing extractor outputs and run only the evaluator.")
    args = parser.parse_args()

    if not CHAPTERS_PATH.exists() or not CLEANED_TEXT_PATH.exists():
        print("Missing cleaned text outputs. Run: python src/extract_clean_text.py")
        raise SystemExit(1)

    LLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)

    chapters = read_json(CHAPTERS_PATH)
    cleaned_text = CLEANED_TEXT_PATH.read_text(encoding="utf-8")
    theme_rows = count_themes(chapters)
    write_theme_counts(theme_rows)
    chapter_source_text = "\n".join(chapter["text"] for chapter in chapters)
    evidence = evidence_lines(chapter_source_text)
    write_json(OUTPUT_DIR / "indicator_evidence_lines.json", evidence)

    check_ollama()

    if args.evaluator_only:
        qwen_path = LLM_OUTPUT_DIR / f"{slug_model(EXTRACTOR_MODELS[0])}_extraction.json"
        llama_path = LLM_OUTPUT_DIR / f"{slug_model(EXTRACTOR_MODELS[1])}_extraction.json"
        if not qwen_path.exists() or not llama_path.exists():
            print("Missing extractor outputs. Run without --evaluator-only first.")
            raise SystemExit(1)
        run_evaluator(read_json(qwen_path), read_json(llama_path), evidence)
        print("Done. Evaluator output is in outputs/llm")
        return

    extractor_results = []
    for model in EXTRACTOR_MODELS:
        extractor_results.append(run_extractor(model, chapters, evidence, theme_rows))

    if not args.skip_evaluator:
        run_evaluator(extractor_results[0], extractor_results[1], evidence)

    print("Done. LLM outputs are in outputs/llm")
    print("Theme counts are in outputs/theme_counts.csv")


if __name__ == "__main__":
    main()
