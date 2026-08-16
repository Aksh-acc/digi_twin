"""
data_prep.py
============
Builds a single, unified dataset from two very different sources so that the
same downstream models can be trained/tested on either one:

  1. AGENTIC data  -- shopping decisions produced by LLM "shopping agents".
     Six agents are present (Claude Opus 4.6 / Sonnet 4.6 / Opus 4.8,
     ChatGPT O3, ChatGPT GPT-5.5, Grok 4.2), each answering the *same* 384
     product-choice prompts. Stored as .xlsx with a `full_chat_prompt`
     column and a `chosen_option` column (A/B/C/D/NONE).

  2. HUMAN data    -- real survey respondents' actual choices on the real
     product options they were shown. Already pre-built into chat-format
     .jsonl by the original project (llm_train_human.jsonl / llm_val_human).

Both are reduced to the SAME schema:

    { "prompt": "<full shopping prompt>",
      "label" : "A" | "B" | "C" | "D" | "NONE",
      "source": "agent" | "human",
      "group" : "<agent name>" or "<participant_id>",   # for grouped splits
      "category": "AIR" | "COF" | "EAR" | "SNK" }

The `prompt` text is what every model consumes. The `label` is what every
model predicts. Keeping the schema identical is what makes the
train-on-X / test-on-Y "gap" measurable at all.

Run:
    python -m src.data_prep --config configs/config.yaml
"""

import argparse
import glob
import json
import os
import re

import pandas as pd
import yaml


VALID = {"A", "B", "C", "D", "NONE"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _norm_choice(v):
    """Normalise a raw chosen_option cell to one of A/B/C/D/NONE, else None."""
    if v is None:
        return None
    s = str(v).strip().upper()
    s = s.replace("OPTION", "").replace(":", "").strip()
    # sometimes stored as a JSON list or with extra words -- grab first A-D/NONE
    m = re.search(r"\b(A|B|C|D|NONE)\b", s)
    return m.group(1) if m else None


def _category_from_prompt(prompt):
    """Infer AIR/COF/EAR/SNK from the product category line in the prompt."""
    p = prompt.lower()
    if "air purifier" in p:
        return "AIR"
    if "coffee" in p:
        return "COF"
    if "earbud" in p:
        return "EAR"
    if "sneaker" in p:
        return "SNK"
    return "UNK"


def _choice_from_raw_response(raw):
    """Grok's NO1-50 file has no chosen_option column but a raw JSON response."""
    if raw is None:
        return None
    try:
        start = str(raw).find("{")
        end = str(raw).rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(str(raw)[start : end + 1])
            return _norm_choice(obj.get("chosen_option"))
    except Exception:
        pass
    m = re.search(r'"chosen_option"\s*:\s*"(A|B|C|D|NONE)"', str(raw))
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Agentic loader
# --------------------------------------------------------------------------- #
def load_agentic(agentic_root):
    """
    Walk the 'Agentic response' tree. Each leaf agent directory holds several
    .xlsx files. Returns a list of unified records.
    """
    records = []
    # Map a directory path to a clean agent name.
    # Structure is e.g. .../Agentic response/Claude/Opus 4.8/*.xlsx
    #                    .../Agentic response/Chatgpt/O3/*.xlsx
    #                    .../Agentic response/Grok 4.2/Grok 4.2/*.xlsx
    xlsx_files = glob.glob(os.path.join(agentic_root, "**", "*.xlsx"), recursive=True)
    xlsx_files = [f for f in xlsx_files if "~$" not in f]  # skip temp lock files

    for fp in sorted(xlsx_files):
        rel = os.path.relpath(fp, agentic_root)
        parts = rel.split(os.sep)
        # agent name = family + variant, deduplicated (Grok 4.2/Grok 4.2 -> Grok 4.2)
        family = parts[0]
        variant = parts[1] if len(parts) > 2 else parts[0]
        if variant == family:
            agent = family
        else:
            agent = f"{family} {variant}"
        agent = agent.replace("Chatgpt", "ChatGPT").strip()

        try:
            df = pd.read_excel(fp)
        except Exception as e:
            print(f"  [skip] {rel}: {e}")
            continue
        df.columns = [str(c).strip() for c in df.columns]

        prompt_col = "full_chat_prompt" if "full_chat_prompt" in df.columns else None
        choice_col = None
        for c in df.columns:
            if c.lower() == "chosen_option":
                choice_col = c
                break

        for _, row in df.iterrows():
            label = _norm_choice(row[choice_col]) if choice_col else None
            if label is None and "raw_response" in df.columns:
                label = _choice_from_raw_response(row.get("raw_response"))
            if label not in VALID:
                continue

            prompt = str(row[prompt_col]) if prompt_col else None
            if not prompt or prompt == "nan":
                # fall back to raw_response context is not a usable prompt; skip
                continue

            records.append(
                {
                    "prompt": prompt,
                    "label": label,
                    "source": "agent",
                    "group": agent,
                    "category": _category_from_prompt(prompt),
                }
            )
    return records


# --------------------------------------------------------------------------- #
# Human loader
# --------------------------------------------------------------------------- #
def load_human(human_files):
    """Load the pre-built human chat-format jsonl file(s) into unified records."""
    records = []
    for fp in human_files:
        if not os.path.exists(fp):
            print(f"  [warn] human file missing: {fp}")
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                prompt = ex["messages"][0]["content"]
                assistant = ex["messages"][1]["content"]
                try:
                    label = _norm_choice(json.loads(assistant).get("chosen_option"))
                except Exception:
                    label = _norm_choice(assistant)
                if label not in VALID:
                    continue
                records.append(
                    {
                        "prompt": prompt,
                        "label": label,
                        "source": "human",
                        "group": ex.get("participant_id", "unknown"),
                        "category": _category_from_prompt(prompt),
                    }
                )
    return records


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["data"]["unified_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("Loading AGENTIC data ...")
    agent_records = load_agentic(cfg["data"]["agentic_root"])
    print(f"  -> {len(agent_records)} agentic records")

    print("Loading HUMAN data ...")
    human_records = load_human(cfg["data"]["human_files"])
    print(f"  -> {len(human_records)} human records")

    # Save each source separately (they get split independently downstream).
    agent_path = os.path.join(out_dir, "agent.jsonl")
    human_path = os.path.join(out_dir, "human.jsonl")
    with open(agent_path, "w", encoding="utf-8") as f:
        for r in agent_records:
            f.write(json.dumps(r) + "\n")
    with open(human_path, "w", encoding="utf-8") as f:
        for r in human_records:
            f.write(json.dumps(r) + "\n")

    # Quick summary
    def dist(records, key):
        from collections import Counter

        return dict(Counter(r[key] for r in records))

    print("\n=== SUMMARY ===")
    print(f"Agent  : {len(agent_records)} rows")
    print(f"  by agent    : {dist(agent_records, 'group')}")
    print(f"  by label    : {dist(agent_records, 'label')}")
    print(f"  by category : {dist(agent_records, 'category')}")
    print(f"Human  : {len(human_records)} rows")
    print(f"  by label    : {dist(human_records, 'label')}")
    print(f"  by category : {dist(human_records, 'category')}")
    print(f"\nSaved:\n  {agent_path}\n  {human_path}")


if __name__ == "__main__":
    main()
