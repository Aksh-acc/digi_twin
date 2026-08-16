"""
splits.py
=========
Loads the unified per-source jsonl and produces reproducible train/test splits.

Three ideas matter here:

1. BLOCKED splitting. Whatever unit we block on never straddles train and
   test, so "accuracy" means generalisation rather than memorisation. WHICH
   unit to block on is the critical design choice -- see the leakage note
   below, and `split_mode` in the config.

2. A FIXED test set per source. The transfer matrix needs the SAME test set
   regardless of what we trained on, otherwise the four cells aren't
   comparable. So we compute, once, a train/test split for each source, and
   every experiment reuses those exact indices.

3. LEAKAGE IS REPORTED, ALWAYS. `build_splits()` prints train/test prompt
   overlap for every source on every run, in every mode (see `_report`).

-------------------------------------------------------------------------------
LEAKAGE NOTE (why `split_mode` exists) -- read before choosing a mode
-------------------------------------------------------------------------------
The original mode ("group") blocks agents by (agent, category) and humans by
participant_id. Neither blocks on the TASK. Because all 6 agents answer
byte-identical prompts, and because only 144 distinct human prompts exist
across 571 participants, this means:

    100% of agent test rows and 100% of human test rows have their exact
    prompt string present in their own TRAIN split.

A non-learning lookup ("find this exact prompt in train, return the modal
label") scores 0.6188 on the agent test set and 0.4281 on the human test set,
recovering 93.1% / 98.8% of TF-IDF's reported 0.6646 / 0.4331. The text models
consume only `prompt` (never `group`), so under "group" mode they are
substantially learning a task->consensus lookup table.

It also breaks the transfer matrix's central metric specifically: agent and
human prompts share ZERO exact strings, so DIAGONAL cells have a memorisable
lookup available and OFF-DIAGONAL cells structurally cannot. `gap = matched -
transferred` therefore conflates a real population difference with the mere
presence/absence of that lookup.

"group" is retained as the DEFAULT purely for backward compatibility, so
previously published results stay bit-reproducible. New work should use
"prompt" or "twoway". See docs/hierarchical_twin_spec.md.
"""

import json

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

VALID_SPLIT_MODES = ("group", "prompt", "twoway")


def load_source(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def grouped_split(rows, test_size=0.2, seed=42, group_key="group"):
    """Return (train_rows, test_rows) split by group so no group leaks."""
    groups = [r[group_key] for r in rows]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(rows, groups=groups))
    train = [rows[i] for i in train_idx]
    test = [rows[i] for i in test_idx]
    return train, test


def _prompt_blocked_split(rows, test_size, seed):
    """Block on the task itself: a held-out prompt is unseen by EVERY
    decision-maker in the training set."""
    for r in rows:
        r["_split_group"] = r["prompt"]
    return grouped_split(rows, test_size, seed, "_split_group")


def _twoway_blocked_split(rows, test_size, seed, unit_key="group"):
    """Two-way blocked design: hold out a fraction of prompts AND a fraction of
    `unit_key` values (participants), then

        test  = rows where BOTH the prompt and the unit are held out
        train = rows where NEITHER is held out

    The two off-diagonal blocks (held-out prompt with seen participant, or
    vice versa) are DISCARDED -- that is exactly what makes the test set
    jointly novel on both axes, i.e. "a new person deciding about a new
    product". Costs data: with 20% held out on each axis the test set is
    ~4% of rows.

    Returns (train, test, n_discarded).
    """
    rng = np.random.RandomState(seed)

    prompts = sorted({r["prompt"] for r in rows})
    units = sorted({r[unit_key] for r in rows})
    p_perm = rng.permutation(len(prompts))
    u_perm = rng.permutation(len(units))

    n_p = max(1, int(round(len(prompts) * test_size)))
    n_u = max(1, int(round(len(units) * test_size)))
    held_prompts = {prompts[i] for i in p_perm[:n_p]}
    held_units = {units[i] for i in u_perm[:n_u]}

    train, test, n_discarded = [], [], 0
    for r in rows:
        p_held = r["prompt"] in held_prompts
        u_held = r[unit_key] in held_units
        if p_held and u_held:
            test.append(r)
        elif not p_held and not u_held:
            train.append(r)
        else:
            n_discarded += 1
    return train, test, n_discarded


def _overlap_frac(train, test, key):
    """Fraction of test rows whose `key` value also appears in train."""
    if not test:
        return 0.0
    seen = {r[key] for r in train}
    return sum(1 for r in test if r[key] in seen) / len(test)


def _report(splits, mode, discarded):
    """Print split composition + the leakage diagnostics. Always runs -- making
    prompt overlap visible by default is the structural guard against silently
    reintroducing the leak documented at the top of this module."""
    print(f"  split_mode = '{mode}'")
    for src in ("agent", "human"):
        tr, te = splits[src]["train"], splits[src]["test"]
        p_ov = _overlap_frac(tr, te, "prompt")
        g_ov = _overlap_frac(tr, te, "group")
        n_cat_tr = len({r["category"] for r in tr})
        n_cat_te = len({r["category"] for r in te})
        n_grp_tr = len({r["group"] for r in tr})
        n_grp_te = len({r["group"] for r in te})
        extra = f"  discarded={discarded[src]}" if discarded.get(src) else ""
        print(
            f"  {src}: train={len(tr)} test={len(te)}{extra}\n"
            f"      prompt overlap (test rows whose prompt is in train): {100*p_ov:6.2f}%\n"
            f"      group  overlap (test rows whose group  is in train): {100*g_ov:6.2f}%\n"
            f"      categories train/test: {n_cat_tr}/{n_cat_te}   groups train/test: {n_grp_tr}/{n_grp_te}"
        )


def build_splits(cfg, verbose=True):
    """
    Build the fixed train/test splits for both sources.

    `cfg["data"]["split_mode"]` selects the blocking design (default "group"
    for backward compatibility -- but see this module's LEAKAGE NOTE):

      "group"  : agents by (agent, category); humans by participant_id.
                 LEAKY on task identity -- 100% prompt overlap both sources.
      "prompt" : both sources blocked on the prompt itself. Held-out tasks are
                 unseen by every decision-maker. (Humans: reintroduces
                 participant overlap, since only 144 distinct human prompts
                 exist -- prompt- and participant-blocking are mutually
                 exclusive on this dataset.)
      "twoway" : agents blocked on prompt (the 6 LLMs ARE the population of
                 interest -- there is no unseen 7th model to generalize to);
                 humans blocked JOINTLY on prompt and participant, so the test
                 set is new people deciding about new products.

    Returns:
        { "agent": {"train": [...], "test": [...]},
          "human": {"train": [...], "test": [...]} }
    """
    unified = cfg["data"]["unified_dir"]
    test_size = cfg["data"]["test_size"]
    seed = cfg["data"]["seed"]
    mode = cfg["data"].get("split_mode", "group")
    if mode not in VALID_SPLIT_MODES:
        raise ValueError(f"unknown split_mode {mode!r}; expected one of {VALID_SPLIT_MODES}")

    agent_rows = load_source(f"{unified}/agent.jsonl")
    human_rows = load_source(f"{unified}/human.jsonl")
    discarded = {"agent": 0, "human": 0}

    if mode == "group":
        # Agents: group by "agent + category". Humans: group by participant.
        for r in agent_rows:
            r["_split_group"] = f"{r['group']}||{r['category']}"
        for r in human_rows:
            r["_split_group"] = r["group"]
        a_train, a_test = grouped_split(agent_rows, test_size, seed, "_split_group")
        h_train, h_test = grouped_split(human_rows, test_size, seed, "_split_group")

    elif mode == "prompt":
        a_train, a_test = _prompt_blocked_split(agent_rows, test_size, seed)
        h_train, h_test = _prompt_blocked_split(human_rows, test_size, seed)

    else:  # "twoway"
        a_train, a_test = _prompt_blocked_split(agent_rows, test_size, seed)
        h_train, h_test, discarded["human"] = _twoway_blocked_split(
            human_rows, test_size, seed, unit_key="group"
        )

    splits = {
        "agent": {"train": a_train, "test": a_test},
        "human": {"train": h_train, "test": h_test},
    }
    if verbose:
        _report(splits, mode, discarded)
    return splits
