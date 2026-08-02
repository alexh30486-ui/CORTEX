# Building `appsec_train.jsonl`

Format: one JSON object per line, shape:
```json
{"instruction": "Explain what an IDOR vulnerability is.", "input": "", "output": "IDOR (Insecure Direct Object Reference)..."}
```

## Sources to pull from (you already have most of this material)

1. **Sentry API scanner logic** — for each of the four OWASP API Security
   Top 10 modules it implements, write 5-10 Q&A pairs: what the
   vulnerability is, how the scanner detects it, how to remediate it.
   This is directly reusable from your scanner's own docstrings/README.

2. **Warlock payload corpus** — the 25 prompt-injection payloads across 5
   OWASP LLM Top 10 categories. For each: "what category of attack is
   this," "why does it work," "how would you defend against it." This
   doubles as training data for both the SLM *and* a future upgrade to
   security/input_guard.py's classifier stub.

3. **OWASP API Security Top 10 + OWASP LLM Top 10 official docs** — pull
   the canonical descriptions and paraphrase into instruction/output
   pairs. Don't copy verbatim (both licensing and it makes for worse
   training data anyway — paraphrased explanations generalize better).

4. **CompTIA Security+ study material** — you already have this; if any
   of it is in your own words (flashcards, notes), it's fair game and
   adds broader security fundamentals beyond just appsec/LLM security.

## Target size

Aim for 300-800 examples for a first pass. LoRA on a 1.5B model can show
meaningful behavior shift with a few hundred well-curated examples —
quality and topic coverage matter more than raw volume here.

## Quick generation shortcut

You can bootstrap this faster by using a larger model (Claude, GPT-4o,
whatever you have API access to) as a *teacher*: feed it your scanner
docstrings / Warlock payload list, ask it to generate instruction/output
pairs in this schema, spot-check for accuracy, then use the result as
your training set. This is literally the "distillation" story from the
project pitch — document it as such, it's a legitimate technique, not
a shortcut to hide.
