# Local Voice Agent

Local Voice Agent is a learning-focused, local-first Python application for turning English audio recordings into timestamped transcripts, structured meeting or lecture summaries, and evidence-based answers.

> Status: transcription and hierarchical summarization slices are working. English
> audio can be transcribed locally, stored with timestamps, split into checkpoints,
> and summarized by a local Ollama model.

## MVP pipeline

```text
English audio file
       ↓
faster-whisper
       ↓
timestamped transcript
       ↓
approximately ten-minute, sentence-aware chunks
       ↓
local Ollama model
       ↓
checkpoint summaries, important notes, topics, and overall summary
       ↓
SQLite
       ↓
manual agent loop with validated tools
       ↓
evidence-based answers with timestamps
```

The application will process audio extracted from a supported local recording. The LLM will receive transcript text, not the original video file.

## Initial scope

- English-only local audio processing
- Local transcription with `faster-whisper`
- Timestamped transcript segments
- Structured summaries validated with Pydantic
- SQLite persistence
- A small manually implemented tool-calling agent
- A command-line interface with an optional debug mode

Speaker diarization, realtime recording, GUIs, mobile clients, embeddings, and cross-session retrieval are intentionally excluded from the first MVP.

## Current capabilities

- Check the local Python, faster-whisper, Pydantic, and Ollama environment.
- Process one local English audio file with `faster-whisper`.
- Store session metadata and timestamped transcript segments in SQLite.
- Display both audio duration and elapsed processing time for completed sessions.
- List stored sessions.
- Print a stored transcript with readable timestamps.
- Search a stored transcript with bounded, relevance-ranked results.
- Record failed processing attempts without losing the error context.
- Group transcript segments into approximately ten-minute chunks and prefer complete
  sentence boundaries near each target.
- Generate a short summary and important notes for every checkpoint with Ollama.
- Combine checkpoints into an overall summary, important notes, and main topics.
- Reject structurally invalid or empty note responses, with one correction attempt.
- Record total summary time, final-summary time, and individual retry times.
- Preserve every successful summary as a separately identifiable run for model and
  prompt comparisons.
- Store and display checkpoint evidence ranges and final summaries in SQLite.

Transcript search, range retrieval, and the first read-only question-answering agent loop are now
available. A versioned transcript QA dataset and deterministic evaluation runner provide the first
baseline for comparing models, agent modes, and future retrieval changes.

## Setup

Python 3.12 or newer and Ollama are expected to be installed locally. On PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m local_voice_agent doctor
```

Pull or verify the default local summary model:

```powershell
ollama pull qwen3.5:4b
```

The initial defaults use `small.en` on CPU with `int8` computation. The first use of a Whisper model downloads its weights from Hugging Face. Audio decoding is handled by PyAV, so a separate system FFmpeg installation is not required by `faster-whisper`.

## Usage

Process an English audio file:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent process C:\path\to\lecture.mp3
```

For a quick, lower-resource capability check, select the smaller model:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent process C:\path\to\sample.wav --model tiny.en
```

List stored sessions and inspect one transcript:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent sessions
.\.venv\Scripts\python.exe -m local_voice_agent show 1
```

Search a stored transcript. Quotation marks keep a multi-word query as one argument:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent search 2 "oral history"
.\.venv\Scripts\python.exe -m local_voice_agent search 2 "poetry" --limit 10
```

Search is local and deterministic: it does not call Whisper or Ollama. Exact phrases rank above
partial term matches, and `--limit` is restricted to 1–20 results.

Read a specific transcript interval using seconds, `MM:SS`, or `HH:MM:SS`:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent transcript-range 2 08:00 10:00
```

One request is limited to ten minutes. Segments that overlap either boundary are included so words
are not cut off at the requested timestamps.

Create approximately ten-minute checkpoints and a final summary from an existing transcript:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent summarize 2 `
    --content-mode informational
.\.venv\Scripts\python.exe -m local_voice_agent summary 2
```

The default model is `qwen3.5:4b`. To try the larger 9B model, download it once and
then select it explicitly when summarizing:

```powershell
ollama pull qwen3.5:9b
.\.venv\Scripts\python.exe -m local_voice_agent summarize 2 --model qwen3.5:9b `
    --content-mode informational --label "9b simple notes"
.\.venv\Scripts\python.exe -m local_voice_agent summary 2
```

Summarization reads the stored transcript and does not run Whisper again. Every successful
run is appended to history; it no longer replaces an earlier summary. List the runs and open
one specific result with:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent summary-runs 2
.\.venv\Scripts\python.exe -m local_voice_agent summary 2 --run-id 3
```

Without `--run-id`, `summary` shows the newest successful run. Existing summaries are migrated
to run history automatically.
The database schema is upgraded automatically while preserving existing sessions and transcripts.

List every important note and checkpoint note stored in all successful summary runs:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent notes
.\.venv\Scripts\python.exe -m local_voice_agent notes --session-id 2
.\.venv\Scripts\python.exe -m local_voice_agent notes --session-id 2 --run-id 3
```

Ask a question about one stored recording:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent ask 2 `
    "How did writing change poetry?" --model qwen3.5:4b
```

Inspect the model's selected actions while developing:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent ask 2 `
    "How did writing change poetry?" --model qwen3.5:4b --debug
```

The default `final` mode uses a bounded manual retrieval loop followed by a separate tool-free
`FinalAnswer` phase. The model returns claims with transcript segment IDs, the application validates
those IDs against retrieved evidence, and the application—not the model—renders timestamp
citations. Once retrieval finishes, the model cannot call another tool.

Run the original manual agent loop as a comparison baseline:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent ask 2 `
    "How did writing change poetry?" --model qwen3.5:4b `
    --agent-mode manual --debug
```

Both modes have only three read-only tools: bounded transcript search, bounded transcript range
retrieval, and the selected session's stored summary. A supported factual answer requires both a
search and inspection of a relevant transcript range. The loop has a fixed step limit.

For range inspection, the model must select a `segment_id` returned by search. The application—not
the model—then reads 30 seconds on both sides of that segment. Invalid selections receive at most
two bounded repair attempts. Transcript search expands common English word forms and synonyms, and
the stored session summary supplies related vocabulary for ranking search results. The summary is
planning context only; final claims must still be verified from transcript segments.

After each retrieved passage, the agent checks whether it directly answers the original question.
It can make one different follow-up search when the first passage is irrelevant or incomplete, and
it cannot declare evidence insufficient until both searches have been tried. In default `final`
mode, two searches plus two transcript reads automatically trigger FinalAnswer. Invalid FinalAnswer
outputs have a separate three-repair allowance and do not consume the configured step budget. With
`--debug`, rejection reasons and the transition to the tool-free final phase are printed. The
original question is repeated after every retrieval result to reduce topic drift.

The default database is `data/local_voice_agent.db`. Put `--database PATH` before the subcommand to use another database.

### Transcript QA evaluations

`evals/poetry_session_2.json` contains 12 hand-authored cases: eight focused questions, two synthesis
questions, and two deliberately unanswerable questions. Evidence spans the substantive poetry
discussion; the introductory contents, term lists, references, and recording metadata are excluded.
The dataset stores reference answers and required points for human review. Nothing in this runner
grades natural-language correctness or calls an LLM judge.

Run the forced FinalAnswer baseline from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent evaluate evals/poetry_session_2.json `
    --session-id 2 `
    --model qwen3.5:4b `
    --agent-mode final `
    --label "4b lexical final baseline"
```

Run the original manual agent comparison with the same session and model:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent evaluate evals/poetry_session_2.json `
    --session-id 2 `
    --model qwen3.5:4b `
    --agent-mode manual `
    --label "4b lexical manual baseline"
```

With the virtual environment activated, `local-voice-agent evaluate ...` is equivalent. Use
`--model qwen3.5:9b` and a corresponding label to compare the larger model after downloading it.
A full 12-question local-model evaluation may take a substantial amount of time: each question
can require several model requests and repairs. `--timeout` is the timeout for **each HTTP
request**, not a total limit for a case or run. Progress is printed before and after each case;
ordinary case errors are saved and the next case still runs. The command exits with status 1 if
any case failed, and 0 when all selected cases completed, regardless of their metric scores.

Use `--limit 1` for a small run, or repeat `--case-id ID` to select named cases. IDs are listed in
the JSON dataset. Case filters are applied in dataset order, followed by the limit. Unknown IDs
and invalid limits fail before model calls. The complete dataset is validated even for a subset.
For comparisons, hold `--max-steps` (default 5) and the stored summary context constant. Pin a
summary with `--run-id ID` (find IDs with `summary-runs 2`); otherwise the latest summary is used.
The summary provides search orientation, so a changed summary can change retrieval results.

Choose the source and history databases separately:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent --database data/local_voice_agent.db `
    evaluate evals/poetry_session_2.json `
    --session-id 2 --agent-mode final --model qwen3.5:4b `
    --evaluation-database data/local_voice_agent_evaluations.db `
    --limit 1 --timeout 600 --label "single-case check"
```

Evaluation history defaults to `data/local_voice_agent_evaluations.db`, separate from the
application database. Each invocation appends a new run and writes each case as it finishes.
Completed history is never replaced. Runs retain dataset/configuration snapshots and fingerprints,
the selected transcript and summary context, model and agent mode, answers, errors, timestamps,
latency, successful tool calls, retrieved rows, citations, and per-case metrics. References and
required points remain available alongside generated answers for human review. Runtime databases
remain ignored by Git; the dataset and runner are version controlled.

Timestamps are the scoring key. Gold segment IDs are audit references, checked against the selected
session before any model request. Gold boundaries from the supplied transcript use whole seconds;
preflight accepts matching stored segment boundaries truncated to those seconds. Missing IDs,
incomplete segment lists, and mismatched ranges are rejected. When importing the same transcript
into a database with different auto-increment IDs, add `--remap-segment-ids` to resolve and validate
the evidence by timestamps; the original and resolved IDs are retained for audit. Remapping checks
each gold range's outer timestamp boundaries and segment count. Use it for copies of the same
transcript; these checks cannot establish that the words or every internal boundary are identical.

The compact report uses these deterministic metrics:

| Metric | Definition |
| --- | --- |
| Execution success | Completed cases divided by attempted cases; also prints successful and failed counts. |
| Answerability accuracy | Correct sufficient/insufficient decisions divided by attempted cases. A correct decision has `evidence_insufficient == (not answerable)`. Failed cases with unknown state count as incorrect in the aggregate; their per-case value is null. |
| Retrieval evidence recall | Fraction of gold ranges overlapped by at least one transcript row actually exposed through search or range retrieval. Summary text does not count. |
| Citation recall | Fraction of gold ranges overlapped by at least one timestamp citation in the accepted answer. |
| Citation precision | Fraction of unique answer citation ranges overlapping any gold range. No citations means null; citations on an unanswerable case score zero. |
| Processing time | Per-case elapsed seconds and their mean, including failed attempts, plus total run duration. |

Recall is null for unanswerable cases. Aggregate recalls and precision are means over applicable
cases, excluding null values; stored denominator counts make this explicit. Failed answerable
cases retain available retrieval telemetry and receive zero citation recall when no accepted answer
exists. Finite, positive-duration intervals overlap only when
`start_a < end_b and end_a > start_b`; touching endpoints do not overlap, and scoring adds no
tolerance. Final mode uses application-rendered citations from
validated segment IDs. Manual mode reports the accepted answer's timestamp ranges and maps them
back to transcript rows. These metrics measure evidence coverage and evidence-state decisions;
even perfect overlap does not establish that an answer is correct or its full claim is supported.

Retrieval remains lexical with the existing deterministic expansion. The evaluation records the
implementation fingerprint so later hybrid-retrieval runs can be compared without changing the
gold dataset. Model execution still depends on the installed Ollama model and local hardware;
the deterministic part is validation and scoring, not a promise of bit-identical model output.

Run the offline unit tests and Python compile checks (tests use fake providers, never Ollama):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src tests
```

### Configuration

The defaults can be overridden with environment variables:

| Variable | Default |
| --- | --- |
| `LVA_APP_LANGUAGE` | `en` |
| `LVA_TRANSCRIPTION_LANGUAGE` | `en` |
| `LVA_DATABASE_PATH` | `data/local_voice_agent.db` |
| `LVA_WHISPER_MODEL` | `small.en` |
| `LVA_WHISPER_DEVICE` | `cpu` |
| `LVA_WHISPER_COMPUTE_TYPE` | `int8` |
| `LVA_WHISPER_BEAM_SIZE` | `5` |
| `LVA_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `LVA_OLLAMA_MODEL` | `qwen3.5:4b` |
| `LVA_OLLAMA_TIMEOUT_SECONDS` | `600` |
| `LVA_CHECKPOINT_TARGET_SECONDS` | `600` |
| `LVA_MINIMUM_FINAL_CHUNK_SECONDS` | `120` |

The MVP intentionally supports only English even though the language is configurable for future development.

## Project structure

```text
src/local_voice_agent/
├── cli.py                  command-line interface
├── agent.py                manual baseline and forced FinalAnswer workflow
├── config.py               environment-backed settings
├── models.py               validated application models
├── evaluation/             dataset validation, deterministic metrics, runner, and history database
├── service.py              audio processing use case
├── llm/
│   ├── base.py             provider protocol and structured response types
│   └── ollama_provider.py  local Ollama HTTP adapter
├── summaries/
│   ├── chunking.py         timestamp-aware transcript grouping
│   ├── models.py           checkpoint and final-summary schemas
│   ├── prompts.py          evidence-constrained English prompts
│   └── summarizer.py       hierarchical summary orchestration
├── storage/database.py     SQLite boundary and schema
└── transcription/
    ├── base.py             provider protocol
    └── faster_whisper.py   faster-whisper implementation
```

## Delivery approach

The MVP will be built as small end-to-end slices:

1. Verify transcription, structured output, and one tool call independently.
2. Process one audio file and persist its timestamped transcript.
3. Generate checkpoints and a final summary.
4. Ask questions through transcript retrieval and a manual agent loop.
5. Add debug output, notes, failure handling, focused tests, and documentation.

## Test material

Use an English recording that you created yourself or that is clearly licensed for reuse, such as public-domain or Creative Commons material. A short recording is preferable for the first technical checks.

The first capability check was completed with OpenAI Whisper's small `tests/jfk.flac` fixture and the `tiny.en` model. The downloaded audio and runtime database live under ignored `data/` and are not committed.

An earlier summary slice was verified with a 15-minute English spoken article. Current summaries
deliberately use a smaller schema focused on useful notes; detailed claims, concepts, terminology
checks, decisions, action items, and evidence linking are deferred to later development.
