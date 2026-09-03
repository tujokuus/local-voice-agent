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

Transcript search and the manual tool-calling agent loop are the next implementation slices.

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

The default database is `data/local_voice_agent.db`. Put `--database PATH` before the subcommand to use another database.

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
├── config.py               environment-backed settings
├── models.py               validated application models
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

See [docs/MVP_PLAN.md](docs/MVP_PLAN.md) for the detailed plan.

## Test material

Use an English recording that you created yourself or that is clearly licensed for reuse, such as public-domain or Creative Commons material. A short recording is preferable for the first technical checks.

The first capability check was completed with OpenAI Whisper's small `tests/jfk.flac` fixture and the `tiny.en` model. The downloaded audio and runtime database live under ignored `data/` and are not committed.

An earlier summary slice was verified with a 15-minute English spoken article. Current summaries
deliberately use a smaller schema focused on useful notes; detailed claims, concepts, terminology
checks, decisions, action items, and evidence linking are deferred to later development.
