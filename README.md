# Local Voice Agent

Local Voice Agent is a learning-focused, local-first Python application for turning English audio recordings into timestamped transcripts, structured meeting or lecture summaries, and evidence-based answers.

> Status: the first end-to-end slice is working. English audio can be transcribed
> locally and stored as timestamped SQLite segments.

## MVP pipeline

```text
English audio file
       ↓
faster-whisper
       ↓
timestamped transcript
       ↓
approximately five-minute chunks
       ↓
local Ollama model
       ↓
structured checkpoints and final summary
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
- List stored sessions.
- Print a stored transcript with readable timestamps.
- Record failed processing attempts without losing the error context.

Checkpoint summaries, the Ollama provider, transcript search, and the manual agent loop are the next implementation slices.

## Setup

Python 3.12 or newer and Ollama are expected to be installed locally. On PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m local_voice_agent doctor
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

The MVP intentionally supports only English even though the language is configurable for future development.

## Project structure

```text
src/local_voice_agent/
├── cli.py                  command-line interface
├── config.py               environment-backed settings
├── models.py               validated application models
├── service.py              audio processing use case
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
