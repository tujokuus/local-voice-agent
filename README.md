# Local Voice Agent

Local Voice Agent is a learning-focused, local-first Python application for turning English audio recordings into timestamped transcripts, structured meeting or lecture summaries, and evidence-based answers.

> Status: planning only. No application code has been implemented yet.

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

