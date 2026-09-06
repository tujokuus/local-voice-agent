# Local Voice Agent MVP Plan

## Current progress

The first two end-to-end slices are implemented and manually verified:

```text
English audio file
    → faster-whisper
    → validated timestamped segments
    → SQLite session and transcript storage
    → CLI session listing and transcript display
    → approximately ten-minute sentence-aware chunks
    → local Ollama checkpoint summaries and notes
    → bounded structured-output correction retry
    → overall summary, important notes, and main topics
    → append-only SQLite summary-run and attempt-timing storage
    → CLI display
    → bounded relevance-ranked transcript search
    → bounded transcript range retrieval
    → bounded read-only question-answering agent loop
```

The next slice hardens retrieval quality and evaluates the agent with real questions.

## Goal

Build a modular desktop-oriented Python MVP that processes an English audio file locally, creates a timestamped transcript and hierarchical summary, stores the results in SQLite, and answers questions through a manually implemented LLM agent.

The project is intentionally educational. The LLM provider, tool registry, tool-call validation, agent loop, conversation state, structured output handling, and context management should remain visible in the code instead of being hidden behind an agent framework.

## Language scope

The MVP is English-only. Audio, transcription, prompts, summaries, tool-assisted answers, and practical CLI output will use English. Configuration should still expose language settings so future multilingual support does not require invasive changes.

Language detection, translation, Finnish support, and multiple prompt sets are excluded.

## Core capabilities

1. Accept a local WAV, MP3, or other supported audio file.
2. Transcribe English speech locally with `faster-whisper`.
3. Store timestamped transcript segments.
4. Group segments into approximately ten-minute chunks, preferring a nearby sentence ending
   instead of cutting solely at an exact time.
5. Generate a Pydantic-validated checkpoint for each chunk.
6. Generate an overall summary, important notes, and main topics from the checkpoints.
7. Persist sessions, transcript segments, versioned summary runs, checkpoints, and later notes
   in SQLite.
8. Let a user ask questions about one selected session.
9. Let the local model select from explicitly registered and validated tools.
10. Keep retrieved evidence bounded instead of sending the complete transcript for every question.

## Recommended implementation slices

### Slice 0: capability checks

- Transcribe one short English audio sample and inspect its timestamps.
- Ask the configured Ollama model for one schema-constrained response.
- Verify one tool call and one tool result round trip.
- Record the hardware, transcription model, LLM model, and observed latency.

These checks determine whether the initially proposed model is sufficiently reliable before application architecture depends on it.

### Slice 1: audio to stored transcript

- Add project configuration and the smallest useful CLI.
- Create a session record.
- Transcribe one audio file.
- Persist timestamped segments in SQLite.
- Provide a way to inspect the stored transcript.

Acceptance criterion: one command processes a short audio file and the stored transcript can be read back with correct timestamps.

### Slice 2: transcript to summaries

- Group timestamped segments into approximate ten-minute sentence-aware chunks.
- Generate a short summary and important-note list for every checkpoint.
- Generate the overall summary, important notes, and main topics from checkpoints.
- Store source time ranges, model metadata, and outputs.
- Reject structurally invalid and empty outputs.
- Record checkpoint, retry, final-summary, and total processing times.
- Keep successful summary runs side by side with model, chunk-size, and optional label metadata.

Acceptance criterion: a processed session has validated checkpoints and a final summary that can be read from SQLite.

### Slice 3: evidence-based session chat

- Implement bounded transcript search with word-form and synonym expansion. Completed.
- Implement transcript range retrieval. Completed for direct CLI use.
- Expose the session summary and use relevant excerpts as search-planning context. Completed.
- Implement a small tool registry and argument validation. Completed for `ask`.
- Implement the manual agent loop with a maximum step count. Completed for `ask`.
- Keep the selected session ID in trusted application context rather than accepting it from the model.
- Allow a different second search when the first inspected passage is insufficient. Completed.
- Keep rejected-answer repairs outside the valid-step budget and show rejection reasons in debug
  output. Completed.

Initial read-only tools:

- `search_transcript(query)`
- `get_transcript(segment_id)`
- `get_session_summary()`

Acceptance criterion: the CLI agent answers a question using retrieved transcript evidence and includes relevant timestamps when available.

The agent's transcript-range tool is tied to a segment returned by its preceding search. Application
code expands that segment by 30 seconds on both sides, preventing the model from requesting an
arbitrary full-recording range. The original question is retained in every tool-result turn. The
stored summary can guide search vocabulary, but transcript evidence remains mandatory for a
supported answer.

### Slice 4: hardening and learning features

- Add agent debug output for steps, tool calls, validated arguments, and bounded tool results.
- Add clear failure messages and limited structured-output retry behavior.
- Add `save_note` with explicit user intent or confirmation because it changes stored state.
- Add focused automated tests and a real-model smoke-test guide.

## Summary scope

Each checkpoint currently contains only a short summary and a list of useful notes. The final stage
combines those into an overall summary, important notes, and main topics. Detailed claims, concepts,
terminology verification, decisions, action items, and evidence linking are deliberately deferred
until the simpler notes pipeline is reliable.

Every checkpoint retains its source start and end timestamps. The final summary is produced from
checkpoints rather than reprocessing the full transcript.

## Storage guidance

Keep storage logic separate from transcription, summarization, provider, and agent logic. Avoid a large repository abstraction until repeated storage operations make one useful.

Useful session metadata includes:

- processing status and timestamps
- source audio path and duration
- configured language
- transcription model
- LLM model
- error information when processing fails

Search results should contain timestamps, text, and stable transcript segment identifiers. Results must have a configured maximum count so tool output cannot grow without bounds.

## Provider boundaries

Application code should depend on small transcription and chat interfaces rather than directly on `faster-whisper` or Ollama. These boundaries prepare the project for a later move to `whisper.cpp`, `llama.cpp`, or another runtime without requiring a framework.

The abstractions should remain lightweight. One interface and one implementation are enough for the MVP.

## Agent safety

The model must not receive unrestricted filesystem, database, shell, or Python access. It may interact with the application only through registered tools whose arguments are validated before execution.

The agent loop must have a fixed maximum number of steps. Unknown tools, invalid arguments, oversized time ranges, and tool failures must become controlled tool results or application errors rather than arbitrary execution.

## Testing approach

The first capability checks may be verified manually. Broad test coverage is not required before the first working slice.

A small automated safety net should be added as the relevant logic appears:

- chunk boundaries and timestamp handling
- rejection of invalid structured output
- unknown tools and invalid arguments
- maximum agent-step behavior
- one SQLite persistence round trip

Whisper and Ollama should also be exercised through a documented smoke test using a short real audio sample. Unit tests should not attempt to test those external projects internally.

## Explicit exclusions

- multilingual support, Finnish, language detection, and translation
- speaker diarization and speaker identification
- realtime recording or transcription
- desktop GUI and mobile applications
- cloud services and phone synchronization
- embeddings, vector databases, and cross-session RAG
- advanced action-item tracking and cross-meeting analysis

These remain future roadmap items and should not influence the first implementation beyond avoiding unnecessary coupling.
