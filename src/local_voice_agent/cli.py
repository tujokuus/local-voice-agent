from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from local_voice_agent.agent import (
    MAX_AGENT_STEPS,
    ForcedFinalAnswerAgent,
    SessionQuestionAgent,
)
from local_voice_agent.config import Settings
from local_voice_agent.evaluation.cli import add_evaluation_parser, evaluate
from local_voice_agent.llm import OllamaProvider
from local_voice_agent.models import SessionStatus
from local_voice_agent.retrieval import (
    MAX_SEARCH_RESULTS,
    get_transcript_range,
    parse_time_value,
    search_transcript,
)
from local_voice_agent.service import SessionProcessor
from local_voice_agent.storage import Database
from local_voice_agent.summaries import SessionSummarizer
from local_voice_agent.transcription import FasterWhisperTranscriber


def _configure_stdio() -> None:
    """Keep transcript output Unicode-safe on Windows and redirected streams."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _database(settings: Settings, override: Path | None) -> Database:
    return Database((override or settings.database_path).expanduser().resolve())


def _build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-voice-agent",
        description="Process English audio locally and inspect timestamped transcripts.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help=f"SQLite path (default: {settings.database_path})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Transcribe and store an audio file")
    process_parser.add_argument("audio_file", type=Path)
    process_parser.add_argument("--model", default=settings.whisper_model)
    process_parser.add_argument("--device", default=settings.whisper_device)
    process_parser.add_argument("--compute-type", default=settings.whisper_compute_type)
    process_parser.add_argument("--beam-size", type=int, default=settings.whisper_beam_size)

    show_parser = subparsers.add_parser("show", help="Show one stored session transcript")
    show_parser.add_argument("session_id", type=int)

    subparsers.add_parser("sessions", help="List stored sessions")

    search_parser = subparsers.add_parser(
        "search", help="Search one stored session transcript"
    )
    search_parser.add_argument("session_id", type=int)
    search_parser.add_argument("query", help="Word or quoted phrase to find")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=f"Maximum results, between 1 and {MAX_SEARCH_RESULTS} (default: 5)",
    )

    range_parser = subparsers.add_parser(
        "transcript-range", help="Show transcript text from a bounded time range"
    )
    range_parser.add_argument("session_id", type=int)
    range_parser.add_argument("start", help="Start as seconds, MM:SS, or HH:MM:SS")
    range_parser.add_argument("end", help="End as seconds, MM:SS, or HH:MM:SS")

    notes_parser = subparsers.add_parser(
        "notes", help="List notes stored in successful summary runs"
    )
    notes_parser.add_argument(
        "--session-id", type=int, help="Limit output to one session"
    )
    notes_parser.add_argument(
        "--run-id", type=int, help="Limit output to one run; requires --session-id"
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Answer a question using bounded read-only transcript tools"
    )
    ask_parser.add_argument("session_id", type=int)
    ask_parser.add_argument("question", help="Question about the selected recording")
    ask_parser.add_argument("--model", default=settings.ollama_model)
    ask_parser.add_argument("--ollama-url", default=settings.ollama_base_url)
    ask_parser.add_argument(
        "--timeout", type=float, default=settings.ollama_timeout_seconds
    )
    ask_parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help=f"Maximum agent steps, between 1 and {MAX_AGENT_STEPS} (default: 5)",
    )
    ask_parser.add_argument(
        "--run-id", type=int, help="Use a specific stored summary run"
    )
    ask_parser.add_argument(
        "--agent-mode",
        choices=("final", "manual"),
        default="final",
        help=(
            "Use forced segment-grounded FinalAnswer or the original manual agent "
            "loop (default: final)"
        ),
    )
    ask_parser.add_argument(
        "--debug", action="store_true", help="Print the agent's selected actions"
    )

    add_evaluation_parser(subparsers, settings)

    summarize_parser = subparsers.add_parser(
        "summarize", help="Create checkpoint notes and a final summary"
    )
    summarize_parser.add_argument("session_id", type=int)
    summarize_parser.add_argument("--model", default=settings.ollama_model)
    summarize_parser.add_argument("--ollama-url", default=settings.ollama_base_url)
    summarize_parser.add_argument(
        "--timeout", type=float, default=settings.ollama_timeout_seconds
    )
    summarize_parser.add_argument(
        "--chunk-seconds", type=float, default=settings.checkpoint_target_seconds
    )
    summarize_parser.add_argument(
        "--label", help="Optional name for comparing this summary run"
    )
    summarize_parser.add_argument(
        "--content-mode",
        choices=("auto", "informational", "meeting"),
        default="auto",
        help="Constrain decisions and action items for the source type",
    )

    summary_parser = subparsers.add_parser(
        "summary", help="Show stored checkpoints and final summary"
    )
    summary_parser.add_argument("session_id", type=int)
    summary_parser.add_argument(
        "--run-id", type=int, help="Show a specific summary run instead of the latest"
    )

    summary_runs_parser = subparsers.add_parser(
        "summary-runs", help="List stored summary runs for one session"
    )
    summary_runs_parser.add_argument("session_id", type=int)

    subparsers.add_parser("doctor", help="Check local runtime prerequisites")
    return parser


def _process(args: argparse.Namespace, settings: Settings) -> int:
    if args.beam_size < 1:
        raise ValueError("--beam-size must be at least 1")

    database = _database(settings, args.database)
    transcriber = FasterWhisperTranscriber(
        model_name=args.model,
        language=settings.transcription_language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
    )
    processor = SessionProcessor(
        database=database,
        transcriber=transcriber,
        language=settings.transcription_language,
        transcription_model=args.model,
    )

    print(f"Loading audio: {args.audio_file}")
    print(
        "Transcribing locally "
        f"(model={args.model}, device={args.device}, compute_type={args.compute_type})..."
    )
    session_id = processor.process(args.audio_file)
    session = database.get_session(session_id)
    segment_count = len(database.get_transcript(session_id))

    print(f"Stored {segment_count} timestamped transcript segments.")
    if session and session.duration_seconds is not None:
        print(f"Audio duration: {_format_timestamp(session.duration_seconds)}")
    if session and session.processing_duration_seconds is not None:
        print(
            "Processing time: "
            f"{_format_timestamp(session.processing_duration_seconds)}"
        )
    print(f"Session ID: {session_id}")
    return 0


def _show(args: argparse.Namespace, settings: Settings) -> int:
    database = _database(settings, args.database)
    database.initialize()
    session = database.get_session(args.session_id)
    if session is None:
        print(f"Session {args.session_id} was not found.", file=sys.stderr)
        return 1

    print(f"Session {session.id}")
    print(f"Status: {session.status.value}")
    print(f"Audio: {session.source_audio_path}")
    print(f"Language: {session.language}")
    print(f"Transcription model: {session.transcription_model}")
    if session.duration_seconds is not None:
        print(f"Audio duration: {_format_timestamp(session.duration_seconds)}")
    if session.processing_duration_seconds is not None:
        print(
            "Processing time: "
            f"{_format_timestamp(session.processing_duration_seconds)}"
        )
    if session.error_message:
        print(f"Error: {session.error_message}")

    segments = database.get_transcript(session.id)
    if not segments:
        print("\nNo transcript segments stored.")
        return 0 if session.status is not SessionStatus.FAILED else 1

    print("\nTranscript")
    for segment in segments:
        start = _format_timestamp(segment.start_seconds)
        end = _format_timestamp(segment.end_seconds)
        print(f"[{start} - {end}] {segment.text}")
    return 0


def _sessions(args: argparse.Namespace, settings: Settings) -> int:
    database = _database(settings, args.database)
    database.initialize()
    sessions = database.list_sessions()
    if not sessions:
        print("No sessions stored.")
        return 0

    for session in sessions:
        audio_duration = (
            _format_timestamp(session.duration_seconds)
            if session.duration_seconds is not None
            else "--:--:--"
        )
        processing_duration = (
            _format_timestamp(session.processing_duration_seconds)
            if session.processing_duration_seconds is not None
            else "--:--:--"
        )
        print(
            f"{session.id:>4}  {session.status.value:<10}  "
            f"audio={audio_duration}  processing={processing_duration}  "
            f"{session.source_audio_path.name}"
        )
    return 0


def _search(args: argparse.Namespace, settings: Settings) -> int:
    database = _database(settings, args.database)
    database.initialize()
    session = database.get_session(args.session_id)
    if session is None:
        raise LookupError(f"Session {args.session_id} was not found")

    results = search_transcript(
        database.get_transcript(session.id),
        args.query,
        limit=args.limit,
    )
    print(f'Search results for "{args.query.strip()}" in session {session.id}')
    if not results:
        print("No transcript matches found.")
        return 0

    for index, result in enumerate(results, start=1):
        start = _format_timestamp(result.start_seconds)
        end = _format_timestamp(result.end_seconds)
        terms = ", ".join(result.matched_terms)
        print(
            f"\n{index}. [{start} - {end}] segment_id={result.segment_id} "
            f"matched={terms}"
        )
        print(result.text)
    return 0


def _transcript_range(args: argparse.Namespace, settings: Settings) -> int:
    database = _database(settings, args.database)
    database.initialize()
    session = database.get_session(args.session_id)
    if session is None:
        raise LookupError(f"Session {args.session_id} was not found")

    start_seconds = parse_time_value(args.start)
    end_seconds = parse_time_value(args.end)
    segments = get_transcript_range(
        database.get_transcript(session.id),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    start = _format_timestamp(start_seconds)
    end = _format_timestamp(end_seconds)
    print(f"Session {session.id} transcript range [{start} - {end}]")
    if not segments:
        print("No transcript segments overlap this range.")
        return 0

    for segment in segments:
        segment_start = _format_timestamp(segment.start_seconds)
        segment_end = _format_timestamp(segment.end_seconds)
        print(
            f"[{segment_start} - {segment_end}] "
            f"segment_id={segment.id} {segment.text}"
        )
    return 0


def _notes(args: argparse.Namespace, settings: Settings) -> int:
    if args.run_id is not None and args.session_id is None:
        raise ValueError("--run-id requires --session-id")

    database = _database(settings, args.database)
    database.initialize()
    if args.session_id is not None:
        session = database.get_session(args.session_id)
        if session is None:
            raise LookupError(f"Session {args.session_id} was not found")
        sessions = [session]
    else:
        sessions = database.list_sessions()

    printed_runs = 0
    for session in sessions:
        runs = database.list_summary_runs(session.id)
        if args.run_id is not None:
            runs = [run for run in runs if run.id == args.run_id]
            if not runs:
                raise LookupError(
                    f"Summary run {args.run_id} was not found for session {session.id}"
                )

        for run in runs:
            summary = database.get_final_summary(session.id, run.id)
            if summary is None:
                continue
            checkpoints = database.get_summary_checkpoints(session.id, run.id)
            checkpoint_notes = [
                (checkpoint, note)
                for checkpoint in checkpoints
                for note in checkpoint.notes
            ]
            if not summary.important_notes and not checkpoint_notes:
                continue

            printed_runs += 1
            label = f"; label={run.label}" if run.label else ""
            print(f"\nSession {session.id}, summary run {run.id}{label}")
            _print_string_list("Important notes", summary.important_notes)
            print("\nCheckpoint notes")
            for checkpoint, note in checkpoint_notes:
                start = _format_timestamp(checkpoint.start_seconds)
                end = _format_timestamp(checkpoint.end_seconds)
                print(f"- [{start} - {end}] {note}")

    if printed_runs == 0:
        print("No stored notes found.")
    return 0


def _ask(args: argparse.Namespace, settings: Settings) -> int:
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")

    database = _database(settings, args.database)
    database.initialize()
    session = database.get_session(args.session_id)
    if session is None:
        raise LookupError(f"Session {args.session_id} was not found")

    transcript = database.get_transcript(session.id)
    summary = database.get_final_summary(session.id, args.run_id)
    if args.run_id is not None and summary is None:
        raise LookupError(
            f"Summary run {args.run_id} was not found for session {session.id}"
        )
    checkpoints = (
        database.get_summary_checkpoints(session.id, summary.id)
        if summary is not None
        else []
    )
    provider = OllamaProvider(
        model_name=args.model,
        base_url=args.ollama_url,
        timeout_seconds=args.timeout,
    )
    agent_class = (
        ForcedFinalAnswerAgent
        if args.agent_mode == "final"
        else SessionQuestionAgent
    )
    agent = agent_class(
        provider=provider,
        transcript=transcript,
        summary=summary,
        checkpoints=checkpoints,
        max_steps=args.max_steps,
    )

    print(
        f"Answering from session {session.id} with {args.model} "
        f"(agent_mode={args.agent_mode})..."
    )
    started_at = perf_counter()
    result = agent.answer(
        args.question,
        trace=(lambda message: print(f"[agent] {message}")) if args.debug else None,
    )
    elapsed = perf_counter() - started_at
    print("\nAnswer")
    print(result.answer)
    print(f"\nAgent steps: {result.step_count}")
    print(f"Tools used: {', '.join(result.tool_calls) or 'none'}")
    print(f"Processing time: {_format_timestamp(elapsed)}")
    return 0


def _summarize(args: argparse.Namespace, settings: Settings) -> int:
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be greater than zero")

    database = _database(settings, args.database)
    database.initialize()
    session = database.get_session(args.session_id)
    if session is None:
        raise LookupError(f"Session {args.session_id} was not found")
    if session.status is not SessionStatus.COMPLETED:
        raise ValueError(
            f"Session {session.id} has status {session.status.value} and cannot be summarized"
        )

    transcript = database.get_transcript(session.id)
    if not transcript:
        raise ValueError(f"Session {session.id} has no transcript segments")

    provider = OllamaProvider(
        model_name=args.model,
        base_url=args.ollama_url,
        timeout_seconds=args.timeout,
    )
    summarizer = SessionSummarizer(
        provider=provider,
        target_chunk_seconds=args.chunk_seconds,
        minimum_final_chunk_seconds=min(
            settings.minimum_final_chunk_seconds,
            args.chunk_seconds / 2,
        ),
    )

    print(f"Summarizing session {session.id} with {args.model}...")

    def report_progress(current: int, total: int) -> None:
        print(f"Creating checkpoint {current}/{total}...")

    bundle = summarizer.summarize(
        transcript,
        content_mode=args.content_mode,
        progress=report_progress,
        final_progress=lambda: print("Creating final session summary..."),
    )
    summary_run_id = database.store_summary_run(
        session_id=session.id,
        model_name=args.model,
        chunk_seconds=args.chunk_seconds,
        content_mode=args.content_mode,
        label=args.label,
        bundle=bundle,
    )

    print(f"Summary run ID: {summary_run_id}")
    print(f"Stored {len(bundle.checkpoints)} checkpoint summaries.")
    print(f"Summary processing time: {_format_timestamp(bundle.processing_seconds)}")
    print(
        "Final summary generation time: "
        f"{_format_timestamp(bundle.final_generation_seconds)}"
    )
    print(f"Retry attempts: {bundle.retry_count}")
    print(
        "Retry processing time: "
        f"{_format_timestamp(bundle.retry_processing_seconds)}"
    )
    print("\nOverall summary")
    print(bundle.final_summary.overall_summary)
    _print_string_list("Important notes", bundle.final_summary.important_notes)
    _print_string_list("Main topics", bundle.final_summary.main_topics)
    return 0


def _summary(args: argparse.Namespace, settings: Settings) -> int:
    database = _database(settings, args.database)
    database.initialize()
    session = database.get_session(args.session_id)
    if session is None:
        raise LookupError(f"Session {args.session_id} was not found")

    final_summary = database.get_final_summary(session.id, args.run_id)
    if final_summary is None:
        requested = f" run {args.run_id}" if args.run_id is not None else ""
        print(
            f"Session {session.id} has no stored summary{requested}. "
            f"Run 'local-voice-agent summarize {session.id}' first."
        )
        return 1

    checkpoints = database.get_summary_checkpoints(session.id, final_summary.id)
    print(f"Session {session.id} summary")
    print(f"Summary run ID: {final_summary.id}")
    if final_summary.label:
        print(f"Label: {final_summary.label}")
    print(f"Model: {final_summary.model_name}")
    if final_summary.chunk_seconds is not None:
        print(f"Target chunk size: {_format_timestamp(final_summary.chunk_seconds)}")
    print(f"Content mode: {final_summary.content_mode}")
    print(f"Checkpoints: {final_summary.checkpoint_count}")
    if final_summary.processing_seconds is not None:
        print(
            "Summary processing time: "
            f"{_format_timestamp(final_summary.processing_seconds)}"
        )
    if final_summary.final_generation_seconds is not None:
        print(
            "Final summary generation time: "
            f"{_format_timestamp(final_summary.final_generation_seconds)}"
        )

    all_attempts = [
        checkpoint.attempt_seconds for checkpoint in checkpoints
    ] + [final_summary.final_attempt_seconds]
    retry_count = sum(max(0, len(attempts) - 1) for attempts in all_attempts)
    retry_seconds = sum(sum(attempts[1:]) for attempts in all_attempts)
    print(f"Retry attempts: {retry_count}")
    print(f"Retry processing time: {_format_timestamp(retry_seconds)}")
    _print_attempt_times("Final summary attempts", final_summary.final_attempt_seconds)

    print("\nOverall summary")
    print(final_summary.overall_summary)
    _print_string_list("Important notes", final_summary.important_notes)
    _print_string_list("Main topics", final_summary.main_topics)

    print("\nCheckpoints")
    for checkpoint in checkpoints:
        start = _format_timestamp(checkpoint.start_seconds)
        end = _format_timestamp(checkpoint.end_seconds)
        duration = (
            f"; generated in {_format_timestamp(checkpoint.generation_seconds)}"
            if checkpoint.generation_seconds is not None
            else ""
        )
        print(f"\n{checkpoint.chunk_index + 1}. [{start} - {end}]{duration}")
        _print_attempt_times("Generation attempts", checkpoint.attempt_seconds)
        print(checkpoint.summary)
        _print_string_list("Notes", checkpoint.notes)
    return 0


def _summary_runs(args: argparse.Namespace, settings: Settings) -> int:
    database = _database(settings, args.database)
    database.initialize()
    session = database.get_session(args.session_id)
    if session is None:
        raise LookupError(f"Session {args.session_id} was not found")

    runs = database.list_summary_runs(session.id)
    if not runs:
        print(f"Session {session.id} has no stored summary runs.")
        return 0

    print(f"Session {session.id} summary runs")
    for run in runs:
        processing = (
            _format_timestamp(run.processing_seconds)
            if run.processing_seconds is not None
            else "--:--:--"
        )
        chunk = (
            _format_timestamp(run.chunk_seconds)
            if run.chunk_seconds is not None
            else "unknown"
        )
        label = f"  label={run.label}" if run.label else ""
        print(
            f"{run.id:>4}  model={run.model_name:<14}  chunks={run.checkpoint_count:<3} "
            f"target={chunk}  mode={run.content_mode:<13} "
            f"processing={processing}{label}"
        )
    return 0


def _print_string_list(title: str, items: list[str]) -> None:
    print(f"\n{title}")
    if not items:
        print("- None")
        return
    for item in items:
        print(f"- {item}")


def _print_attempt_times(title: str, attempts: tuple[float, ...]) -> None:
    if not attempts:
        return
    print(f"{title}:")
    for index, seconds in enumerate(attempts):
        label = "Attempt 1" if index == 0 else f"Retry {index}"
        print(f"- {label}: {_format_timestamp(seconds)}")


def _doctor(_: argparse.Namespace, settings: Settings) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python >= 3.12", sys.version_info >= (3, 12), sys.version.split()[0]))
    checks.append(
        (
            "pydantic installed",
            importlib.util.find_spec("pydantic") is not None,
            "required for application models",
        )
    )
    checks.append(
        (
            "faster-whisper installed",
            importlib.util.find_spec("faster_whisper") is not None,
            "required for transcription",
        )
    )

    ollama_path = shutil.which("ollama")
    ollama_detail = "not found"
    if ollama_path:
        result = subprocess.run(
            [ollama_path, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        ollama_detail = (result.stdout or result.stderr).strip()
    checks.append(("Ollama available", ollama_path is not None, ollama_detail))

    for label, passed, detail in checks:
        marker = "OK" if passed else "MISSING"
        print(f"[{marker:<7}] {label}: {detail}")

    print("\nEffective transcription settings")
    print(f"Language: {settings.transcription_language}")
    print(f"Model: {settings.whisper_model}")
    print(f"Device: {settings.whisper_device}")
    print(f"Compute type: {settings.whisper_compute_type}")
    print(f"Ollama URL: {settings.ollama_base_url}")
    print(f"Ollama model: {settings.ollama_model}")
    return 0 if all(passed for _, passed, _ in checks) else 1


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    try:
        settings = Settings.from_environment()
        parser = _build_parser(settings)
        args = parser.parse_args(argv)

        handlers = {
            "process": _process,
            "show": _show,
            "sessions": _sessions,
            "search": _search,
            "transcript-range": _transcript_range,
            "notes": _notes,
            "ask": _ask,
            "evaluate": evaluate,
            "summarize": _summarize,
            "summary": _summary,
            "summary-runs": _summary_runs,
            "doctor": _doctor,
        }
        return handlers[args.command](args, settings)
    except (FileNotFoundError, LookupError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
