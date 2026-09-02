from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from local_voice_agent.config import Settings
from local_voice_agent.llm import OllamaProvider
from local_voice_agent.models import SessionStatus
from local_voice_agent.service import SessionProcessor
from local_voice_agent.storage import Database
from local_voice_agent.summaries import (
    EvidenceItem,
    KeyConcept,
    SessionSummarizer,
    StoredFinalSummary,
    TermToVerify,
)
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

    summarize_parser = subparsers.add_parser(
        "summarize", help="Create structured checkpoints and a final summary"
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
        progress=report_progress,
        final_progress=lambda: print("Creating final session summary..."),
    )
    summary_run_id = database.store_summary_run(
        session_id=session.id,
        model_name=args.model,
        chunk_seconds=args.chunk_seconds,
        label=args.label,
        bundle=bundle,
    )

    print(f"Summary run ID: {summary_run_id}")
    print(f"Stored {len(bundle.checkpoints)} structured checkpoints.")
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
    _print_string_list("Main topics", final_summary.main_topics)
    _print_evidence_items("Key claims", final_summary.key_claims)
    _print_key_concepts("Key concepts", final_summary.key_concepts)
    _print_evidence_items(
        "Uncertainties and debates", final_summary.uncertainties_and_debates
    )
    _print_terms_to_verify("Terms to verify", final_summary.terms_to_verify)
    _print_string_list("Decisions", final_summary.decisions)
    _print_action_items(final_summary)
    _print_string_list("Open questions", final_summary.open_questions)

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
        _print_evidence_items("Key claims", checkpoint.key_claims)
        _print_key_concepts("Key concepts", checkpoint.key_concepts)
        _print_evidence_items(
            "Uncertainties and debates", checkpoint.uncertainties_and_debates
        )
        _print_terms_to_verify("Terms to verify", checkpoint.terms_to_verify)
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
            f"target={chunk}  processing={processing}{label}"
        )
    return 0


def _print_string_list(title: str, items: list[str]) -> None:
    print(f"\n{title}")
    if not items:
        print("- None")
        return
    for item in items:
        print(f"- {item}")


def _print_evidence_items(title: str, items: list[EvidenceItem]) -> None:
    print(f"\n{title}")
    if not items:
        print("- None")
        return
    for item in items:
        start = _format_timestamp(item.start_seconds)
        end = _format_timestamp(item.end_seconds)
        print(f"- [{start} - {end}] {item.text}")


def _print_key_concepts(title: str, concepts: list[KeyConcept]) -> None:
    print(f"\n{title}")
    if not concepts:
        print("- None")
        return
    for concept in concepts:
        start = _format_timestamp(concept.start_seconds)
        end = _format_timestamp(concept.end_seconds)
        print(f"- [{start} - {end}] {concept.term}: {concept.explanation}")


def _print_terms_to_verify(title: str, terms: list[TermToVerify]) -> None:
    print(f"\n{title}")
    if not terms:
        print("- None")
        return
    for term in terms:
        start = _format_timestamp(term.start_seconds)
        end = _format_timestamp(term.end_seconds)
        suggestion = (
            f" -> suggested: {term.suggested_form}" if term.suggested_form else ""
        )
        print(
            f"- [{start} - {end}] {term.transcript_form}{suggestion}; "
            f"reason: {term.reason}"
        )


def _print_attempt_times(title: str, attempts: tuple[float, ...]) -> None:
    if not attempts:
        return
    print(f"{title}:")
    for index, seconds in enumerate(attempts):
        label = "Attempt 1" if index == 0 else f"Retry {index}"
        print(f"- {label}: {_format_timestamp(seconds)}")


def _print_action_items(summary: StoredFinalSummary) -> None:
    print("\nAction items")
    if not summary.action_items:
        print("- None")
        return
    for item in summary.action_items:
        owner = f" (owner: {item.owner})" if item.owner else ""
        print(f"- {item.task}{owner}")


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
