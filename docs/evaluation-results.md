# Initial agent comparison

[Project overview](../README.md) · [Commands and metric definitions](usage.md)

This report summarizes local evaluation runs 1–5 from September 7–9, 2026, and an
AI-assisted review of their saved answers against the transcript. It compares application
configurations, not the general capability of the libraries or models.

## Experiment

- One approximately 15-minute English spoken article about poetry, stored as 145 transcript
  segments in local session 2.
- [Twelve versioned questions](../evals/poetry_session_2.json): eight focused questions, two
  synthesis questions, and two deliberately unanswerable questions.
- Identical dataset, transcript, and summary snapshots across all five runs. Summary run 4
  provided search orientation; each question had a five-step budget and a 600-second timeout
  per model request.
- Shared deterministic lexical search and context retrieval. The custom workflows used
  Ollama's `/api/chat`; PydanticAI 2.41.0 used its OpenAI-compatible interface.
- One run per tested model/mode combination: 60 attempts, 51 accepted answers, and nine
  execution failures. There was no manual + 9b run in this comparison.

The snapshot fingerprints recorded with all five runs were:

```text
Dataset:    0b03b63e64f8c83615c764245c8129c2ccdbf67834bdbaaf6d1c45c2c61f20d7
Transcript: 99033856eaa37eb10e0589f8df8ecad40d788278786c6d7f8f2a665a069f5ce6
Summary:    41e463e1247797ac4b0f02882615d7b3a415712ef087db2a788a7cd86845e604
```

## Measured results

| Run | Mode | Model | Completed | Correct answerability decisions¹ | Mean seconds/attempt² |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | final | qwen3.5:4b | 12/12 | 11/12 | 97 |
| 2 | manual | qwen3.5:4b | 5/12 | 5/12 | 134 |
| 3 | final | qwen3.5:9b | 11/12 | 11/12 | 225 |
| 4 | pydanticai | qwen3.5:4b | 11/12 | 10/12 | 130 |
| 5 | pydanticai | qwen3.5:9b | 12/12 | 12/12 | 211 |

¹ Whether the output's sufficient/insufficient-evidence flag matched the case label.
Unknown states on failures count as incorrect in the aggregate. This is not a correctness
score for the answer's content.

² Mean elapsed question-processing time, including failures, rounded to whole seconds.
Hardware, model warm-up, and background load were not controlled for this report; these
numbers describe the observed runs rather than portable performance expectations.

All five runs scored 100% retrieval evidence recall under the existing overlap metric.
That metric requires only some retrieved text to overlap each gold range, not the complete
passage or every fact needed to answer the question.

## AI-assisted content review

The review read the accepted answers, failure records, and full saved transcript. It assessed
factual faithfulness to that transcript, coverage of the question's required points, appropriate
abstention, and citation support. It did not check the recording's claims against external
historical sources or grade the transcription against the audio.

This was a qualitative AI-assisted review, not an independent human annotation study or a
new automatic metric in the evaluation runner. No numerical semantic-quality score was assigned.

**PydanticAI + 9b was strongest overall in these runs.** It completed every case, covered the
main synthesis questions, and correctly declined to supply both an unnamed person and missing
popularity statistics. Custom final + 4b offered the best observed speed/quality trade-off.
PydanticAI + 4b did not improve on custom final + 4b. The original manual + 4b workflow failed
to deliver an accepted answer for seven questions within its validation and step limits.

### Examples behind the assessment

| Case | Transcript evidence and observed difference |
| --- | --- |
| `unanswerable_named_introducer_of_rhyme` | At 04:33–04:45, the source describes adoption from Arabic but names no individual poet. Both PydanticAI runs recognized the missing identity. Final + 4b substituted “the Arabs” for the requested person. Final + 9b failed because its insufficient-evidence output did not satisfy the workflow's search requirement. |
| `oral_writing_printing_transition` | At 10:19–11:02, the source links variable oral wording, written fixation, absent readers, and printing. Both 9b answers covered this progression. The final + 4b and PydanticAI + 4b answers omitted an explicit explanation of wording varying between performances. |
| `free_verse_and_meter_comparison` | Relevant passages occur at 04:02–04:09 and 05:10–05:34. PydanticAI + 4b incorrectly reported that the required comparison was unavailable. Its retrieval telemetry omitted central segments 50–52, despite scoring 100% overlap recall. Both custom final runs and PydanticAI + 9b supplied the main comparison. |
| `translation_form_and_psalms` | Final + 9b added that imagery, word associations, and musical qualities are lost in translation. This is stronger than the source's explanation at 01:13–01:45. PydanticAI + 9b described those features without adding that categorical claim. |
| `prose_expansiveness_and_logic` | The source explicitly says poetry is not necessarily illogical at 02:18–02:29. Even PydanticAI + 9b omitted this qualification. PydanticAI + 4b also introduced a contrast about definitive interpretations of prose that the cited passage does not establish. |

Citation validation also has limits. In PydanticAI + 9b's rhythm answer, a claim about meter
and listener expectations cites 05:10–05:22, while the source sentence continues to 05:28.
The answer is substantively correct, but the citation does not cover the complete supporting
sentence. A valid segment ID or overlapping timestamp cannot establish full claim support.

## What this supports, and what it does not

The results support keeping custom final + 4b as a fast baseline and using PydanticAI + 9b
as the next candidate for broader quality evaluation. They also show that adopting an agent
framework does not automatically improve results on a smaller model.

There is only one recording and one run per combination. Orchestration, prompts, and model
transport differ, so the experiment cannot isolate the framework as the cause of every
difference. The content review is not blinded or independently rated. Note-generation quality
on lectures and meetings was not evaluated by this QA test set.

Next steps are repeated runs, additional recordings including lectures and meetings, a separate
note-quality assessment, and claim-level checks for omissions, unsupported additions, and
complete citation spans. Hybrid retrieval should be compared against this lexical baseline.

## Inspect or repeat the experiment

In the original local history, inspect an entire run or a specific answer with:

```powershell
.\.venv\Scripts\python.exe -m local_voice_agent evaluation-show 5
.\.venv\Scripts\python.exe -m local_voice_agent evaluation-compare 1 3 4 5 --case-id oral_writing_printing_transition
```

The dataset and evaluator are version controlled. Audio, transcript/summary snapshots, and raw
answer history are in the ignored local databases, so a fresh clone does not include these
records or local run IDs. Repeating this exact experiment requires the matching source data;
re-transcribing audio can change boundaries and segment IDs. See the [usage guide](usage.md)
for dataset validation, remapping, summary selection, matrix commands, and history inspection.
