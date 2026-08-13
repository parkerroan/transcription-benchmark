.PHONY: setup lint typecheck test generate-fixtures benchmark plot

setup:
	uv sync

lint:
	uv run ruff check .

typecheck:
	uv run pyright scripts/

test:
	uv run pytest tests/ -q -m "not integration"

generate-fixtures:
	uv run python scripts/generate_transcription_fixtures.py

benchmark:
	uv run python scripts/benchmark_multimodal_transcription.py \
		--manifest tests/integration/fixtures/multimodal_eval/manifest.jsonl \
		--output /tmp/multimodal-transcription-benchmark.json

plot:
	uv run python scripts/plot_multimodal_transcription_benchmark.py \
		--report /tmp/multimodal-transcription-benchmark.json \
		--out-dir /tmp/multimodal-transcription-charts
