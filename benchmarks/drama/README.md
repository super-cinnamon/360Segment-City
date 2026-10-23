
# DRAMA Benchmarking Harness

This harness benchmarks the two-wheeler risk-assessment pipeline on the DRAMA dataset.

## Setup
1. **Dataset**: Place the DRAMA dataset in a directory. Ensure it follows the layout:
   - `combined/<clip_id>/frame_xxxxxx.png`
   - `integrated_output_v2.json`
2. **Config**: Edit `benchmarks/drama/config_drama.yaml` to set `dataset.root`.

## Running the Benchmark

### 1. Validate Dataset
Check if the dataset is correctly placed and formatted:
```bash
python3 -m benchmarks.drama.runner --dataset-root /path/to/drama --validate-dataset
```

### 2. Run Pipeline
Process the clips and generate risk assessments:
```bash
python3 -m benchmarks.drama.runner --dataset-root /path/to/drama --config benchmarks/drama/config_drama.yaml --output-dir outputs/drama_results
```

### 3. Extract & Evaluate
(Note: The adapter currently needs to be integrated into the evaluation flow)
Run the evaluation script to get metrics:
```bash
python3 benchmarks/drama/evaluate.py
```

## Synthetic Test
To test the plumbing without the real dataset:
1. Generate fixture: `python3 benchmarks/drama/tests/gen_fixture.py`
2. Run runner: `python3 -m benchmarks.drama.runner --dataset-root benchmarks/drama/tests/fixtures --output-dir outputs/drama_synth`
3. Run evaluation on `outputs/drama_synth`.

## Implementation Notes
- **Pipeline Changes**: Modified `src/process.py` and `src/tasks/risk_assessment.py` to support `split_panorama=false` and configurable `danger_zone_threshold`.
- **Input Translation**: The runner creates temporary `.mp4` files from PNG frames to ensure the unmodified pipeline can process the data.
- **Filtering**: The adapter removes reasons with zero probability or leaked CoT markers.
