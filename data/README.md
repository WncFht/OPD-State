# `data/`

Every parquet the launchers read. `_common.sh` defaults `DATA_BASE` to
`${OPD_ROOT}/data`, so a fresh clone needs no data wiring.

No dataset ships with this repository. `data/` is gitignored except for this
file and the two fixtures under `smoke/`. Every row below other than `smoke/`
must be rebuilt from its public source with the scripts in
[`prep/`](prep/README.md). The row counts, `data_source` and
`ability` values listed here are what a correct rebuild produces.

## Layout

```
data/<domain>/<dataset>/{train,test}.parquet
```

| Directory | Dataset | Rows | `data_source` | `ability` | Use |
| --- | --- | --- | --- | --- | --- |
| `math/` | `DAPO/train.parquet` | 17917 | `DAPO` | `math` | Math training set |
| | `MATH-500/test.parquet` | 500 | `MATH-500` | `math` | Math validation set |
| | `DAPO-oneshot/t22.parquet` | 64 | `DAPO` | `math` | 1-shot training set: DAPO problem 22, repeated 64 times |
| `code/` | `TACO/train.parquet` | 24701 | `taco` | `code` | Code training set |
| | `LCBv6/test.parquet` | 175 | `livecodebench` | `code` | Code validation set |
| `if/` | `MultiIF/train.parquet` | 4501 | `MultiIF` | `instruction_following` | IF training set, first turn only |
| | `MultiIF/eval_3turn.parquet` | 4445 | — | — | IF validation set, read by `eval/run_if_eval.py`, not by verl |
| `agentic/` | `xlam/train.parquet` | 60000 | `xlam` | `function_calling` | FC training set |
| | `BFCL/{simple,multiple,parallel,parallel_multiple,live_simple}/test.parquet` | 400/200/200/200/258 | `bfcl_*` | `function_calling` | FC validation set, five subsets |
| `chat/` | `WildChat/train.parquet` | 192824 | `WildChat` | `general` | Read by no launcher |
| `smoke/` | Two fixtures | 16 / 192 | mixed | mixed | Unused since `recipe/smoke.sh` was removed |

There is no `template/` directory. `MODE=template` reads no corpus:
`TemplatePromptDataset` emits placeholder rows and `data_files` is ignored.

## `data_source` selects the verifier branch

`opd_mixed_verify` routes on exact `data_source` membership, with math as the
fallback branch. A wrong string does not raise. It falls through to the math
verifier, which scores Python programs near zero.

## Schema requirements for MOPD

MOPD passes the math, code and IF training sets as one `data.train_files`, and
the three validation sets as one `data.val_files`. `datasets` requires
field-by-field identical schemas to concatenate them.

- The six files are written through `align_mopd_schema.write_aligned()`, against
  a schema defined in `align_mopd_schema.py`.
- `python data/prep/align_mopd_schema.py` with no arguments verifies them.
- Struct field ordering is not a difference; `datasets` normalizes it.
- `agentic/` and `chat/` are excluded: FC trains and validates alone, and no
  launcher reads WildChat.
- `if/MultiIF/eval_3turn.parquet` is excluded and not named `test.parquet`. It
  holds three prompts per row, has no `reward_model` column, and is read only by
  `eval/run_if_eval.py`.

## Fresh math prompt pool

`build_fresh_math_data.py` builds the entire unique prompt pool independently of training steps. It filters by the canonical rendered token length, removes duplicate token prefixes, excludes every copy when answers conflict, and records data/tokenizer/template hashes. Training owns the seeded epoch permutation; prompt visits are counted by content.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=verl ../envs/opd-state/bin/python \
    data/prep/build_fresh_math_data.py \
    --input ../MD-OPD/data/math/DAPO/train.parquet \
    --student ../models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output runs/native-data/math-full
NATIVE_DATA_MANIFEST="$PWD/runs/native-data/math-full/manifest.json" \
    OPD_FREEZE_SOURCE=1 bash recipe/math_native.sh student_fresh
```

Use a fresh output directory when rebuilding. The launcher binds the manifest to the exact Parquet, tokenizer/template and prompt cap before initializing GPU workers. The native path supports `student_teacher_fusion` and `initial_teacher_fusion` with `TEACHER_WEIGHT`, sequence directions with `SWITCH_TOKENS`, and batch mixtures with `TEACHER_FRACTION`; the experiment report distinguishes verified short runs from full quality results.
