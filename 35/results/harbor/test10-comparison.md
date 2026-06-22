# test10 comparison for article

All rows use `deepseek/deepseek-v4-flash`, `CONCURRENCY=1`, `USE_SHARED_DOCKER_BRIDGE=1`, `N_ATTEMPTS=1`, and `trace_mode=none`.
The `test10` traces were not inspected while selecting harness changes.

| step | harness | source run | main change | pass | exceptions | note |
|---|---|---|---|---:|---:|---|
| baseline | `harness-runs/baseline/harness.json` | `tb2-t10s1-baseline-test10-20260621-210351` | minimal prompt, `temperature=1`, `recursionLimit=150` | 4/10 | 0 | starting point |
| improvement-1 | `harness-runs/improvement-1/harness.json` | `tb2-t10s1-temp-only-01-test10-20260622-001203` | lower `temperature` to `0.1` | 6/10 | 0 | clean gain without prompt changes |
| improvement-2 | `harness-runs/improvement-2/harness.json` | `tb2-t10s1-short-discipline-01-test10-20260622-065033` | add generic work discipline and raise `recursionLimit` to `220` | 7/10 | 3 | higher pass count, with timeout caveat |

## Task outcomes

| task | baseline | improvement-1 | improvement-2 |
|---|---:|---:|---:|
| `cobol-modernization` | 1/1 | 1/1 | 1/1 |
| `crack-7z-hash` | 0/1 | 0/1 | 0/1, timeout |
| `extract-elf` | 0/1 | 1/1 | 0/1 |
| `kv-store-grpc` | 1/1 | 0/1 | 1/1 |
| `merge-diff-arc-agi-task` | 1/1 | 1/1 | 1/1 |
| `polyglot-c-py` | 0/1 | 0/1 | 1/1, timeout |
| `pypi-server` | 0/1 | 1/1 | 1/1, timeout |
| `raman-fitting` | 0/1 | 0/1 | 0/1 |
| `sqlite-db-truncate` | 0/1 | 1/1 | 1/1 |
| `sqlite-with-gcov` | 1/1 | 1/1 | 1/1 |

## Interpretation

The first improvement shows that reducing sampling variance can recover two additional tasks without adding prompt guidance.
The second improvement raises the score by one more task, but the extra timeout exceptions show that the harness is spending more time before returning control to Harbor.
For the article, this is a useful result because it shows both sides of harness engineering: pass count can improve, while runtime discipline still needs measurement.
