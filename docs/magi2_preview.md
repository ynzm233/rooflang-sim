# MAGI-2 Preview workflow

命令使用 Windows PowerShell；请先按 [README](../README.md#1-安装并运行) 创建虚拟环境并安装依赖，再在项目根目录执行。多行命令末尾的反引号（`` ` ``）后不能有空格。

Run the complete configured workflow from the repository root:

```powershell
.\.venv\Scripts\python.exe -m rooflang `
  --model magi2_preview --hardware B300Cluster1Node `
  --workflow i2v --output-dir outputs/magi2_preview/i2v
```

Use `--workflow t2v` for text-to-video, `--resolution 540p` or `272p` to omit
refinement, and `--preview-steps` / `--refiner-steps` to change diffusion calls.
`--config` accepts the local workload JSON or a JSON object of overrides, e.g.
`{"workflow":"t2v","resolution":"540p","preview_steps":20,"cp_size":4,"ep_size":4}`.
Explicit command-line options override config fields. The pinned model supports
one 10-second clip per request, CP=EP in 1/2/4/8 on a shared device mesh, and the
resolution presets declared in the local configuration. Unsupported settings
raise an error rather than silently changing the workload. The chosen hardware
must supply enough GPU components for the mesh.

## Model-owned stages

| Stage | Included when | Graphs and invocation counts |
| --- | --- | --- |
| `text_encode` | Always | Positive and negative CPU text forwards; one embedding broadcast |
| `image_vae_encode` | I2V | Image VAE encoder and image-condition broadcast |
| `preview_denoise` | Always | One preview graph, repeated `preview_steps` times |
| `refiner_denoise` | 1080p | Upsample once, conditional and unconditional graphs each repeated `refiner_steps` times |
| `video_vae_decode` | Always | One graph per actual turbo decoding window |
| `audio_vae_decode` | Always | One audio decoder graph |

Each selected stage produces one `<stage>.csv`. All of its branches and windows
share that file. `graph_label` identifies the graph; `graph_multiplier` is the
number of calls not expanded into rows. Layer repetitions are already expanded.
Defaults produce 100 preview calls, 11 refiner-stage invocations (one upsample
plus two branches times five), and nine video windows at 1080p.

`stage_metrics.csv` summarizes stages; `magi2_preview_plan_metrics.csv` records
configuration, shapes, graph-to-stage mapping, counts, timing scope and memory
accounting. The repository-wide export configuration defines its CSV fields and
adds `export_schema_version=2`; optional fields without a defined value for this run are
omitted. The unified entry point also writes `summary.json`, `config.snapshot.json`,
and `provenance.json`; the low-level CSV exporter writes CSV only by default. A successful rerun in the
same output directory removes obsolete stage CSVs listed in this model's prior
plan manifest, preserving unrelated files.

For this serial workflow:

```text
stage execution_time_ms = sum(row execution_time_ms * row graph_multiplier)
plan execution_time_ms  = sum(stage execution_time_ms)
```

Untimed rows are excluded from timing sums. FLOPs and traffic follow the same
multiplication rule. Repetition does not multiply resident weights or activation
peaks. Composite operators are lowered into ordered native subcalls; CSV
metadata retains their node, layer, recipe index and subcall identity. One
composite operator can therefore occupy multiple rows while its total work and
traffic remain conserved.

The shared exporter also carries model-specific statistics:

| Fields | Meaning and availability |
| --- | --- |
| `tag_mask`, `kv_storage_dtype` | Component tags and each operator's configured storage dtype. Composite subcalls inherit their parent operator's declaration. These fields are blank on CPU accounting rows. |
| `cpu_flops_breakdown` | JSON object containing the seven CPU forward FLOPs components; their sum equals `cpu_flops`. Only the text stage has this column. |
| `overlapped_compute_ms`, `overlapped_memory_ms` | Ideal roofline bottleneck attribution: keep compute time when compute >= memory, and memory time when memory >= compute. Ties appear in both columns. They exclude network time and DES contention, and must not be added to reconstruct wall time. Communication and untimed CPU rows are blank; the text stage omits these columns. |
| `weight_saved`, `resident_weight_bytes` | Weight storage charged by the operator formula versus physical weight tensors after sharing. Reuse rows may have zero `weight_saved` while referencing resident weights. Stage residency deduplicates physical weights. |
| Omitted columns | `cube_compute_ms` and `vector_compute_ms` require separate engine throughput absent from the current MAGI hardware model. MAGI has no KV cache accounting, so `kv_cache_read`, `kv_cache_saved`, and `kv_cache_load` remain absent. |

Schema version 2 reports the operator-accounted storage in `weight_saved` and
the physical tensor storage in `resident_weight_bytes` where applicable.
The stage `cpu_execution_time_ms` is blank when CPU execution is absent or
unestimated; a modeled zero remains zero. I2V/T2V regression checks cover FLOPs,
traffic, stage memory totals and simulated times at this graph row granularity.

## Runtime and accounting scope

The implementation is self-contained. `architecture.json` and `workload.json`
provide architecture and numeric workload inputs. Normal simulations do not read
precomputed CSVs, model weights, tokenizer files, prompts, or images and do not
use network access. The local cost modules derive FLOPs and bytes from shapes.
Native graphs, tensors, placement, communication kernels and the discrete-event
simulator determine execution time.

Distributed transformer costs model the critical rank after CP/EP partitioning,
including head padding and rank-specific adapter work. Input and output adapter
ranks are selected independently with native timing. Collective participants use
the selected hardware topology; CP and EP share the same mesh. These are
per-rank costs, not global work divided by the number of cards.

Numerical times depend on the selected device peaks, fabrics and scheduling
model. MAGI uses the selected device's dtype peaks without separate cube/vector
throughput or launch calibrations. Explicit compute amplification multipliers,
where present, affect timing work rather than mathematical FLOPs. These are
analytical estimates, not measured device latency.

CPU text encoding retains independently computed FLOPs and resident-memory
figures, but its execution time is blank and `timing_included=False`. CPU latency,
postprocessing/mux, and cold loading/restoring model weights are not estimated.
The plan explicitly marks end-to-end timing incomplete. `request_per_second`
means modeled accelerator clips/s, not full pipeline service throughput.

Aggregate traffic tensors retain their full `size_bytes` for bandwidth timing,
with explicit `storage_bytes=0` because repeated traffic is not a simultaneous
allocation. Ordinary tensors retain their original allocation behavior. Native
graph memory therefore tracks actual resident weight descriptors; the plan uses
checkpoint-derived module residency, persistent
latent tensors, loading/unloading schedule, and explicit temporary-memory
scenarios. `capacity_verdict_is_provisional=True` preserves that limitation;
unknown library workspaces and allocator fragmentation are not included.

## Configuration and validation

The packaged MAGI-2 Preview architecture and workload files define component
dimensions, execution counts and input geometry. Runtime imports use only local
modules. Run records preserve the resolved configuration, hardware description,
package fingerprint and environment; frozen numerical baselines support
regression checks.

Validation covers shapes and accounting across I2V/T2V, all three resolutions
and multiple CP/EP sizes; parameter-derived operator equations; native
communication and graph scheduling; exact stage/file sets; weighted CSV
reconstruction; and step changes without multiplying resident storage. An audited
subprocess rejects network access and application input reads outside this
repository and its Python runtime while running the CLI.

Default I2V/T2V baseline checks include memory traffic, resident weights,
activation proxies, invocation counts and CPU FLOPs. Accelerator FLOPs and
timing are checked with floating-point summation tolerance. Each check runs the
native simulator before comparing its results with the frozen baseline.
