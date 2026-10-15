The analytical formulas and MAGI component recipes come from `Markov_Infer_Sim`.
`SOURCES.json` records the reference Git revision and SHA-256 hash of each source
file; the source files can include local changes beyond that revision. The model
geometry is pinned to official MAGI-2-preview revision `f68a0f9`.

These modules contain shape, FLOP, byte-volume, precision and ordered-subcall
descriptions only. They do not import an external checkout, read output CSVs,
invoke Markov's simulation backend, load model weights, or perform inference.
`bridge.py` creates ordinary RoofLang compute graphs and placements, and the
RoofLang discrete-event simulator calculates duration on the selected hardware.

Numerical scope:

1. Per-stage compute describes the critical active rank. It must not be divided
   by CP again or multiplied by all participating GPUs. CP/EP collectives use
   their actual devices and native RoofLang fabric bandwidth/latency.
2. Sequential composite subcalls retain their precision and timing order.
   Mathematical FLOPs remain separate from explicit compute amplification.
   RoofLang uses the selected device's dtype peak for both vector and matrix
   work; this differs from Markov's independent unit efficiencies and is an
   uncalibrated analytical estimate.
3. The reference formulas give aggregate activation traffic. The native adapter
   charges that volume on HBM, whose read and write rates are equal in supported
   B300/H200 presets; asymmetric HBM is rejected. These traffic descriptors have
   zero physical storage and do not reconstruct real tensor lifetimes; stage
   capacity uses the separate model-owned residency/working-set proxy.
4. Binary-tree broadcasts execute sequential critical hops. CSV traffic counts
   the source collective's send/receive boundary once, on its first hop.
5. CPU text FLOPs and memory are reported separately with unknown CPU duration.
   Untimed weight loading/offloading and output postprocessing prevent these
   results from representing calibrated end-to-end latency.
