These modules define analytical formulas and component recipes for MAGI-2
Preview. The packaged architecture and workload configuration determine model
geometry, input sizes and execution counts.

These modules contain shape, FLOP, byte-volume, precision and ordered-subcall
descriptions only. Normal simulations do not read output CSVs, load model weights,
or perform inference. `bridge.py` creates native compute graphs and placements,
and the discrete-event simulator calculates duration on the selected hardware.

Numerical scope:

1. Per-stage compute describes the critical active rank. It must not be divided
   by CP again or multiplied by all participating GPUs. CP/EP collectives use
   their actual devices and the configured fabric bandwidth/latency.
2. Sequential composite subcalls retain their precision and timing order.
   Mathematical FLOPs remain separate from explicit compute amplification.
   MAGI uses the selected device's dtype peak for both vector and matrix work;
   this is an uncalibrated analytical estimate without independent unit
   efficiencies.
3. The formulas give aggregate activation traffic. The native adapter
   charges that volume on HBM, whose read and write rates are equal in supported
   B300/H200 presets; asymmetric HBM is rejected. These traffic descriptors have
   zero physical storage and do not reconstruct real tensor lifetimes; stage
   capacity uses the separate model-owned residency/working-set proxy.
4. Binary-tree broadcasts execute sequential critical hops. CSV traffic counts
   the collective's send/receive boundary once, on its first hop.
5. CPU text FLOPs and memory are reported separately with unknown CPU duration.
   Untimed weight loading/offloading and output postprocessing prevent these
   results from representing calibrated end-to-end latency.
