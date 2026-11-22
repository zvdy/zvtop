// Package zvtop is the module root of zvtop, a btop-inspired terminal monitor
// for NVIDIA GPUs.
//
// zvtop is an application, not an importable library: the module contains no
// exported Go API. The documentation for the program itself lives with the
// command, see [github.com/zvdy/zvtop/cmd/zvtop].
//
// # Install
//
//	go install github.com/zvdy/zvtop/cmd/zvtop@latest
//
// Prebuilt Linux binaries are attached to each GitHub release, and a container
// image is published to ghcr.io/zvdy/zvtop.
//
// # Overview
//
// zvtop reads telemetry through the NVIDIA Management Library (NVML) and
// renders it as a live dashboard: braille history graphs for utilization,
// memory, power, temperature and PCIe throughput, meters for fan, clocks and
// encoder/decoder load, and a per-process table that can send signals.
//
// The module is organised as follows. These packages are internal and are
// listed for contributors:
//
//   - cmd/zvtop: flag parsing, backend selection and program lifecycle
//   - internal/gpu: backend-agnostic telemetry types, the NVML backend and a
//     deterministic simulated backend
//   - internal/history: fixed-capacity time series storage
//   - internal/ui: the interactive dashboard, themes and drawing primitives
package zvtop
