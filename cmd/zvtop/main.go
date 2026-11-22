/*
Command zvtop is a btop-inspired terminal monitor for NVIDIA GPUs.

It samples the NVIDIA Management Library (NVML) at a fixed interval and shows
the result as a live dashboard: history graphs for utilization, VRAM, power,
temperature and PCIe throughput, meters for fan speed, memory bandwidth,
NVENC/NVDEC load, clocks and performance state, and a table of processes using
the GPU. Metrics a device does not support are shown as n/a.

# Usage

	zvtop [flags]

The flags are:

	-i, -interval duration
	    refresh interval (default 1s)
	-theme name
	    color theme: btop, crimson, dracula, gruvbox or nord
	-mock n
	    simulate n GPUs instead of querying NVML
	-json
	    print a single snapshot as JSON and exit
	-version
	    print the version and exit

Examples:

	# Refresh four times a second with the nord theme.
	zvtop -i 250ms -theme nord

	# Take one snapshot for a script or monitoring pipeline.
	zvtop -json | jq '.devices[] | {name, util, temp, power}'

	# Try the interface without NVIDIA hardware.
	zvtop -mock 4

# Key bindings

	q, ctrl+c           quit
	?                   help
	tab, right, l       next GPU
	shift+tab, left, h  previous GPU
	0-9                 jump to GPU
	up, down, j, k      move the process selection
	pgup, pgdn, g, G    page, first and last process
	s                   cycle the sort column
	r                   reverse the sort order
	x, delete           signal the selected process (t: SIGTERM, k: SIGKILL)
	+, -                slower, faster refresh
	space, p            pause
	t                   cycle themes

# Requirements

Monitoring real hardware requires Linux, an NVIDIA driver that provides
libnvidia-ml.so, and a binary built with cgo. Builds without cgo, and hosts
without a driver, can still run with -mock. In a container, pass the GPUs
through with the NVIDIA container toolkit (docker run --gpus all).

# Exit status

zvtop exits with status 1 and prints a message prefixed with "zvtop:" to
standard error if no GPU backend can be initialized or a flag is invalid.
*/
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/zvdy/zvtop/internal/gpu"
	"github.com/zvdy/zvtop/internal/ui"
)

var version = "dev"

type options struct {
	interval time.Duration
	theme    string
	mock     int
	jsonOut  bool
	version  bool
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "zvtop:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	opts, err := parseFlags(args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return nil
	}
	if err != nil {
		return err
	}

	if opts.version {
		fmt.Fprintln(stdout, "zvtop", buildVersion())
		return nil
	}

	src, err := openSource(opts.mock)
	if err != nil {
		return err
	}
	defer src.Close()

	if opts.jsonOut {
		return printJSON(src, stdout)
	}

	theme, err := ui.LoadTheme(opts.theme)
	if err != nil {
		return err
	}
	model := ui.New(ui.Config{Source: src, Interval: opts.interval, Theme: theme})
	_, err = tea.NewProgram(model, tea.WithAltScreen()).Run()
	return err
}

func parseFlags(args []string, stderr io.Writer) (options, error) {
	var o options
	fs := flag.NewFlagSet("zvtop", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.DurationVar(&o.interval, "interval", time.Second, "refresh interval")
	fs.DurationVar(&o.interval, "i", time.Second, "shorthand for -interval")
	fs.StringVar(&o.theme, "theme", ui.DefaultTheme, "color theme: "+strings.Join(ui.ThemeNames(), ", "))
	fs.IntVar(&o.mock, "mock", 0, "simulate `n` GPUs instead of querying NVML")
	fs.BoolVar(&o.jsonOut, "json", false, "print a single snapshot as JSON and exit")
	fs.BoolVar(&o.version, "version", false, "print version and exit")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: zvtop [flags]\n\nA btop-inspired terminal monitor for NVIDIA GPUs.\n\nFlags:\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return o, err
	}
	if fs.NArg() > 0 {
		return o, fmt.Errorf("unexpected arguments: %v", fs.Args())
	}
	if o.interval <= 0 {
		return o, fmt.Errorf("interval must be positive, got %s", o.interval)
	}
	if o.mock < 0 {
		return o, fmt.Errorf("mock GPU count must be non-negative, got %d", o.mock)
	}
	return o, nil
}

func openSource(mock int) (gpu.Source, error) {
	if mock > 0 {
		return gpu.NewMock(mock, uint64(time.Now().UnixNano())), nil
	}
	src, err := gpu.NewNVML()
	if err != nil {
		return nil, fmt.Errorf("%w\nhint: run with -mock 2 to try the interface without an NVIDIA GPU", err)
	}
	return src, nil
}

func printJSON(src gpu.Source, w io.Writer) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	snap, err := src.Snapshot(ctx)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(snap)
}

func buildVersion() string {
	if version != "dev" {
		return version
	}
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return version
}
