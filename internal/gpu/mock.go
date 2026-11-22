package gpu

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"time"
)

// Mock simulates GPUs running realistic, randomly changing workloads:
// training runs with periodic eval/checkpoint dips, bursty inference
// servers, renders and encodes, plus a churning set of desktop and helper
// processes. Temperature, fan, power and clocks follow the load with lag
// the way real cards do. It is useful for demos, screenshots and development
// on machines without NVIDIA hardware. A fixed seed and clock yield a
// reproducible sequence.
type Mock struct {
	mu      sync.Mutex
	rng     *rand.Rand
	now     func() time.Time
	start   time.Time
	last    float64
	nextPID uint32
	gpus    []*mockGPU
}

type mockModel struct {
	name           string
	memGiB         uint64
	tdp, idleW     float64
	gfxMax, memMax uint32
	slowdown       uint32
	pcie           float64 // bytes/s per direction
}

var mockModels = []mockModel{
	{"NVIDIA GeForce RTX 5090", 32, 575, 24, 3090, 14001, 90, 63e9},
	{"NVIDIA GeForce RTX 5090", 32, 575, 24, 3090, 14001, 90, 63e9},
	{"NVIDIA GeForce RTX 5080", 16, 360, 16, 3105, 15001, 90, 63e9},
	{"NVIDIA GeForce RTX 4090", 24, 450, 20, 3105, 10501, 90, 31.5e9},
}

type workload int

const (
	wlIdle workload = iota
	wlTraining
	wlInference
	wlRender
	wlEncode
)

type mockGPU struct {
	index   int
	model   mockModel
	uuid    string
	desktop bool

	job     workload
	jobProc *mockProc
	until   float64
	// training: eval/checkpoint dip every period seconds
	period, offset float64
	// inference: request bursts
	burstUntil, nextBurst, burstLevel float64

	procs []*mockProc

	load, temp, fan float64
}

type mockProc struct {
	Process
	target  uint64  // steady-state VRAM
	ramp    float64 // seconds to allocate target
	born    float64 // sim time
	until   float64 // sim time of exit, 0 = until removed
	share   float64 // fraction of device SM it accounts for
	active  float64 // probability NVML has a utilization sample
	enc     [2]int
	dec     [2]int
	reserve bool // job process: VRAM scales with device total
}

type procTemplate struct {
	name, user, cmd string
	kind            ProcKind
	memMiB          [2]int
	share, active   float64
	enc, dec        [2]int
}

var desktopProcs = []procTemplate{
	{name: "kitty", user: "demo", cmd: "kitty", kind: KindGraphics, memMiB: [2]int{24, 64}, share: 0.01, active: 0.2},
	{name: "kitty", user: "demo", cmd: "kitty --single-instance", kind: KindGraphics, memMiB: [2]int{24, 64}, share: 0.01, active: 0.2},
	{name: "firefox", user: "demo", cmd: "/usr/lib/firefox/firefox", kind: KindGraphics, memMiB: [2]int{180, 620}, share: 0.04, active: 0.6, dec: [2]int{0, 14}},
	{name: "chrome", user: "demo", cmd: "/opt/google/chrome/chrome --type=gpu-process --ozone-platform=wayland --use-gl=angle --use-angle=vulkan", kind: KindGraphics, memMiB: [2]int{220, 760}, share: 0.04, active: 0.5, dec: [2]int{0, 10}},
	{name: "code", user: "demo", cmd: "/opt/visual-studio-code/code --type=gpu-process --ozone-platform=wayland", kind: KindGraphics, memMiB: [2]int{90, 260}, share: 0.01, active: 0.3},
	{name: "Discord", user: "demo", cmd: "/opt/discord/Discord --type=gpu-process --ozone-platform=wayland", kind: KindGraphics, memMiB: [2]int{120, 420}, share: 0.01, active: 0.3},
	{name: "obs", user: "demo", cmd: "obs --startreplaybuffer --minimize-to-tray", kind: KindGraphics | KindCompute, memMiB: [2]int{320, 720}, share: 0.05, active: 0.9, enc: [2]int{12, 28}},
	{name: "steamwebhelper", user: "demo", cmd: "./steamwebhelper -nocrashdialog -lang=en_US -cachedir=/home/demo/.local/share/Steam/config/htmlcache", kind: KindGraphics, memMiB: [2]int{40, 130}, share: 0.01, active: 0.1},
	{name: "mpv", user: "demo", cmd: "mpv --hwdec=nvdec-copy lectures/cs336-lecture-09.mkv", kind: KindGraphics, memMiB: [2]int{90, 180}, share: 0.02, active: 0.9, dec: [2]int{6, 18}},
}

var sideProcs = []procTemplate{
	{name: "python3", user: "ml", cmd: "python3 eval.py --ckpt runs/exp42/step_12000.pt --tasks hellaswag,arc_easy", kind: KindCompute, memMiB: [2]int{1800, 5200}, share: 0.25, active: 0.9},
	{name: "python3", user: "ml", cmd: "python3 -m ipykernel_launcher -f /run/user/1000/jupyter/runtime/kernel-7f3a9c.json", kind: KindCompute, memMiB: [2]int{600, 4200}, share: 0.06, active: 0.4},
	{name: "ffmpeg", user: "demo", cmd: "ffmpeg -hwaccel cuda -i clips/raw_0412.mkv -vf thumbnail_cuda,scale_cuda=640:-2 -frames:v 1 thumbs/0412.jpg", kind: KindCompute, memMiB: [2]int{180, 420}, share: 0.04, active: 0.9, enc: [2]int{0, 8}, dec: [2]int{30, 70}},
	{name: "python3", user: "ml", cmd: "python3 embed.py --model BAAI/bge-m3 --input data/corpus.parquet --batch-size 256", kind: KindCompute, memMiB: [2]int{2400, 6200}, share: 0.3, active: 0.95},
}

type jobTemplate struct {
	procTemplate
	memFrac [2]float64 // of device VRAM
}

var jobs = map[workload][]jobTemplate{
	wlTraining: {
		{procTemplate{name: "python3", user: "ml", cmd: "python3 train.py --config configs/llama3-8b-lora.yaml --bf16 --grad-checkpointing", kind: KindCompute, share: 0.96, active: 1}, [2]float64{0.62, 0.86}},
		{procTemplate{name: "torchrun", user: "ml", cmd: "torchrun --standalone --nproc_per_node=1 pretrain.py --model gpt2-xl --batch-size 16 --compile", kind: KindCompute, share: 0.95, active: 1}, [2]float64{0.55, 0.8}},
		{procTemplate{name: "python3", user: "ml", cmd: "accelerate launch finetune.py --model mistralai/Mistral-7B-v0.3 --lr 2e-4 --epochs 3", kind: KindCompute, share: 0.95, active: 1}, [2]float64{0.6, 0.84}},
	},
	wlInference: {
		{procTemplate{name: "ollama", user: "ollama", cmd: "/usr/lib/ollama/llama-server --model /var/lib/ollama/models/blobs/sha256-a3de86cd1c132c82 --ctx-size 8192 --n-gpu-layers 99 --port 41217", kind: KindCompute, share: 0.97, active: 1}, [2]float64{0.3, 0.6}},
		{procTemplate{name: "python3", user: "ml", cmd: "python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct --gpu-memory-utilization 0.85 --max-model-len 16384", kind: KindCompute, share: 0.97, active: 1}, [2]float64{0.8, 0.85}},
		{procTemplate{name: "python3", user: "demo", cmd: "python3 main.py --listen 0.0.0.0 --port 8188 --highvram --preview-method auto", kind: KindCompute, share: 0.97, active: 1}, [2]float64{0.35, 0.7}},
	},
	wlRender: {
		{procTemplate{name: "blender", user: "demo", cmd: "blender -b shots/sh040_v12.blend -E CYCLES -s 1 -e 240 -a -- --cycles-device OPTIX", kind: KindCompute | KindGraphics, share: 0.94, active: 1}, [2]float64{0.2, 0.5}},
	},
	wlEncode: {
		{procTemplate{name: "ffmpeg", user: "demo", cmd: "ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i raw/take_07.mov -c:v hevc_nvenc -preset p5 -tune hq -b:v 40M out/take_07.mp4", kind: KindCompute, share: 0.3, active: 1, enc: [2]int{62, 96}, dec: [2]int{28, 55}}, [2]float64{0.02, 0.04}},
	},
}

// NewMock returns a simulated source with n devices. A fixed seed yields a
// reproducible sequence.
func NewMock(n int, seed uint64) *Mock {
	m := &Mock{
		rng:   rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)),
		now:   time.Now,
		start: time.Now(),
	}
	m.nextPID = 1800 + uint32(m.rng.IntN(1200))

	for i := range max(n, 1) {
		g := &mockGPU{
			index:   i,
			model:   mockModels[i%len(mockModels)],
			desktop: i == 0,
			uuid: fmt.Sprintf("GPU-%08x-%04x-%04x-%04x-%012x",
				m.rng.Uint32(), m.rng.IntN(1<<16), m.rng.IntN(1<<16), m.rng.IntN(1<<16), m.rng.Uint64()>>16),
		}
		if g.desktop {
			xorg := &mockProc{
				Process: Process{
					PID: 900 + uint32(m.rng.IntN(600)), Name: "Xorg", User: "root",
					Command: "/usr/lib/Xorg :0 -seat seat0 -auth /run/sddm/xauth_kqWv -nolisten tcp -background none -noreset vt1",
					Kind:    KindGraphics,
				},
				target: m.mib(220, 480), share: 0.03, active: 0.9, born: -3600,
			}
			g.procs = append(g.procs, xorg)
			for _, j := range m.rng.Perm(len(desktopProcs))[:5+m.rng.IntN(3)] {
				g.procs = append(g.procs, m.spawn(desktopProcs[j], -m.between(60, 3600), 0))
			}
		} else if m.chance(0.5) {
			g.procs = append(g.procs, m.spawn(sideProcs[1], -m.between(60, 600), 0))
		}
		// Start mid-workload, never idle, so the first screens are busy.
		m.startJob(g, -m.between(20, 240), m.pick(g, false))
		g.until = max(g.until, m.between(45, 150))
		g.load = g.targetLoad(m, 0)
		g.temp = 30 + g.load*46
		g.fan = fanCurve(g.temp)
		m.gpus = append(m.gpus, g)
	}
	return m
}

// Close implements Source.
func (m *Mock) Close() error { return nil }

// Snapshot implements Source.
func (m *Mock) Snapshot(ctx context.Context) (Snapshot, error) {
	if err := ctx.Err(); err != nil {
		return Snapshot{}, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	now := m.now()
	t := now.Sub(m.start).Seconds()
	dt := min(max(t-m.last, 0.05), 5)
	m.last = t

	snap := Snapshot{Taken: now, Driver: "580.95.05", CUDA: "13.0"}
	for _, g := range m.gpus {
		m.step(g, t, dt)
		snap.Devices = append(snap.Devices, m.device(g, t))
	}
	return snap, nil
}

// step advances the workload, process set and thermals of one GPU by dt.
func (m *Mock) step(g *mockGPU, t, dt float64) {
	if t >= g.until {
		m.startJob(g, t, m.pick(g, true))
	}

	g.procs = slices.DeleteFunc(g.procs, func(p *mockProc) bool { return p.until > 0 && t >= p.until })
	if m.chance(dt * 0.025) {
		tpl := sideProcs[m.rng.IntN(len(sideProcs))]
		g.procs = append(g.procs, m.spawn(tpl, t, t+m.between(6, 90)))
	}
	if g.desktop && m.chance(dt*0.02) {
		if len(g.procs) > 6 && m.chance(0.5) {
			// a desktop app exits (never Xorg or the job)
			i := 1 + m.rng.IntN(len(g.procs)-1)
			if g.procs[i] != g.jobProc && g.procs[i].until == 0 {
				g.procs = slices.Delete(g.procs, i, i+1)
			}
		} else {
			g.procs = append(g.procs, m.spawn(desktopProcs[m.rng.IntN(len(desktopProcs))], t, 0))
		}
	}

	ease := func(tau float64) float64 { return 1 - math.Exp(-dt/tau) }
	g.load += (g.targetLoad(m, t) - g.load) * ease(0.35)
	g.load = clamp01(g.load)
	g.temp += (30 + g.load*46 + m.jitter(0.6) - g.temp) * ease(7)
	// fans spin up and down slowly; zero-RPM below ~50°C
	target := fanCurve(g.temp)
	g.fan += max(min(target-g.fan, 6*dt), -4*dt)
	if g.fan < 1 && target == 0 {
		g.fan = 0
	}
}

func (g *mockGPU) targetLoad(m *Mock, t float64) float64 {
	base := 0.0
	if g.desktop {
		base = 0.03 + m.between(0, 0.05)
	}
	switch g.job {
	case wlTraining:
		if g.inDip(t) {
			return base + 0.08 + m.jitter(0.05)
		}
		return 0.93 + m.jitter(0.04)
	case wlInference:
		if t >= g.nextBurst && t >= g.burstUntil {
			g.burstUntil = t + m.between(0.8, 7)
			g.burstLevel = m.between(0.55, 0.99)
			g.nextBurst = g.burstUntil + m.rng.ExpFloat64()*4
		}
		if t < g.burstUntil {
			return g.burstLevel + m.jitter(0.05)
		}
		return base + 0.02
	case wlRender:
		if math.Mod(t-g.offset, g.period) < 1 {
			return base + 0.25 + m.jitter(0.05) // frame write-out
		}
		return 0.86 + 0.06*math.Sin(t/3) + m.jitter(0.05)
	case wlEncode:
		return base + 0.28 + m.jitter(0.06)
	default:
		return base + m.between(0, 0.03)
	}
}

func (g *mockGPU) inDip(t float64) bool {
	return math.Mod(t-g.offset, g.period) < 4
}

func fanCurve(temp float64) float64 {
	if temp < 50 {
		return 0
	}
	return min(30+(temp-50)*2.4, 100)
}

// pick chooses the next workload. Secondary GPUs are mostly compute boxes.
func (m *Mock) pick(g *mockGPU, allowIdle bool) workload {
	weights := map[workload]float64{wlTraining: 0.45, wlInference: 0.3, wlRender: 0.1, wlEncode: 0.05, wlIdle: 0.1}
	if g.desktop {
		weights = map[workload]float64{wlTraining: 0.2, wlInference: 0.35, wlRender: 0.15, wlEncode: 0.1, wlIdle: 0.2}
	}
	if !allowIdle {
		weights[wlIdle] = 0
	}
	var sum float64
	for _, w := range weights {
		sum += w
	}
	r := m.rng.Float64() * sum
	for _, wl := range []workload{wlIdle, wlTraining, wlInference, wlRender, wlEncode} {
		if r -= weights[wl]; r < 0 {
			return wl
		}
	}
	return wlTraining
}

func (m *Mock) startJob(g *mockGPU, t float64, wl workload) {
	if g.jobProc != nil {
		g.procs = slices.DeleteFunc(g.procs, func(p *mockProc) bool { return p == g.jobProc })
		g.jobProc = nil
	}
	g.job = wl
	dur := map[workload][2]float64{
		wlIdle: {15, 45}, wlTraining: {120, 400}, wlInference: {60, 200}, wlRender: {60, 180}, wlEncode: {25, 70},
	}[wl]
	g.until = t + m.between(dur[0], dur[1])
	g.period = m.between(25, 55)
	g.offset = m.between(0, g.period)
	g.burstUntil, g.nextBurst = 0, t

	if wl == wlIdle {
		return
	}
	opts := jobs[wl]
	tpl := opts[m.rng.IntN(len(opts))]
	p := m.spawn(tpl.procTemplate, t, 0)
	total := g.model.memGiB << 30
	var used uint64 = 400 << 20 // driver/context reserve
	for _, q := range g.procs {
		used += q.target
	}
	free := total - min(used, total)
	p.target = min(uint64(m.between(tpl.memFrac[0], tpl.memFrac[1])*float64(total)), free*95/100)
	p.ramp = m.between(4, 15)
	p.reserve = true
	g.jobProc = p
	g.procs = append(g.procs, p)
}

func (m *Mock) spawn(tpl procTemplate, born, until float64) *mockProc {
	m.nextPID += 1 + uint32(m.rng.IntN(900))
	return &mockProc{
		Process: Process{PID: m.nextPID, Name: tpl.name, User: tpl.user, Command: tpl.cmd, Kind: tpl.kind},
		target:  m.mib(tpl.memMiB[0], tpl.memMiB[1]),
		ramp:    m.between(0.5, 3),
		born:    born,
		until:   until,
		share:   tpl.share,
		active:  tpl.active,
		enc:     tpl.enc,
		dec:     tpl.dec,
	}
}

func (m *Mock) device(g *mockGPU, t float64) Device {
	md := g.model
	total := md.memGiB << 30
	load := g.load

	var used uint64 = 400 << 20
	var enc, dec uint32
	procs := make([]Process, 0, len(g.procs))
	for _, mp := range g.procs {
		p := mp.Process
		grow := clamp01((t - mp.born) / mp.ramp)
		p.GPUMem = uint64(float64(mp.target) * (0.15 + 0.85*grow))
		if !mp.reserve {
			p.GPUMem += uint64(m.jitter(2)) << 20
		}
		used += p.GPUMem

		if m.chance(mp.active) {
			sm := mp.share * 100
			if mp == g.jobProc {
				sm *= load
			} else {
				sm *= m.between(0.2, 1.2)
			}
			p.SM = Some(uint32(min(max(sm+m.jitter(1.5), 0), 100)))
			if mp.enc[1] > 0 || mp.dec[1] > 0 {
				p.Enc = Some(uint32(mp.enc[0] + m.rng.IntN(mp.enc[1]-mp.enc[0]+1)))
				p.Dec = Some(uint32(mp.dec[0] + m.rng.IntN(mp.dec[1]-mp.dec[0]+1)))
				enc, dec = min(enc+p.Enc.V, 100), min(dec+p.Dec.V, 100)
			} else {
				p.Enc, p.Dec = Some[uint32](0), Some[uint32](0)
			}
		}
		procs = append(procs, p)
	}
	slices.SortFunc(procs, func(a, b Process) int { return cmp.Compare(a.PID, b.PID) })
	used = min(used, total)

	power := md.idleW + (md.tdp-md.idleW)*math.Pow(load, 1.15)*m.between(0.9, 1.02)
	power = min(max(power+m.jitter(3), md.idleW*0.8), md.tdp)

	gfx, mem := uint32(180+m.rng.IntN(150)), uint32(405)
	pstate := 8
	if load > 0.05 {
		throttle := max(g.temp-55, 0) * 9
		gfx = uint32(max(float64(md.gfxMax)-throttle-m.between(0, 60), 900))
		mem = md.memMax
		pstate = 0
		if g.job != wlIdle {
			pstate = 2 // CUDA contexts run in P2 on GeForce
		}
	}

	var rx, tx, bw float64
	switch g.job {
	case wlTraining:
		rx, tx, bw = m.between(1.5e9, 4e9), m.between(0.1e9, 0.4e9), 0.55
		if g.inDip(t) {
			rx, tx = m.between(0.2e9, 1e9), m.between(6e9, 22e9) // checkpoint write
		}
	case wlInference:
		rx, tx, bw = m.between(2e6, 30e6), m.between(1e6, 12e6), 0.85
		if t < g.burstUntil {
			rx, tx = m.between(80e6, 600e6), m.between(20e6, 120e6)
		}
	case wlRender:
		rx, tx, bw = m.between(10e6, 400e6), m.between(5e6, 80e6), 0.4
	case wlEncode:
		rx, tx, bw = m.between(300e6, 900e6), m.between(80e6, 300e6), 0.25
	default:
		rx, tx, bw = m.between(0.2e6, 3e6), m.between(0.2e6, 3e6), 0.3
	}

	return Device{
		Index:        g.index,
		Name:         md.name,
		UUID:         g.uuid,
		BusID:        fmt.Sprintf("00000000:%02X:00.0", 0x01+g.index*0x40),
		Util:         Some(uint32(load * 100)),
		MemUtil:      Some(uint32(clamp01(load*bw+m.jitter(0.03)) * 100)),
		EncUtil:      Some(enc),
		DecUtil:      Some(dec),
		MemUsed:      used,
		MemTotal:     total,
		Temp:         Some(uint32(g.temp)),
		TempSlowdown: Some(md.slowdown),
		Power:        Some(power),
		PowerLimit:   Some(md.tdp),
		Fan:          Some(uint32(g.fan)),
		ClockGfx:     Some(gfx),
		ClockGfxMax:  Some(md.gfxMax),
		ClockMem:     Some(mem),
		ClockMemMax:  Some(md.memMax),
		PCIeRx:       Some(uint64(min(rx, md.pcie))),
		PCIeTx:       Some(uint64(min(tx, md.pcie))),
		PState:       Some(pstate),
		Processes:    procs,
	}
}

func (m *Mock) jitter(scale float64) float64 { return (m.rng.Float64()*2 - 1) * scale }
func (m *Mock) between(a, b float64) float64 { return a + m.rng.Float64()*(b-a) }
func (m *Mock) chance(p float64) bool        { return m.rng.Float64() < p }
func (m *Mock) mib(lo, hi int) uint64        { return uint64(lo+m.rng.IntN(max(hi-lo, 1))) << 20 }

func clamp01(v float64) float64 { return min(max(v, 0), 1) }
