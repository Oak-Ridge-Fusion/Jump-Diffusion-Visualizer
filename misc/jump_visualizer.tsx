import React, { useEffect, useMemo, useRef, useState } from "react";

const clamp = (v: number, min: number, max: number): number => Math.min(max, Math.max(min, v));

type Particle = {
  x: number;
  alive: boolean;
  color: string;
  trail: number[];
  jumpFlash: number;
};

// Simple UI components to replace external dependencies
const Card: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
  <div className={`rounded-lg border border-gray-300 bg-white shadow-md ${className}`}>{children}</div>
);

const CardHeader: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
  <div className={`p-4 border-b border-gray-200 ${className}`}>{children}</div>
);

const CardTitle: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
  <h2 className={`text-lg font-bold ${className}`}>{children}</h2>
);

const CardContent: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
  <div className={`p-4 ${className}`}>{children}</div>
);

const Button: React.FC<{ children: React.ReactNode; onClick?: () => void; variant?: string; className?: string }> = ({
  children,
  onClick,
  className = "",
}) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition ${className}`}
  >
    {children}
  </button>
);

const Badge: React.FC<{ children: React.ReactNode; variant?: string; className?: string }> = ({ children, className = "" }) => (
  <span className={`inline-block px-2 py-1 text-sm bg-gray-200 text-gray-800 rounded ${className}`}>{children}</span>
);

const Slider: React.FC<{
  value: number[];
  min: number;
  max: number;
  step: number;
  onValueChange: (v: number[]) => void;
}> = ({ value, min, max, step, onValueChange }) => (
  <input
    type="range"
    min={min}
    max={max}
    step={step}
    value={value[0]}
    onChange={(e) => onValueChange([parseFloat(e.target.value)])}
    className="w-full"
  />
);

// Icon components
const Play: React.FC<{ className?: string }> = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M8 5v14l11-7z" />
  </svg>
);

const Pause: React.FC<{ className?: string }> = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M6 4h4v16H6zm8 0h4v16h-4z" />
  </svg>
);

const RotateCcw: React.FC<{ className?: string }> = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
    <polyline points="1 4 1 10 7 10" />
    <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
  </svg>
);

const Zap: React.FC<{ className?: string }> = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
);

const Waves: React.FC<{ className?: string }> = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
    <path d="M12 3v.01M19.07 4.93l-.707.707M20.97 11h-.01M19.07 19.07l-.707-.707M12 21v.01M4.93 19.07l-.707-.707M3 12h.01M4.93 4.93l-.707.707" />
  </svg>
);

const MoveRight: React.FC<{ className?: string }> = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
    <path d="M13 5l7 7-7 7M3 12h17" />
  </svg>
);

const ShieldAlert: React.FC<{ className?: string }> = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <line x1={12} y1={8} x2={12} y2={12} />
    <line x1={12} y1={16} x2={12.01} y2={16} />
  </svg>
);

const W = 900;
const H = 380;
const PAD = 44;
const XMIN = 0;
const XMAX = 6;
const DT = 0.03;

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function samplePoisson(rateDt: number) {
  // Good enough for small rateDt in visualization.
  return Math.random() < rateDt ? 1 : 0;
}

function xToPx(x: number) {
  return PAD + ((x - XMIN) / (XMAX - XMIN)) * (W - 2 * PAD);
}

export default function JumpDiffusionVisualizer() {
  const [running, setRunning] = useState(true);
  const [time, setTime] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const [particles, setParticles] = useState<Particle[]>(() =>
    Array.from({ length: 6 }, (_, i) => ({
      x: 1 + 0.5 * i,
      alive: true,
      color: ["#2563eb", "#7c3aed", "#0f766e", "#dc2626", "#ca8a04", "#0891b2"][i % 6],
      trail: [1 + 0.5 * i],
      jumpFlash: 0,
    }))
  );

  const [drift, setDrift] = useState(0.8);
  const [diffusion, setDiffusion] = useState(0.45);
  const [jumpRate, setJumpRate] = useState(0.65);
  const [jumpScale, setJumpScale] = useState(0.8);
  const [boundary, setBoundary] = useState(5.2);

  const aliveCount = particles.filter((p: Particle) => p.alive).length;
  const exitCount = particles.length - aliveCount;

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => {
      setTime((t: number) => t + DT);
      setParticles((prev: Particle[]) =>
        prev.map((p: Particle) => {
          if (!p.alive) {
            return { ...p, trail: [...p.trail, p.x].slice(-180), jumpFlash: Math.max(0, p.jumpFlash - 1) };
          }

          const driftStep = drift * DT;
          const diffusionStep = diffusion * Math.sqrt(DT) * randn();
          const jumpHappens = samplePoisson(jumpRate * DT);
          const jumpStep = jumpHappens ? jumpScale * randn() : 0;
          const nextX = p.x + driftStep + diffusionStep + jumpStep;
          const exited = nextX >= boundary || nextX <= XMIN;
          const safeX = exited ? clamp(nextX, XMIN, XMAX) : nextX;
          return {
            ...p,
            x: safeX,
            alive: !exited,
            trail: [...p.trail, safeX].slice(-180),
            jumpFlash: jumpHappens ? 10 : Math.max(0, p.jumpFlash - 1),
          };
        })
      );
      setHistory((h: number[]) => [...h, aliveCount / particles.length].slice(-220));
    }, 30);
    return () => window.clearInterval(id);
  }, [running, drift, diffusion, jumpRate, jumpScale, boundary, aliveCount, particles.length]);

  const reset = () => {
    setRunning(false);
    setTime(0);
    setParticles(
      Array.from({ length: 6 }, (_, i) => ({
        x: 1 + 0.5 * i,
        alive: true,
        color: ["#2563eb", "#7c3aed", "#0f766e", "#dc2626", "#ca8a04", "#0891b2"][i % 6],
        trail: [1 + 0.5 * i],
        jumpFlash: 0,
      }))
    );
    setHistory([]);
  };

  const plotPoints = useMemo(() => {
    const maxLen = Math.max(1, history.length - 1);
    return history.map((y: number, i: number) => ({
      x: PAD + (i / maxLen) * (W - 2 * PAD),
      y: 300 - y * 150,
    }));
  }, [history]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="space-y-2">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Jump-Diffusion Electron Visualizer</h1>
          <p className="text-slate-300 max-w-3xl">
            See how an electron-like particle moves under <span className="text-sky-300">drift</span>,
            <span className="text-violet-300"> diffusion</span>, and occasional <span className="text-amber-300">jumps</span>,
            with an absorbing boundary that acts like an exit.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.35fr_0.9fr]">
          <Card className="bg-slate-900/80 border-slate-800 shadow-2xl">
            <CardHeader className="space-y-3">
              <div className="flex flex-wrap items-center gap-3 justify-between">
                <CardTitle className="text-xl">Visual state</CardTitle>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary" className="bg-slate-800 text-slate-100">t = {time.toFixed(2)}</Badge>
                  <Badge variant="secondary" className="bg-slate-800 text-slate-100">alive = {aliveCount}</Badge>
                  <Badge variant="secondary" className="bg-slate-800 text-slate-100">exited = {exitCount}</Badge>
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={() => setRunning((r: boolean) => !r)} className="rounded-2xl">
                  {running ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
                  {running ? "Pause" : "Play"}
                </Button>
                <Button onClick={reset} variant="secondary" className="rounded-2xl bg-slate-800 text-slate-100 hover:bg-slate-700">
                  <RotateCcw className="mr-2 h-4 w-4" /> Reset
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="relative rounded-3xl border border-slate-800 bg-slate-950 overflow-hidden">
                <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[420px] block">
                  <defs>
                    <linearGradient id="bgGrad" x1="0" x2="1" y1="0" y2="0">
                      <stop offset="0%" stopColor="#08111f" />
                      <stop offset="100%" stopColor="#0f172a" />
                    </linearGradient>
                  </defs>
                  <rect x="0" y="0" width={W} height={H} fill="url(#bgGrad)" />

                  {/* axis */}
                  <line x1={PAD} y1={300} x2={W - PAD} y2={300} stroke="#475569" strokeWidth="2" />
                  <text x={PAD} y={324} fill="#94a3b8" fontSize="12">0</text>
                  <text x={xToPx(3)} y={324} fill="#94a3b8" fontSize="12" textAnchor="middle">mid</text>
                  <text x={xToPx(XMAX)} y={324} fill="#94a3b8" fontSize="12" textAnchor="end">boundary</text>

                  {/* boundary */}
                  <line x1={xToPx(boundary)} y1={70} x2={xToPx(boundary)} y2={300} stroke="#f59e0b" strokeWidth="4" strokeDasharray="8 6" />
                  <text x={xToPx(boundary)} y={60} fill="#fbbf24" fontSize="13" textAnchor="middle">absorbing boundary</text>

                  {/* drift direction */}
                  <path d={`M ${PAD + 20} 70 L ${PAD + 110} 70`} stroke="#38bdf8" strokeWidth="4" strokeLinecap="round" />
                  <polygon points={`${PAD + 110},70 ${PAD + 98},64 ${PAD + 98},76`} fill="#38bdf8" />
                  <text x={PAD + 20} y={56} fill="#7dd3fc" fontSize="13">drift pushes right</text>

                  {/* trajectory curves */}
                  {particles.map((p, idx) => {
                    const d = p.trail
                      .map((x, i) => `${i === 0 ? "M" : "L"} ${xToPx(x)} ${110 + idx * 22}`)
                      .join(" ");
                    return (
                      <g key={idx}>
                        <path d={d} fill="none" stroke={p.color} strokeWidth="2.5" opacity="0.5" />
                        {p.trail.length > 0 && (
                          <circle cx={xToPx(p.x)} cy={110 + idx * 22} r={p.jumpFlash > 0 ? 9 : 7} fill={p.color} opacity={p.alive ? 1 : 0.35} />
                        )}
                        {p.jumpFlash > 0 && <circle cx={xToPx(p.x)} cy={110 + idx * 22} r={16} fill={p.color} opacity="0.18" />}
                      </g>
                    );
                  })}

                  {/* legend */}
                  <g transform="translate(52,350)">
                    <rect x="0" y="0" width="14" height="14" rx="4" fill="#38bdf8" />
                    <text x="22" y="12" fill="#cbd5e1" fontSize="13">drift</text>
                    <rect x="82" y="0" width="14" height="14" rx="4" fill="#a78bfa" />
                    <text x="104" y="12" fill="#cbd5e1" fontSize="13">diffusion</text>
                    <rect x="188" y="0" width="14" height="14" rx="4" fill="#fbbf24" />
                    <text x="210" y="12" fill="#cbd5e1" fontSize="13">jump / exit</text>
                  </g>
                </svg>
              </div>
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card className="bg-slate-900/80 border-slate-800 shadow-2xl">
              <CardHeader>
                <CardTitle className="text-xl">Equation decoded</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-sm text-slate-300">
                <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4 space-y-3">
                  <div className="font-mono text-slate-100">dX<tspan className="align-super text-xs">t</tspan> = b(X<tspan className="align-super text-xs">t</tspan>)dt + σ(X<tspan className="align-super text-xs">t</tspan>)dW<tspan className="align-super text-xs">t</tspan> + dJ<tspan className="align-super text-xs">t</tspan></div>
                  <div className="grid gap-3">
                    <div className="flex items-start gap-3"><MoveRight className="mt-0.5 h-4 w-4 text-sky-300" /><span><b className="text-slate-100">Drift:</b> the average push. In the paper this comes from the electric field minus collisional drag.</span></div>
                    <div className="flex items-start gap-3"><Waves className="mt-0.5 h-4 w-4 text-violet-300" /><span><b className="text-slate-100">Diffusion:</b> lots of tiny random kicks, like many small collisions adding up.</span></div>
                    <div className="flex items-start gap-3"><Zap className="mt-0.5 h-4 w-4 text-amber-300" /><span><b className="text-slate-100">Jumps:</b> rare big kicks, like a sudden collision that changes momentum a lot.</span></div>
                    <div className="flex items-start gap-3"><ShieldAlert className="mt-0.5 h-4 w-4 text-rose-300" /><span><b className="text-slate-100">Boundary:</b> once the particle reaches the wall, it exits and stops moving.</span></div>
                  </div>
                </div>
                <p>Use the sliders to see how each term changes the motion. The model is 1D here, but the same ideas extend to the runaway-electron phase space in the paper.</p>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/80 border-slate-800 shadow-2xl">
              <CardHeader>
                <CardTitle className="text-xl">Controls</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                {[
                  { label: "Drift", value: drift, set: setDrift, min: 0, max: 2, step: 0.05, icon: <MoveRight className="h-4 w-4 text-sky-300" /> },
                  { label: "Diffusion", value: diffusion, set: setDiffusion, min: 0, max: 1.25, step: 0.01, icon: <Waves className="h-4 w-4 text-violet-300" /> },
                  { label: "Jump rate", value: jumpRate, set: setJumpRate, min: 0, max: 3, step: 0.05, icon: <Zap className="h-4 w-4 text-amber-300" /> },
                  { label: "Jump size", value: jumpScale, set: setJumpScale, min: 0, max: 2, step: 0.05, icon: <Zap className="h-4 w-4 text-amber-300" /> },
                  { label: "Boundary", value: boundary, set: setBoundary, min: 2.5, max: 6, step: 0.05, icon: <ShieldAlert className="h-4 w-4 text-rose-300" /> },
                ].map((ctrl) => (
                  <div key={ctrl.label} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2 text-slate-200">
                        {ctrl.icon}
                        {ctrl.label}
                      </div>
                      <span className="font-mono text-slate-400">{ctrl.value.toFixed(2)}</span>
                    </div>
                    <Slider
                      value={[ctrl.value]}
                      min={ctrl.min}
                      max={ctrl.max}
                      step={ctrl.step}
                      onValueChange={(v) => ctrl.set(v[0])}
                    />
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card className="bg-slate-900/80 border-slate-800 shadow-2xl">
              <CardHeader>
                <CardTitle className="text-xl">Survival over time</CardTitle>
              </CardHeader>
              <CardContent>
                <svg viewBox="0 0 900 320" className="w-full h-64 block rounded-2xl border border-slate-800 bg-slate-950">
                  <rect x="0" y="0" width="900" height="320" fill="#020617" />
                  <line x1={PAD} y1={40} x2={PAD} y2={280} stroke="#334155" />
                  <line x1={PAD} y1={280} x2={860} y2={280} stroke="#334155" />
                  <text x={18} y={46} fill="#94a3b8" fontSize="12">1.0</text>
                  <text x={18} y={284} fill="#94a3b8" fontSize="12">0</text>
                  <text x={820} y={304} fill="#94a3b8" fontSize="12">time</text>
                  <text x={10} y={22} fill="#94a3b8" fontSize="12">alive fraction</text>
                  {plotPoints.length > 1 && (
                    <path
                      d={plotPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ")}
                      fill="none"
                      stroke="#22c55e"
                      strokeWidth="3"
                    />
                  )}
                  {plotPoints.length === 0 && <text x="50%" y="50%" textAnchor="middle" fill="#64748b">Run the simulation to see survival decay.</text>}
                </svg>
                <div className="mt-3 text-sm text-slate-300">
                  The line drops when particles hit the boundary and exit.
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
