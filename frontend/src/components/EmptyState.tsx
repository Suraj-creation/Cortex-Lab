"use client";

import { useEffect, useState, useRef } from "react";
import {
  AlertCircle,
  ArrowRight,
  Brain,
  Database,
  GitBranch,
  Search,
  Shield,
  Layers,
  Cpu,
  Network,
  Sparkles,
} from "lucide-react";

interface Props {
  isOnline: boolean;
  onSuggestion: (text: string) => void;
}

/* ── 9-Layer Architecture Labels ────────────────────────────────── */
const layers = [
  "Input Acquisition",
  "Memory Ingestion",
  "Multi-Rep Storage",
  "Query Intelligence",
  "Agent Orchestration",
  "Hybrid Retrieval",
  "Post-Retrieval",
  "Self-Reflective Gen",
  "Memory Evolution",
];

/* ── Capability pills reflecting vision ─────────────────────────── */
const capabilities = [
  { label: "9-Layer Agentic RAG", icon: Layers },
  { label: "Knowledge Graph", icon: GitBranch },
  { label: "Self-Reflective Generation", icon: Brain },
  { label: "Hybrid Retrieval", icon: Search },
  { label: "Memory Evolution", icon: Database },
  { label: "Privacy-First", icon: Shield },
];

/* ── Suggestion cards aligned with vision ───────────────────────── */
const suggestions = [
  {
    icon: Brain,
    label: "Personal Memory",
    text: "What have I been working on recently? Trace back through my conversations and identify the key themes and decisions I've made.",
    color: "text-violet-600",
    bg: "bg-violet-50/80 border-violet-200/60 hover:border-violet-400 hover:bg-violet-100/80",
    accent: "from-violet-500 to-purple-600",
  },
  {
    icon: Network,
    label: "Multi-Hop Reasoning",
    text: "How are my career goals connected to the projects I've been building? Use the knowledge graph to trace the causal chain.",
    color: "text-cyan-600",
    bg: "bg-cyan-50/80 border-cyan-200/60 hover:border-cyan-400 hover:bg-cyan-100/80",
    accent: "from-cyan-500 to-blue-600",
  },
  {
    icon: Sparkles,
    label: "Causal Analysis",
    text: "What led me to change my approach on the last project? Analyze the timeline of my decisions and identify the turning points.",
    color: "text-amber-600",
    bg: "bg-amber-50/80 border-amber-200/60 hover:border-amber-400 hover:bg-amber-100/80",
    accent: "from-amber-500 to-orange-600",
  },
  {
    icon: Cpu,
    label: "Deep Synthesis",
    text: "Synthesize everything you know about me — my skills, interests, and goals — into a coherent profile. What patterns emerge?",
    color: "text-emerald-600",
    bg: "bg-emerald-50/80 border-emerald-200/60 hover:border-emerald-400 hover:bg-emerald-100/80",
    accent: "from-emerald-500 to-teal-600",
  },
];

/* ── Neural network background (SVG nodes + connections) ────────── */
function NeuralBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let particles: Array<{
      x: number; y: number; vx: number; vy: number;
      size: number; opacity: number; hue: number;
    }> = [];

    const resize = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    const initParticles = () => {
      const count = Math.min(60, Math.floor(canvas.offsetWidth * canvas.offsetHeight / 12000));
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * canvas.offsetWidth,
        y: Math.random() * canvas.offsetHeight,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        size: Math.random() * 2 + 1,
        opacity: Math.random() * 0.4 + 0.1,
        hue: Math.random() > 0.5 ? 240 : 270,
      }));
    };

    const draw = () => {
      const w = canvas.offsetWidth;
      const h = canvas.offsetHeight;
      ctx.clearRect(0, 0, w, h);

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            const alpha = (1 - dist / 120) * 0.08;
            ctx.strokeStyle = `hsla(250, 60%, 65%, ${alpha})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw particles
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 70%, 65%, ${p.opacity})`;
        ctx.fill();
      }

      animationId = requestAnimationFrame(draw);
    };

    resize();
    initParticles();
    draw();

    window.addEventListener("resize", () => { resize(); initParticles(); });
    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ opacity: 0.6 }}
    />
  );
}

/* ── Animated 9-Layer Pipeline ──────────────────────────────────── */
function LayerPipeline() {
  const [activeLayer, setActiveLayer] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveLayer((prev) => (prev + 1) % 9);
    }, 1800);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="layer-pipeline flex items-center gap-1 py-3 px-2 overflow-x-auto">
      {layers.map((layer, i) => (
        <div key={layer} className="flex items-center">
          <div
            className={`layer-node relative px-2.5 py-1.5 rounded-lg text-[9px] font-semibold tracking-wide whitespace-nowrap transition-all duration-500 cursor-default ${
              i === activeLayer
                ? "bg-indigo-500 text-white shadow-lg shadow-indigo-300/40 scale-110"
                : i < activeLayer
                ? "bg-indigo-100 text-indigo-600 border border-indigo-200"
                : "bg-slate-100 text-slate-400 border border-slate-200"
            }`}
          >
            <span className="relative z-10">L{i}</span>
            {i === activeLayer && (
              <div className="absolute inset-0 rounded-lg bg-indigo-400/30 animate-ping-slow" />
            )}
          </div>
          {i < 8 && (
            <div className={`w-3 h-px mx-0.5 transition-colors duration-500 ${
              i < activeLayer ? "bg-indigo-400" : "bg-slate-200"
            }`}>
              <div
                className={`h-full transition-all duration-500 ${
                  i === activeLayer ? "bg-indigo-500 animate-data-flow" : ""
                }`}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/* ── Main EmptyState Component ──────────────────────────────────── */
export function EmptyState({ isOnline, onSuggestion }: Props) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="relative flex flex-col items-center justify-center min-h-[70vh] px-4 overflow-hidden">
      {/* Neural network animated background */}
      <NeuralBackground />

      {/* Ambient gradient orbs */}
      <div className="hero-orb-1 absolute top-10 -left-20 w-72 h-72 pointer-events-none" />
      <div className="hero-orb-2 absolute bottom-10 -right-20 w-80 h-80 pointer-events-none" />
      <div className="hero-orb-3 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 pointer-events-none" />

      {/* Hero Section */}
      <div className={`relative z-10 mb-8 text-center transition-all duration-1000 ${
        mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
      }`}>
        {/* Animated brain icon */}
        <div className="relative inline-flex items-center justify-center mb-6">
          {/* Pulsing rings */}
          <div className="absolute w-28 h-28 rounded-full border border-indigo-300/20 animate-pulse-ring" />
          <div className="absolute w-36 h-36 rounded-full border border-violet-300/10 animate-pulse-ring-delayed" />

          {/* Core icon */}
          <div className="relative h-20 w-20 rounded-3xl cortex-icon-bg border border-indigo-300/30 flex items-center justify-center shadow-2xl shadow-indigo-300/30">
            <svg width="42" height="42" viewBox="0 0 48 48" fill="none" className="cortex-brain-svg">
              {/* Outer neural ring */}
              <circle cx="24" cy="24" r="18" stroke="white" strokeWidth="0.8" opacity="0.15" strokeDasharray="4 3" className="animate-spin-slow" />
              <circle cx="24" cy="24" r="14" stroke="white" strokeWidth="0.8" opacity="0.25" strokeDasharray="3 4" className="animate-spin-reverse" />
              {/* Neural nodes */}
              <circle cx="24" cy="10" r="1.5" fill="white" opacity="0.5" className="animate-pulse-node" />
              <circle cx="38" cy="24" r="1.5" fill="white" opacity="0.5" className="animate-pulse-node-delayed" />
              <circle cx="24" cy="38" r="1.5" fill="white" opacity="0.5" className="animate-pulse-node" />
              <circle cx="10" cy="24" r="1.5" fill="white" opacity="0.5" className="animate-pulse-node-delayed" />
              {/* Connection lines */}
              <line x1="24" y1="10" x2="24" y2="19" stroke="white" strokeWidth="0.6" opacity="0.3" />
              <line x1="38" y1="24" x2="29" y2="24" stroke="white" strokeWidth="0.6" opacity="0.3" />
              <line x1="24" y1="38" x2="24" y2="29" stroke="white" strokeWidth="0.6" opacity="0.3" />
              <line x1="10" y1="24" x2="19" y2="24" stroke="white" strokeWidth="0.6" opacity="0.3" />
              {/* Core brain */}
              <circle cx="24" cy="24" r="5" fill="white" opacity="0.9" />
              <circle cx="24" cy="24" r="7" stroke="white" strokeWidth="1" opacity="0.4" />
              {/* Synaptic flashes */}
              <circle cx="16" cy="16" r="0.8" fill="white" opacity="0.6" className="animate-synapse-1" />
              <circle cx="32" cy="16" r="0.8" fill="white" opacity="0.6" className="animate-synapse-2" />
              <circle cx="32" cy="32" r="0.8" fill="white" opacity="0.6" className="animate-synapse-3" />
              <circle cx="16" cy="32" r="0.8" fill="white" opacity="0.6" className="animate-synapse-4" />
            </svg>
            <div className="absolute -inset-px rounded-3xl bg-gradient-to-b from-white/20 to-transparent" />
          </div>
        </div>

        {/* Title */}
        <h2 className="text-4xl font-extrabold cortex-hero-text mb-3 tracking-tight">
          Cortex Lab
        </h2>
        <p className="text-sm text-slate-500 max-w-md mx-auto leading-relaxed">
          <span className="text-slate-700 font-medium">Qwen3.5-9B Opus</span> reasoning model ·
          Claude 4.6 Opus reasoning distilled ·
          Advanced chain-of-thought reasoning
        </p>
        <p className="text-sm text-slate-400 max-w-md mx-auto leading-relaxed mt-1">
          9-Layer Agentic RAG · 25+ Research Techniques · Continuous Memory &amp; Reasoning
        </p>

        {/* Animated Layer Pipeline */}
        <div className={`mt-5 transition-all duration-1000 delay-300 ${
          mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
        }`}>
          <LayerPipeline />
        </div>

        {/* Capability pills */}
        <div className={`flex flex-wrap items-center justify-center gap-2 mt-5 transition-all duration-1000 delay-500 ${
          mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
        }`}>
          {capabilities.map(({ label, icon: Icon }, i) => (
            <span
              key={label}
              className="capability-pill group flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/70 border border-slate-200/80 text-[10px] text-slate-600 font-medium backdrop-blur-sm hover:border-indigo-300 hover:text-indigo-600 hover:bg-indigo-50/50 transition-all duration-300 cursor-default"
              style={{ animationDelay: `${i * 100}ms` }}
            >
              <Icon size={10} className="text-indigo-400 group-hover:text-indigo-600 transition-colors" />
              {label}
            </span>
          ))}
        </div>
      </div>

      {/* Status */}
      {!isOnline && (
        <div className="relative z-10 mb-6 flex items-center gap-2.5 rounded-xl bg-amber-50/90 border border-amber-200 px-4 py-3 text-sm text-amber-700 backdrop-blur-sm">
          <AlertCircle size={16} />
          <span>
            Model is loading. Start the backend with{" "}
            <code className="bg-amber-100 px-1.5 py-0.5 rounded-md text-xs font-mono border border-amber-200">
              python server.py
            </code>
          </span>
        </div>
      )}

      {/* Suggestion Cards */}
      <div className={`relative z-10 grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl w-full transition-all duration-1000 delay-700 ${
        mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
      }`}>
        {suggestions.map((s, i) => (
          <button
            key={s.label}
            onClick={() => onSuggestion(s.text)}
            disabled={!isOnline}
            className={`suggestion-card group relative flex flex-col items-start gap-3 rounded-2xl border p-5 text-left disabled:opacity-30 disabled:cursor-not-allowed overflow-hidden ${s.bg}`}
            style={{ animationDelay: `${800 + i * 120}ms` }}
          >
            {/* Hover gradient overlay */}
            <div className={`absolute inset-0 bg-gradient-to-br ${s.accent} opacity-0 group-hover:opacity-[0.04] transition-opacity duration-500`} />

            <div className="relative flex items-center gap-2.5 w-full">
              <div className={`p-2 rounded-xl bg-white/80 border border-white/60 shadow-sm ${s.color} group-hover:scale-110 transition-transform duration-300`}>
                <s.icon size={15} />
              </div>
              <span className={`text-xs font-bold tracking-wide uppercase ${s.color}`}>
                {s.label}
              </span>
              <ArrowRight
                size={13}
                className="ml-auto text-slate-300 group-hover:text-indigo-400 group-hover:translate-x-1 transition-all duration-300"
              />
            </div>
            <p className="relative text-xs leading-relaxed text-slate-500 group-hover:text-slate-700 transition-colors duration-300 line-clamp-3">
              {s.text}
            </p>

            {/* Bottom shine effect on hover */}
            <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-indigo-300/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          </button>
        ))}
      </div>

      {/* Architecture description */}
      <div className={`relative z-10 mt-8 text-center max-w-lg transition-all duration-1000 delay-1000 ${
        mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
      }`}>
        <div className="flex items-center justify-center gap-6 text-[10px] text-slate-400">
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            GraphRAG
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" style={{ animationDelay: "0.5s" }} />
            Self-RAG
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" style={{ animationDelay: "1s" }} />
            CRAG
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" style={{ animationDelay: "1.5s" }} />
            FLARE
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400 animate-pulse" style={{ animationDelay: "2s" }} />
            RAPTOR
          </span>
        </div>
        <p className="mt-2.5 text-[11px] text-slate-400 leading-relaxed">
          Persistent cognitive memory · Causal reasoning · Temporal understanding · Belief evolution tracking
        </p>
      </div>
    </div>
  );
}
