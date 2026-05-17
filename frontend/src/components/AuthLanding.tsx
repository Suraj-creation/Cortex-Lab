"use client";

import { ArrowRight, BrainCircuit, CheckCircle2, Cloud, Database, LockKeyhole, Mic2, ShieldCheck, Sparkles } from "lucide-react";

import { AuthStatus } from "@/lib/types";

interface AuthLandingProps {
  status: AuthStatus | null;
  loading: boolean;
  error: string;
  onSignIn: () => void;
  onContinueLocal: () => void;
}

const FEATURE_CARDS = [
  {
    icon: BrainCircuit,
    title: "Personal RAG memory",
    body: "Every useful turn becomes searchable context for agents, wiki pages, graph links, and future retrieval.",
  },
  {
    icon: Mic2,
    title: "Eva voice companion",
    body: "A conversational surface for ambient capture, wake-word flows, and speech-to-speech assistance.",
  },
  {
    icon: Database,
    title: "Local-first backup",
    body: "Your device remains authoritative while Supabase and Google Drive provide encrypted recovery paths.",
  },
];

export function AuthLanding({
  status,
  loading,
  error,
  onSignIn,
  onContinueLocal,
}: AuthLandingProps) {
  const googleReady = Boolean(status?.enabled && status.google.configured);
  const backupReady = Boolean(
    status?.backup.supabase_postgres_configured
      || status?.backup.supabase_storage_configured
      || status?.backup.google_drive_configured,
  );
  const signedIn = Boolean(status?.authenticated);

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#f7f3ea] text-stone-950">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_15%,rgba(234,179,8,0.18),transparent_30%),radial-gradient(circle_at_80%_5%,rgba(15,118,110,0.16),transparent_28%),linear-gradient(135deg,#fffaf0_0%,#eef6f1_48%,#f8fafc_100%)]" />
      <div className="absolute left-[-8rem] top-24 h-72 w-72 rounded-full bg-white/60 blur-3xl" />
      <div className="absolute bottom-[-10rem] right-[-6rem] h-96 w-96 rounded-full bg-emerald-200/50 blur-3xl" />

      <section className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col px-6 py-8 lg:px-10">
        <nav className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-2xl border border-white/70 bg-white/70 shadow-[0_18px_50px_rgba(15,23,42,0.12)] backdrop-blur-xl">
              <BrainCircuit className="h-5 w-5 text-emerald-700" />
            </div>
            <div>
              <p className="text-sm font-black uppercase tracking-[0.28em] text-stone-900">Cortex Lab</p>
              <p className="text-xs font-semibold text-stone-500">Eva memory operating system</p>
            </div>
          </div>
          <div className="hidden items-center gap-2 rounded-full border border-white/70 bg-white/65 px-4 py-2 text-xs font-bold text-stone-600 shadow-sm backdrop-blur-xl sm:flex">
            <ShieldCheck className="h-4 w-4 text-emerald-700" />
            Google OAuth + Supabase ready
          </div>
        </nav>

        <div className="grid flex-1 items-center gap-10 py-10 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="max-w-3xl">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-amber-200/80 bg-amber-50/80 px-4 py-2 text-xs font-black uppercase tracking-[0.2em] text-amber-800 shadow-sm">
              <Sparkles className="h-4 w-4" />
              Auth-first memory workspace
            </div>

            <h1 className="text-5xl font-black tracking-[-0.06em] text-stone-950 sm:text-6xl lg:text-7xl">
              Sign in once.
              <span className="block text-emerald-800">Let Eva remember the work.</span>
            </h1>

            <p className="mt-6 max-w-2xl text-lg font-medium leading-8 text-stone-600">
              Cortex Lab connects your agent chat, ambient listening, personal wiki,
              knowledge graph, RAG dashboard, and document memory into one private
              identity. Google OAuth unlocks cloud backup while the app keeps the
              local-first memory workflow intact.
            </p>

            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <button
                type="button"
                onClick={onSignIn}
                disabled={!googleReady || loading}
                className="group inline-flex items-center justify-center gap-3 rounded-2xl bg-stone-950 px-6 py-4 text-sm font-black text-white shadow-[0_20px_50px_rgba(15,23,42,0.25)] transition hover:-translate-y-0.5 hover:bg-emerald-900 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Checking auth..." : "Continue with Google"}
                <ArrowRight className="h-4 w-4 transition group-hover:translate-x-1" />
              </button>
              <button
                type="button"
                onClick={onContinueLocal}
                className="inline-flex items-center justify-center rounded-2xl border border-white/80 bg-white/70 px-6 py-4 text-sm font-black text-stone-700 shadow-sm backdrop-blur-xl transition hover:-translate-y-0.5 hover:bg-white"
              >
                Use local-first mode
              </button>
            </div>

            {!googleReady ? (
              <p className="mt-4 rounded-2xl border border-amber-200 bg-amber-50/80 p-4 text-sm font-semibold leading-6 text-amber-900">
                Google OAuth is not configured on the active backend yet. Add the
                backend auth environment variables in Render or local `.env`, then
                this button will start the full OAuth flow.
              </p>
            ) : null}
            {error ? (
              <p className="mt-4 rounded-2xl border border-rose-200 bg-rose-50/80 p-4 text-sm font-semibold leading-6 text-rose-800">
                {error}
              </p>
            ) : null}
          </div>

          <div className="relative">
            <div className="rounded-[2.25rem] border border-white/80 bg-white/65 p-5 shadow-[0_30px_90px_rgba(15,23,42,0.18)] backdrop-blur-2xl">
              <div className="rounded-[1.75rem] border border-stone-200/70 bg-[#fffdf7] p-5">
                <div className="flex items-center justify-between border-b border-stone-200 pb-5">
                  <div>
                    <p className="text-xs font-black uppercase tracking-[0.24em] text-stone-400">Session identity</p>
                    <p className="mt-2 text-2xl font-black tracking-tight text-stone-950">
                      {signedIn ? status?.user?.name || status?.user?.email : "Awaiting Google sign-in"}
                    </p>
                  </div>
                  <div className="grid h-12 w-12 place-items-center rounded-2xl bg-emerald-100 text-emerald-800">
                    <LockKeyhole className="h-5 w-5" />
                  </div>
                </div>

                <div className="mt-5 grid gap-3">
                  <StatusRow label="Google OAuth" ready={googleReady} detail={googleReady ? "Configured" : "Missing backend env"} />
                  <StatusRow label="Supabase / Drive backup" ready={backupReady} detail={backupReady ? "Backup target online" : "Waiting for cloud env"} />
                  <StatusRow label="Local-first memory" ready detail="Available before and after sign-in" />
                </div>

                <div className="mt-6 grid gap-3">
                  {FEATURE_CARDS.map((feature) => {
                    const Icon = feature.icon;
                    return (
                      <div key={feature.title} className="rounded-2xl border border-stone-200 bg-white/80 p-4">
                        <div className="flex gap-3">
                          <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-stone-950 text-white">
                            <Icon className="h-4 w-4" />
                          </div>
                          <div>
                            <h2 className="text-sm font-black text-stone-900">{feature.title}</h2>
                            <p className="mt-1 text-sm font-medium leading-6 text-stone-500">{feature.body}</p>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

function StatusRow({ label, ready, detail }: { label: string; ready: boolean; detail: string }) {
  return (
    <div className="flex items-center justify-between rounded-2xl border border-stone-200 bg-white/70 px-4 py-3">
      <div>
        <p className="text-sm font-black text-stone-900">{label}</p>
        <p className="text-xs font-semibold text-stone-500">{detail}</p>
      </div>
      {ready ? (
        <CheckCircle2 className="h-5 w-5 text-emerald-700" />
      ) : (
        <Cloud className="h-5 w-5 text-amber-600" />
      )}
    </div>
  );
}
