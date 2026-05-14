"use client";

import { useEffect, useState } from "react";
import { PanelLeftOpen, Cpu, FlaskConical, LogIn, LogOut, CloudUpload } from "lucide-react";

import { buildGoogleAuthStartUrl, getAuthStatus, logoutAuth, runBackup } from "@/lib/api";
import { ModelStatus } from "@/lib/types";

interface Props {
  modelStatus: ModelStatus;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

export function Header({ modelStatus, sidebarOpen, onToggleSidebar }: Props) {
  const [authState, setAuthState] = useState({
    authenticated: false,
    enabled: false,
    name: "",
  });
  const [authBusy, setAuthBusy] = useState(false);
  const [backupBusy, setBackupBusy] = useState(false);

  useEffect(() => {
    let mounted = true;

    const loadAuth = async () => {
      try {
        const status = await getAuthStatus();
        if (!mounted) {
          return;
        }
        setAuthState({
          authenticated: status.authenticated,
          enabled: status.enabled && status.google.configured,
          name: status.user?.name || status.user?.email || "",
        });
      } catch {
        if (!mounted) {
          return;
        }
        setAuthState({
          authenticated: false,
          enabled: false,
          name: "",
        });
      }
    };

    void loadAuth();

    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      if (url.searchParams.get("auth") === "success") {
        url.searchParams.delete("auth");
        window.history.replaceState({}, "", url.toString());
      }
    }

    return () => {
      mounted = false;
    };
  }, []);

  const handleSignIn = () => {
    if (typeof window === "undefined") {
      return;
    }
    window.location.href = buildGoogleAuthStartUrl({ nextUrl: window.location.href });
  };

  const handleLogout = async () => {
    setAuthBusy(true);
    try {
      await logoutAuth();
      setAuthState((prev) => ({
        ...prev,
        authenticated: false,
        name: "",
      }));
    } finally {
      setAuthBusy(false);
    }
  };

  const handleBackup = async () => {
    setBackupBusy(true);
    try {
      await runBackup({
        platform: "web",
        captured_at: new Date().toISOString(),
      });
    } finally {
      setBackupBusy(false);
    }
  };

  const statusColor = modelStatus.model_loaded
    ? "text-emerald-600"
    : modelStatus.status === "loading"
      ? "text-amber-600"
      : "text-red-500";

  const statusBg = modelStatus.model_loaded
    ? "bg-emerald-50 border-emerald-200"
    : modelStatus.status === "loading"
      ? "bg-amber-50 border-amber-200"
      : "bg-red-50 border-red-200";

  const statusDot = modelStatus.model_loaded
    ? "bg-emerald-500"
    : modelStatus.status === "loading"
      ? "bg-amber-500"
      : "bg-red-500";

  const isFT = modelStatus.model_info.fine_tuned;
  const stages = modelStatus.model_info.training_stages_completed ?? 0;

  return (
    <header className="relative flex items-center justify-between border-b border-slate-300/80 bg-white/90 backdrop-blur-2xl px-5 py-3.5">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-slate-200 to-transparent" />

      <div className="flex items-center gap-3">
        {!sidebarOpen && (
          <button
            onClick={onToggleSidebar}
            className="rounded-lg p-1.5 text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-all duration-200"
          >
            <PanelLeftOpen size={18} />
          </button>
        )}
        <div className="flex items-center gap-2.5">
          <h1 className="text-sm font-semibold text-slate-800 tracking-tight">
            Cortex Lab
          </h1>
          <span className="rounded-md bg-indigo-50 border border-indigo-200 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-600 tracking-wide">
            7B
          </span>
          {isFT && (
            <span className="flex items-center gap-1 rounded-md bg-violet-50 border border-violet-200 px-1.5 py-0.5 text-[10px] font-semibold text-violet-600 tracking-wide">
              <FlaskConical size={9} />
              Fine-Tuned · {stages}/15
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2">
        {authState.authenticated ? (
          <>
            <button
              onClick={handleBackup}
              disabled={backupBusy}
              className="flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3 py-2 text-[11px] font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-60"
            >
              <CloudUpload size={13} />
              {backupBusy ? "Backing up" : "Backup"}
            </button>
            <button
              onClick={handleLogout}
              disabled={authBusy}
              className="flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3 py-2 text-[11px] font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-60"
              title={authState.name || "Signed in"}
            >
              <LogOut size={13} />
              {authState.name ? authState.name.split(" ")[0] : "Sign out"}
            </button>
          </>
        ) : authState.enabled ? (
          <button
            onClick={handleSignIn}
            className="flex items-center gap-1.5 rounded-xl border border-indigo-200 bg-indigo-50 px-3 py-2 text-[11px] font-medium text-indigo-700 hover:bg-indigo-100"
          >
            <LogIn size={13} />
            Sign in
          </button>
        ) : null}

        <div
          className={`flex items-center gap-2.5 rounded-xl border px-3.5 py-2 transition-all duration-300 ${statusBg}`}
        >
          <div className="relative">
            <div className={`h-2 w-2 rounded-full ${statusDot}`} />
            {modelStatus.model_loaded && (
              <div className={`absolute inset-0 h-2 w-2 rounded-full ${statusDot} animate-ping opacity-40`} />
            )}
          </div>
          <span className={`text-[11px] font-medium ${statusColor}`}>
            {modelStatus.model_loaded
              ? "Online"
              : modelStatus.status === "loading"
                ? "Loading Model..."
                : "Offline"}
          </span>
          {modelStatus.model_info.quantization && modelStatus.model_info.quantization !== "N/A" && (
            <span className="text-[10px] text-slate-500 border-l border-slate-300 pl-2.5 ml-0.5">
              {modelStatus.model_info.quantization}
            </span>
          )}
          {modelStatus.model_info.device && modelStatus.model_info.device !== "N/A" && (
            <span className="flex items-center gap-1 text-[10px] text-slate-500 border-l border-slate-300 pl-2.5 ml-0.5">
              {modelStatus.model_info.device === "API" ? (
                <>
                  <Cpu size={10} />
                  {modelStatus.model_info.base_model ?? "API"}
                </>
              ) : (
                <>
                  <Cpu size={10} />
                  {modelStatus.model_info.device.split(" ").slice(-1)[0]}
                </>
              )}
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
