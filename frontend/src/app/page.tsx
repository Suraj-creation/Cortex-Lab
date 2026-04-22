"use client";

import { useState, useEffect } from "react";
import { ChatPanel } from "@/components/ChatPanel";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { MemoryBrowser } from "@/components/MemoryBrowser";
import { KnowledgeGraph } from "@/components/KnowledgeGraph";
import { RAGDashboard } from "@/components/RAGDashboard";
import { ObservabilityDashboard } from "@/components/ObservabilityDashboard";
import { AmbientPanel } from "@/components/AmbientPanel";
import { DocumentsPanel } from "@/components/DocumentsPanel";
import { AgentChatPanel } from "@/components/agent/AgentChatPanel";
import { WikiBrowser } from "@/components/agent/WikiBrowser";
import { useGlobalAgentEvents } from "@/lib/agent/useAgentEvents";
import { getRuntimeSafetyExecutorStatus, getRuntimeSafetyPermissions } from "@/lib/api";
import { ModelStatus, RuntimeApprovalSummary } from "@/lib/types";

type ActiveView = "chat" | "agent" | "wiki" | "memories" | "graph" | "dashboard" | "observability" | "ambient" | "documents";

export default function Home() {
  useGlobalAgentEvents();

  const [modelStatus, setModelStatus] = useState<ModelStatus>({
    status: "loading",
    model_loaded: false,
    model_info: {},
  });
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeView, setActiveView] = useState<ActiveView>("agent");
  const [conversations, setConversations] = useState<
    { id: string; title: string; date: string }[]
  >([]);
  const [activeConversation, setActiveConversation] = useState("");
  const [approvalSummary, setApprovalSummary] = useState<RuntimeApprovalSummary>({
    pending: 0,
    expired: 0,
    approved_total: 0,
    running: 0,
    waiting_retry: 0,
    failed: 0,
    completed: 0,
  });

  // Load conversations from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem("cortex-conversations");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setConversations(parsed);
          setActiveConversation(parsed[0].id);
          return;
        }
      }
    } catch { /* ignore parse errors */ }
    // Default: start with one new chat
    const id = Date.now().toString();
    const initial = [{ id, title: "New Chat", date: new Date().toISOString() }];
    setConversations(initial);
    setActiveConversation(id);
  }, []);

  // Persist conversations to localStorage whenever they change
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem("cortex-conversations", JSON.stringify(conversations));
    }
  }, [conversations]);

  // Poll model health — retry faster on initial connect, slower once healthy
  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval>;
    let healthy = false;

    const check = async () => {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);
        const res = await fetch("/api/health", { signal: controller.signal });
        clearTimeout(timeout);
        if (res.ok) {
          const data = await res.json();
          setModelStatus(data);
          // Once healthy, poll less frequently
          if (!healthy) {
            healthy = true;
            clearInterval(intervalId);
            intervalId = setInterval(check, 15_000);
          }
        }
      } catch {
        // Don't immediately mark as offline — could be a transient proxy hiccup
        if (healthy) {
          // Was healthy before — mark offline only after a second consecutive failure
          healthy = false;
          clearInterval(intervalId);
          intervalId = setInterval(check, 3_000);
        } else {
          setModelStatus({
            status: "offline",
            model_loaded: false,
            model_info: {},
          });
        }
      }
    };
    check();
    // Start with faster polling while connecting
    intervalId = setInterval(check, 3_000);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    let mounted = true;

    const loadApprovalSummary = async () => {
      try {
        const [permissions, executor] = await Promise.all([
          getRuntimeSafetyPermissions(),
          getRuntimeSafetyExecutorStatus(),
        ]);
        if (!mounted) return;

        setApprovalSummary({
          pending: permissions.count || 0,
          expired: permissions.expired_count || 0,
          approved_total: executor.summary?.approved_total || 0,
          running: executor.summary?.running || 0,
          waiting_retry: executor.summary?.waiting_retry || 0,
          failed: executor.summary?.failed || 0,
          completed: executor.summary?.completed || 0,
        });
      } catch {
        if (!mounted) return;
        setApprovalSummary((prev) => ({
          ...prev,
          pending: 0,
          expired: 0,
          waiting_retry: 0,
        }));
      }
    };

    loadApprovalSummary();
    const interval = setInterval(loadApprovalSummary, 7000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleNewChat = () => {
    const id = Date.now().toString();
    setConversations((prev) => [
      { id, title: "New Chat", date: new Date().toISOString() },
      ...prev,
    ]);
    setActiveConversation(id);
    setActiveView("chat");
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#f8fafc]">
      {/* Ambient background glow */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[40%] -left-[20%] w-[70%] h-[70%] rounded-full bg-indigo-500/[0.03] blur-[120px]" />
        <div className="absolute -bottom-[30%] -right-[20%] w-[60%] h-[60%] rounded-full bg-violet-500/[0.03] blur-[120px]" />
      </div>

      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={activeConversation}
        onSelect={(id) => {
          setActiveConversation(id);
          setActiveView("chat");
        }}
        onNewChat={handleNewChat}
        onToggle={() => setSidebarOpen((p) => !p)}
        activeView={activeView}
        onNavigate={setActiveView}
        approvalSummary={approvalSummary}
      />

      {/* Main Area */}
      <div className="flex flex-1 flex-col min-w-0">
        <Header
          modelStatus={modelStatus}
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen((p) => !p)}
        />
        {activeView === "agent" && <AgentChatPanel />}
        {activeView === "wiki" && <WikiBrowser />}
        {activeView === "chat" && activeConversation && (
          <ChatPanel
            key={activeConversation}
            modelStatus={modelStatus}
            conversationId={activeConversation}
            onTitleUpdate={(title) =>
              setConversations((prev) =>
                prev.map((c) =>
                  c.id === activeConversation ? { ...c, title } : c,
                ),
              )
            }
          />
        )}
        {activeView === "memories" && (
          <MemoryBrowser onBack={() => setActiveView("chat")} />
        )}
        {activeView === "graph" && (
          <KnowledgeGraph onBack={() => setActiveView("chat")} />
        )}
        {activeView === "dashboard" && (
          <RAGDashboard onBack={() => setActiveView("chat")} />
        )}
        {activeView === "observability" && (
          <ObservabilityDashboard onBack={() => setActiveView("chat")} />
        )}
        {activeView === "ambient" && (
          <AmbientPanel onBack={() => setActiveView("chat")} />
        )}
        {activeView === "documents" && (
          <DocumentsPanel onBack={() => setActiveView("chat")} />
        )}
      </div>
    </div>
  );
}
