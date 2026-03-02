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
import { ModelStatus } from "@/lib/types";

type ActiveView = "chat" | "memories" | "graph" | "dashboard" | "observability" | "ambient";

export default function Home() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>({
    status: "loading",
    model_loaded: false,
    model_info: {},
  });
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeView, setActiveView] = useState<ActiveView>("chat");
  const [conversations, setConversations] = useState<
    { id: string; title: string; date: string }[]
  >([]);
  const [activeConversation, setActiveConversation] = useState("");

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
      />

      {/* Main Area */}
      <div className="flex flex-1 flex-col min-w-0">
        <Header
          modelStatus={modelStatus}
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen((p) => !p)}
        />
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
      </div>
    </div>
  );
}
