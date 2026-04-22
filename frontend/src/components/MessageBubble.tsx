"use client";

import { useState, useEffect, useRef, memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { ChatMessage } from "@/lib/types";
import {
  User,
  Bot,
  ChevronDown,
  ChevronRight,
  Brain,
  Copy,
  Check,
  Zap,
  Sparkles,
  FileText,
  Network,
  Shield,
  Timer,
  Lightbulb,
  Eye,
  EyeOff,
  CheckCircle,
} from "lucide-react";
import { TTSPlayback } from "./TTSPlayback";
import { PipelineTracePanel } from "./PipelineTracePanel";

interface Props {
  message: ChatMessage;
}

// React.memo — only re-render when message content/streaming state actually changes (§10.1)
export const MessageBubble = memo(function MessageBubble({ message }: Props) {
  const [showThinking, setShowThinking] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);
  const [copied, setCopied] = useState(false);
  const thinkingEndRef = useRef<HTMLDivElement>(null);
  const thinkingContainerRef = useRef<HTMLDivElement>(null);

  const isUser = message.role === "user";

  // Parse thinking from streamed content (supports <think> and <Thinking> variants)
  let displayContent = message.content;
  let streamedThinking: string | null = null;
  let isCurrentlyThinking = false;

  if (!isUser) {
    // Check for lowercase <think>...</think> (streaming injection)
    if (message.content.includes("<think>")) {
      const thinkStart = message.content.indexOf("<think>") + 7;
      const thinkEnd = message.content.indexOf("</think>");
      if (thinkEnd > -1) {
        streamedThinking = message.content.slice(thinkStart, thinkEnd).trim();
        displayContent = message.content.slice(thinkEnd + 8).trim();
      } else {
        // Still thinking — </think> hasn't arrived yet
        streamedThinking = message.content.slice(thinkStart).trim();
        displayContent = "";
        isCurrentlyThinking = message.isStreaming || false;
      }
    }
    // Also handle <Thinking>...</Thinking> (model's visible reasoning format)
    if (displayContent.includes("<Thinking>")) {
      const tStart = displayContent.indexOf("<Thinking>") + 10;
      const tEnd = displayContent.indexOf("</Thinking>");
      if (tEnd > -1) {
        const visibleThinking = displayContent.slice(tStart, tEnd).trim();
        streamedThinking = streamedThinking
          ? streamedThinking + "\n\n---\n\n" + visibleThinking
          : visibleThinking;
        displayContent = displayContent.slice(tEnd + 12).trim();
      }
    }
  }

  // Priority: LLM's deep reasoning (from <think> tags in stream) > orchestrator summary (from rag_meta)
  // streamedThinking is the actual model reasoning; message.thinking may be either:
  //   - The orchestrator's brief note (~11 words) — set by onMeta before streaming
  //   - The full LLM reasoning — persisted by onDone after streaming completes
  // Use whichever is longer to ensure we never lose the deep reasoning.
  const thinking = (streamedThinking && message.thinking)
    ? (streamedThinking.length >= (message.thinking?.length || 0) ? streamedThinking : message.thinking)
    : (streamedThinking || message.thinking);
  const hasOutput = displayContent.trim().length > 0;

  // Auto-expand thinking when it first appears and keep expanded
  useEffect(() => {
    if (thinking && thinking.length > 0) {
      setShowThinking(true);
    }
  }, [thinking]);

  // Auto-scroll thinking panel content (NOT the page) while actively thinking
  useEffect(() => {
    if (showThinking && isCurrentlyThinking && thinkingContainerRef.current) {
      // Scroll the thinking container to its bottom, not the page
      const container = thinkingContainerRef.current;
      container.scrollTop = container.scrollHeight;
    }
  }, [thinking, showThinking, isCurrentlyThinking]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(displayContent || message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`group py-5 ${isUser ? "" : ""}`}>
      <div className="flex gap-4">
        {/* Avatar */}
        <div className="flex-shrink-0 mt-0.5">
          {isUser ? (
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 ring-1 ring-indigo-200">
              <User size={15} />
            </div>
          ) : (
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-50 to-emerald-100 text-emerald-600 ring-1 ring-emerald-200">
              <Bot size={15} />
            </div>
          )}
        </div>

        {/* Content */}
        <div className="min-w-0 flex-1 space-y-2.5">
          {/* Role label */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-700">
              {isUser ? "You" : "Cortex Lab"}
            </span>
            {message.isStreaming && (
              <span className="flex items-center gap-1.5 text-[10px] text-indigo-500">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-indigo-400 opacity-75" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-indigo-400" />
                </span>
                generating
              </span>
            )}
          </div>

          {/* Live Thinking Panel — Modern Collapsible Reasoning Display */}
          {thinking && (
            <div className={`rounded-2xl overflow-hidden transition-all duration-300 ${
              isCurrentlyThinking 
                ? 'bg-gradient-to-br from-indigo-50/80 via-violet-50/40 to-purple-50/30 shadow-lg shadow-indigo-100/40 ring-1 ring-indigo-200/60 thinking-active' 
                : 'bg-gradient-to-br from-slate-50/80 via-gray-50/40 to-slate-50/30 ring-1 ring-slate-200/60'
            }`}>
              {/* Header */}
              <button
                onClick={() => setShowThinking((p) => !p)}
                className={`flex w-full items-center gap-2.5 px-4 py-3 text-sm font-medium transition-all duration-200 ${
                  isCurrentlyThinking
                    ? 'text-indigo-700 hover:bg-indigo-50/50'
                    : 'text-slate-500 hover:bg-slate-50/50'
                }`}
              >
                {/* Brain icon with glow when active */}
                <div className={`relative flex items-center justify-center w-7 h-7 rounded-lg ${
                  isCurrentlyThinking
                    ? 'bg-indigo-100 shadow-sm shadow-indigo-200/50'
                    : 'bg-slate-100'
                }`}>
                  {isCurrentlyThinking ? (
                    <Brain size={14} className="text-indigo-600 animate-pulse" />
                  ) : (
                    <Lightbulb size={14} className="text-slate-400" />
                  )}
                  {isCurrentlyThinking && (
                    <span className="absolute -top-0.5 -right-0.5 flex h-2.5 w-2.5">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-indigo-400 opacity-60" />
                      <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-indigo-500" />
                    </span>
                  )}
                </div>

                <div className="flex flex-col items-start gap-0.5">
                  <span className="text-[13px] font-semibold">
                    {isCurrentlyThinking ? "Qwen is thinking…" : "Reasoning Process"}
                  </span>
                  {!isCurrentlyThinking && thinking && (
                    <span className="text-[10px] text-slate-400 font-normal">
                      {thinking.split(/\s+/).length} words of reasoning
                    </span>
                  )}
                </div>

                <div className="ml-auto flex items-center gap-2">
                  {isCurrentlyThinking && (
                    <span className="flex items-center gap-1.5 text-[10px] text-indigo-500 bg-indigo-50 px-2 py-0.5 rounded-full">
                      <span className="relative flex h-1.5 w-1.5">
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-indigo-400 opacity-75" />
                        <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-indigo-500" />
                      </span>
                      Live
                    </span>
                  )}
                  <div className={`flex items-center justify-center w-6 h-6 rounded-md transition-colors ${
                    showThinking ? 'bg-indigo-100/80 text-indigo-600' : 'bg-slate-100/80 text-slate-400'
                  }`}>
                    {showThinking ? <EyeOff size={12} /> : <Eye size={12} />}
                  </div>
                </div>
              </button>

              {/* Thinking content */}
              {showThinking && (
                <div
                  ref={thinkingContainerRef}
                  className={`border-t px-4 py-4 text-sm leading-relaxed max-h-[40rem] overflow-y-auto scroll-smooth thinking-content-enter ${
                  isCurrentlyThinking 
                    ? 'border-indigo-100/80 bg-white/40' 
                    : 'border-slate-100/80 bg-white/30'
                }`}>
                  <div className="space-y-2">
                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      className="prose-thinking"
                    >
                      {thinking}
                    </ReactMarkdown>
                    {isCurrentlyThinking && (
                      <div className="flex items-center gap-2 pt-3 text-xs text-indigo-400/80">
                        <Sparkles size={12} className="animate-pulse" />
                        <span className="italic">Reasoning in progress…</span>
                        <span className="flex gap-0.5">
                          <span className="typing-dot h-1 w-1 rounded-full bg-indigo-300" />
                          <span className="typing-dot h-1 w-1 rounded-full bg-indigo-300" />
                          <span className="typing-dot h-1 w-1 rounded-full bg-indigo-300" />
                        </span>
                      </div>
                    )}
                    {!isCurrentlyThinking && !message.isStreaming && thinking && (
                      <div className="flex items-center gap-2 pt-3 text-xs text-emerald-500/80">
                        <CheckCircle size={12} />
                        <span>Reasoning complete · {thinking.split(/\s+/).length} words</span>
                      </div>
                    )}
                  </div>
                  <div ref={thinkingEndRef} />
                </div>
              )}
            </div>
          )}

          {/* Main Output — Separated from Thinking */}
          {hasOutput && (
            <>
              {thinking && (
                <div className="flex items-center gap-3 pt-1.5 pb-1.5">
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-indigo-200 to-transparent" />
                  <span className="text-[10px] uppercase tracking-widest text-indigo-400/80 flex items-center gap-1.5 font-medium">
                    <Zap size={10} className="text-indigo-400" />
                    Final Answer
                  </span>
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-indigo-200 to-transparent" />
                </div>
              )}
              <div className="prose-chat text-slate-700">
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    code({ className, children, ...props }: any) {
                      const match = /language-(\w+)/.exec(className || "");
                      const isBlock =
                        typeof children === "string" && children.includes("\n");
                      if (isBlock || match) {
                        return (
                          <div className="relative group/code">
                            {match && (
                              <div className="absolute right-3 top-2 text-[10px] text-slate-400 uppercase tracking-wider">
                                {match[1]}
                              </div>
                            )}
                            <pre className="!mt-0">
                              <code className={className} {...props}>
                                {children}
                              </code>
                            </pre>
                          </div>
                        );
                      }
                      return (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {displayContent}
                </ReactMarkdown>
              </div>
            </>
          )}

          {/* Streaming cursor */}
          {message.isStreaming && !displayContent && !thinking && (
            <div className="flex items-center gap-1.5 py-2">
              <div className="typing-dot h-1.5 w-1.5 rounded-full bg-slate-400" />
              <div className="typing-dot h-1.5 w-1.5 rounded-full bg-slate-400" />
              <div className="typing-dot h-1.5 w-1.5 rounded-full bg-slate-400" />
            </div>
          )}

          {/* RAG Evidence Cards — collapsible to save space */}
          {!isUser && message.evidence && message.evidence.length > 0 && (
            <div className="mt-3">
              <button
                onClick={() => setShowEvidence((p) => !p)}
                className="flex items-center gap-2 w-full text-left group/ev hover:bg-slate-50/50 rounded-lg px-2 py-1.5 -mx-2 transition-colors"
              >
                <FileText size={12} className="text-slate-400" />
                <span className="text-[11px] font-medium text-slate-500">
                  Evidence ({message.evidence.length} memories)
                </span>
                {message.confidence !== undefined && (
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-md ${
                    message.confidence > 0.7
                      ? "bg-emerald-50 text-emerald-600 border border-emerald-200"
                      : message.confidence > 0.4
                      ? "bg-amber-50 text-amber-600 border border-amber-200"
                      : "bg-red-50 text-red-500 border border-red-200"
                  }`}>
                    {Math.round(message.confidence * 100)}% confidence
                  </span>
                )}
                <div className="ml-auto">
                  {showEvidence ? (
                    <ChevronDown size={12} className="text-slate-400" />
                  ) : (
                    <ChevronRight size={12} className="text-slate-400" />
                  )}
                </div>
              </button>
              {showEvidence && (
                <div className="grid gap-2 mt-2 max-h-[24rem] overflow-y-auto">
                  {message.evidence.slice(0, 5).map((ev, idx) => (
                    <div
                      key={idx}
                      className="rounded-xl border border-slate-200 bg-slate-50/50 p-3 text-xs hover:border-slate-300 transition-colors"
                    >
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="px-1.5 py-0.5 rounded-md bg-indigo-50 text-indigo-600 text-[9px] font-medium uppercase tracking-wider">
                          {ev.memory_type}
                        </span>
                        <span className="text-slate-400 text-[9px]">
                          {new Date(ev.timestamp).toLocaleDateString()}
                        </span>
                        <span className="text-slate-400 text-[9px]">
                          Score: {ev.score}
                        </span>
                        {ev.channel && (
                          <span className="text-slate-400 text-[9px]">
                            via {ev.channel}
                          </span>
                        )}
                      </div>
                      <p className="text-slate-600 leading-relaxed">
                        {ev.content}
                      </p>
                      {ev.entities && ev.entities.length > 0 && (
                        <div className="flex gap-1 mt-1.5">
                          {ev.entities.map((entity, i) => (
                            <span
                              key={i}
                              className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 text-[9px]"
                            >
                              {entity}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* RAG Metadata Bar */}
          {!isUser && !message.isStreaming && (message.agentsUsed || message.queryAnalysis) && (
            <div className="flex flex-wrap items-center gap-2 mt-2 text-[10px] text-slate-400">
              {message.agentsUsed && message.agentsUsed.length > 0 && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-lg bg-slate-50 border border-slate-200">
                  <Network size={9} />
                  Agents: {message.agentsUsed.join(", ")}
                </span>
              )}
              {message.queryAnalysis && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-lg bg-slate-50 border border-slate-200">
                  <Shield size={9} />
                  {message.queryAnalysis.intent} · {message.queryAnalysis.routing}
                </span>
              )}
              {message.processingTimeMs && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-lg bg-slate-50 border border-slate-200">
                  <Timer size={9} />
                  {Math.round(message.processingTimeMs)}ms
                </span>
              )}
              {message.cacheHit && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-600">
                  <Zap size={9} />
                  Cached
                </span>
              )}
            </div>
          )}

          {/* Pipeline Trace Observability Panel */}
          {!isUser && !message.isStreaming && message.pipelineTrace && (
            <PipelineTracePanel trace={message.pipelineTrace} />
          )}

          {/* Footer: usage + actions */}
          {!isUser && !message.isStreaming && displayContent && (
            <div className="flex items-center gap-3 pt-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-600 transition-colors"
              >
                {copied ? <Check size={11} /> : <Copy size={11} />}
                {copied ? "Copied" : "Copy"}
              </button>
              <TTSPlayback text={displayContent} />
              {message.usage && (
                <span className="flex items-center gap-1 text-[10px] text-slate-400">
                  <Zap size={10} />
                  {message.usage.completion_tokens} tokens
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}, (prev, next) => {
  // Custom comparator: re-render when message content OR observability data changes
  return prev.message.id === next.message.id
    && prev.message.content === next.message.content
    && prev.message.isStreaming === next.message.isStreaming
    && prev.message.isRAG === next.message.isRAG
    && prev.message.thinking === next.message.thinking
    && prev.message.pipelineTrace === next.message.pipelineTrace
    && prev.message.evidence === next.message.evidence
    && prev.message.confidence === next.message.confidence
    && prev.message.agentsUsed === next.message.agentsUsed;
});
