"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  ArrowLeft,
  Loader2,
  FileText,
  Upload,
  Trash2,
  Search,
  RefreshCw,
  CheckCircle2,
  Clock,
  Cloud,
  CloudOff,
  FileUp,
  BarChart3,
  X,
  AlertCircle,
  TreeDeciduous,
  Eye,
} from "lucide-react";
import {
  listDocuments,
  uploadDocument,
  deleteDocument,
  queryDocuments,
  getPageIndexUsage,
  getDocumentTree,
  PageIndexDocument,
  PageIndexUsage,
} from "@/lib/api";

interface TreeNode {
  title?: string;
  node_id?: string;
  page_index?: number;
  summary?: string;
  text?: string;
  sub_nodes?: TreeNode[];
}

export function DocumentsPanel({ onBack }: { onBack: () => void }) {
  const [documents, setDocuments] = useState<PageIndexDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [usage, setUsage] = useState<PageIndexUsage | null>(null);
  const [connected, setConnected] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const [error, setError] = useState("");

  // Query state
  const [queryText, setQueryText] = useState("");
  const [querying, setQuerying] = useState(false);
  const [queryResult, setQueryResult] = useState<{
    answer: string;
    sections: { page: number; content: string; doc_id: string; score: number }[];
  } | null>(null);

  // Tree view state
  const [treeDocId, setTreeDocId] = useState<string | null>(null);
  const [treeData, setTreeData] = useState<TreeNode[] | null>(null);
  const [treeLoading, setTreeLoading] = useState(false);

  // Drag-and-drop
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadData = useCallback(async () => {
    try {
      const [docsRes, usageRes] = await Promise.allSettled([
        listDocuments(),
        getPageIndexUsage(),
      ]);

      if (docsRes.status === "fulfilled") {
        setDocuments(docsRes.value.documents || []);
        setEnabled(docsRes.value.pageindex_enabled);
      }

      if (usageRes.status === "fulfilled") {
        if (usageRes.value.usage) setUsage(usageRes.value.usage);
        setConnected(usageRes.value.connected ?? false);
      }

      // Only show error if both failed
      if (docsRes.status === "rejected" && usageRes.status === "rejected") {
        setError("Failed to connect to backend");
      }
    } catch (err) {
      console.error("Failed to load documents:", err);
      setError("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, [loadData]);

  // ── Upload handler ──────────────────────────────────────────────

  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }

    setUploading(true);
    setUploadProgress(`Uploading ${file.name}...`);
    setError("");

    try {
      const result = await uploadDocument(file);
      if (result.already_indexed) {
        setUploadProgress(`"${file.name}" is already indexed.`);
      } else {
        setUploadProgress(
          `"${file.name}" uploaded! Processing by PageIndex...`
        );
      }
      await loadData();
      setTimeout(() => setUploadProgress(""), 4000);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Upload failed"
      );
      setUploadProgress("");
    } finally {
      setUploading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
    e.target.value = "";
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  // ── Delete handler ──────────────────────────────────────────────

  const handleDelete = async (docId: string) => {
    try {
      await deleteDocument(docId);
      await loadData();
    } catch (err) {
      setError("Failed to delete document");
    }
  };

  // ── Query handler ───────────────────────────────────────────────

  const handleQuery = async () => {
    if (!queryText.trim()) return;
    setQuerying(true);
    setQueryResult(null);
    setError("");

    try {
      const result = await queryDocuments(queryText.trim());
      setQueryResult(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Query failed"
      );
    } finally {
      setQuerying(false);
    }
  };

  // ── Tree view handler ──────────────────────────────────────────

  const handleViewTree = async (docId: string) => {
    if (treeDocId === docId) {
      setTreeDocId(null);
      setTreeData(null);
      return;
    }
    setTreeDocId(docId);
    setTreeLoading(true);
    try {
      const res = await getDocumentTree(docId);
      const tree = res.tree as { result?: TreeNode[] };
      setTreeData(tree?.result ?? []);
    } catch {
      setError("Failed to load document tree");
      setTreeData(null);
    } finally {
      setTreeLoading(false);
    }
  };

  // ── Tree renderer ───────────────────────────────────────────────

  const renderTreeNode = (node: TreeNode, depth: number = 0) => (
    <div
      key={node.node_id ?? node.title}
      className="border-l-2 border-indigo-200"
      style={{ marginLeft: depth * 16 }}
    >
      <div className="pl-3 py-1.5">
        <p className="text-xs font-medium text-slate-700">
          {node.title || "Untitled Section"}
          {node.page_index != null && (
            <span className="ml-2 text-[10px] text-slate-400 font-normal">
              p.{node.page_index}
            </span>
          )}
        </p>
        {node.summary && node.summary !== node.text && (
          <p className="text-[11px] text-slate-500 mt-0.5 line-clamp-2">
            {node.summary.slice(0, 200)}
          </p>
        )}
      </div>
      {node.sub_nodes?.map((child) => renderTreeNode(child, depth + 1))}
    </div>
  );

  // ── Render ──────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-indigo-500" />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white/80 backdrop-blur-xl border-b border-slate-200/60">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="rounded-xl p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all"
            >
              <ArrowLeft size={18} />
            </button>
            <div>
              <h1 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                <FileText size={20} className="text-indigo-500" />
                PageIndex Documents
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">
                Upload PDFs for reasoning-based document retrieval
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
                connected
                  ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                  : "bg-red-50 text-red-700 border border-red-200"
              }`}
            >
              {connected ? (
                <Cloud size={12} />
              ) : (
                <CloudOff size={12} />
              )}
              {connected ? "Connected" : "Disconnected"}
            </div>
            <button
              onClick={loadData}
              className="rounded-xl p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all"
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">
        {/* Error banner */}
        {error && (
          <div className="rounded-xl bg-red-50 border border-red-200 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-red-700 text-sm">
              <AlertCircle size={16} />
              {error}
            </div>
            <button onClick={() => setError("")} className="text-red-400 hover:text-red-600">
              <X size={14} />
            </button>
          </div>
        )}

        {/* Not enabled state */}
        {!enabled && (
          <div className="rounded-2xl border border-amber-200 bg-amber-50 p-6 text-center">
            <CloudOff className="mx-auto h-10 w-10 text-amber-400 mb-3" />
            <h3 className="text-sm font-semibold text-amber-800 mb-1">
              PageIndex Not Enabled
            </h3>
            <p className="text-xs text-amber-600">
              Enable PageIndex in <code className="bg-amber-100 px-1 rounded">config/pageindex_config.py</code> and restart the backend.
            </p>
          </div>
        )}

        {enabled && (
          <>
            {/* Stats row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <StatCard
                icon={FileText}
                label="Documents"
                value={documents.length}
                color="bg-indigo-50 text-indigo-600"
              />
              <StatCard
                icon={CheckCircle2}
                label="Ready"
                value={documents.filter((d) => d.status === "ready").length}
                color="bg-emerald-50 text-emerald-600"
              />
              <StatCard
                icon={Search}
                label="Queries Used"
                value={`${usage?.queries_used ?? 0}/${usage?.queries_limit ?? 500}`}
                color="bg-blue-50 text-blue-600"
              />
              <StatCard
                icon={BarChart3}
                label="Pages Used"
                value={`${usage?.pages_used ?? 0}/${usage?.pages_limit ?? 2000}`}
                color="bg-violet-50 text-violet-600"
              />
            </div>

            {/* Upload zone */}
            <div
              className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 ${
                dragOver
                  ? "border-indigo-400 bg-indigo-50/50 scale-[1.01]"
                  : "border-slate-200 bg-white hover:border-slate-300"
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <div className="flex flex-col items-center py-8 gap-3">
                {uploading ? (
                  <>
                    <Loader2 className="h-8 w-8 animate-spin text-indigo-500" />
                    <p className="text-sm text-indigo-600 font-medium">
                      {uploadProgress}
                    </p>
                  </>
                ) : uploadProgress ? (
                  <>
                    <CheckCircle2 className="h-8 w-8 text-emerald-500" />
                    <p className="text-sm text-emerald-600 font-medium">
                      {uploadProgress}
                    </p>
                  </>
                ) : (
                  <>
                    <div className="rounded-2xl bg-indigo-50 p-3">
                      <FileUp className="h-8 w-8 text-indigo-500" />
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-medium text-slate-700">
                        Drop a PDF here or{" "}
                        <button
                          onClick={() => fileInputRef.current?.click()}
                          className="text-indigo-600 hover:text-indigo-700 underline underline-offset-2"
                        >
                          browse files
                        </button>
                      </p>
                      <p className="text-xs text-slate-400 mt-1">
                        PDF files only · Indexed by PageIndex cloud AI · Non-sensitive documents
                      </p>
                    </div>
                  </>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>

            {/* Document list */}
            {documents.length > 0 && (
              <div className="space-y-2">
                <h3 className="text-xs font-medium uppercase tracking-widest text-slate-400 px-1">
                  Indexed Documents ({documents.length})
                </h3>
                {documents.map((doc) => (
                  <div key={doc.doc_id}>
                    <div className="rounded-xl border border-slate-200 bg-white p-4 flex items-center justify-between group hover:border-slate-300 transition-all">
                      <div className="flex items-center gap-3 min-w-0 flex-1">
                        <div
                          className={`p-2 rounded-xl ${
                            doc.status === "ready"
                              ? "bg-emerald-50"
                              : "bg-amber-50"
                          }`}
                        >
                          <FileText
                            size={16}
                            className={
                              doc.status === "ready"
                                ? "text-emerald-600"
                                : "text-amber-600"
                            }
                          />
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium text-slate-700 truncate">
                            {doc.filename}
                          </p>
                          <div className="flex items-center gap-3 mt-0.5">
                            <span
                              className={`inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full ${
                                doc.status === "ready"
                                  ? "bg-emerald-50 text-emerald-700"
                                  : "bg-amber-50 text-amber-700"
                              }`}
                            >
                              {doc.status === "ready" ? (
                                <CheckCircle2 size={10} />
                              ) : (
                                <Clock size={10} />
                              )}
                              {doc.status}
                            </span>
                            {doc.estimated_pages > 0 && (
                              <span className="text-[10px] text-slate-400">
                                ~{doc.estimated_pages} pages
                              </span>
                            )}
                            <span className="text-[10px] text-slate-400">
                              {new Date(doc.uploaded_at).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={() => handleViewTree(doc.doc_id)}
                          className={`rounded-lg p-2 transition-all ${
                            treeDocId === doc.doc_id
                              ? "bg-indigo-100 text-indigo-600"
                              : "text-slate-400 hover:text-indigo-600 hover:bg-indigo-50"
                          }`}
                          title="View tree structure"
                        >
                          <TreeDeciduous size={14} />
                        </button>
                        <button
                          onClick={() => handleDelete(doc.doc_id)}
                          className="rounded-lg p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 transition-all"
                          title="Delete document"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>

                    {/* Tree view (expanded) */}
                    {treeDocId === doc.doc_id && (
                      <div className="mt-1 ml-6 rounded-xl border border-slate-200 bg-slate-50 p-4 max-h-64 overflow-y-auto">
                        {treeLoading ? (
                          <div className="flex items-center gap-2 text-slate-400 text-xs">
                            <Loader2 size={14} className="animate-spin" />
                            Loading tree...
                          </div>
                        ) : treeData && treeData.length > 0 ? (
                          <div className="space-y-1">
                            <p className="text-[10px] font-medium uppercase tracking-widest text-slate-400 mb-2">
                              Document Structure
                            </p>
                            {treeData.map((node) => renderTreeNode(node))}
                          </div>
                        ) : (
                          <p className="text-xs text-slate-400">
                            No tree structure available
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Empty state */}
            {documents.length === 0 && !uploading && (
              <div className="rounded-2xl border border-slate-200 bg-white p-10 text-center">
                <FileText className="mx-auto h-12 w-12 text-slate-300 mb-4" />
                <h3 className="text-sm font-semibold text-slate-600 mb-1">
                  No documents indexed yet
                </h3>
                <p className="text-xs text-slate-400 max-w-sm mx-auto">
                  Upload a PDF to enable reasoning-based document retrieval.
                  PageIndex preserves document structure, follows cross-references,
                  and uses LLM reasoning to find relevant sections.
                </p>
              </div>
            )}

            {/* Document Query */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5">
              <h3 className="text-xs font-medium uppercase tracking-widest text-slate-400 mb-3 flex items-center gap-2">
                <Search size={12} />
                Query Documents
              </h3>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleQuery()}
                  placeholder={
                    documents.length === 0
                      ? "Upload a document first..."
                      : "Ask a question about your documents..."
                  }
                  disabled={documents.length === 0}
                  className="flex-1 rounded-xl border border-slate-200 px-4 py-2.5 text-sm text-slate-700 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-200 focus:border-indigo-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                />
                <button
                  onClick={handleQuery}
                  disabled={!queryText.trim() || querying || documents.length === 0}
                  className="rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
                >
                  {querying ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Search size={14} />
                  )}
                  Query
                </button>
              </div>

              {/* Query result */}
              {queryResult && (
                <div className="mt-4 space-y-3">
                  {queryResult.answer && (
                    <div className="rounded-xl bg-indigo-50 border border-indigo-100 p-4">
                      <p className="text-[10px] font-medium uppercase tracking-widest text-indigo-400 mb-1.5">
                        PageIndex Answer
                      </p>
                      <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
                        {queryResult.answer}
                      </p>
                    </div>
                  )}
                  {queryResult.sections.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-[10px] font-medium uppercase tracking-widest text-slate-400">
                        Retrieved Sections ({queryResult.sections.length})
                      </p>
                      {queryResult.sections.map((section, i) => (
                        <div
                          key={i}
                          className="rounded-xl border border-slate-200 p-3"
                        >
                          <div className="flex items-center gap-2 mb-1.5">
                            <span className="text-[10px] font-mono bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded">
                              Page {section.page}
                            </span>
                            <span className="text-[10px] text-slate-400">
                              Score: {(section.score * 100).toFixed(0)}%
                            </span>
                          </div>
                          <p className="text-xs text-slate-600 leading-relaxed line-clamp-4">
                            {section.content}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* How it works */}
            <div className="rounded-2xl border border-slate-200 bg-gradient-to-br from-slate-50 to-white p-5">
              <h3 className="text-xs font-medium uppercase tracking-widest text-slate-400 mb-3 flex items-center gap-2">
                <Eye size={12} />
                How PageIndex Works
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="mx-auto w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center mb-2">
                    <Upload size={14} className="text-indigo-600" />
                  </div>
                  <p className="text-xs font-medium text-slate-700">1. Upload PDF</p>
                  <p className="text-[10px] text-slate-400 mt-0.5">
                    Document is processed into a hierarchical tree index
                  </p>
                </div>
                <div className="text-center">
                  <div className="mx-auto w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center mb-2">
                    <TreeDeciduous size={14} className="text-violet-600" />
                  </div>
                  <p className="text-xs font-medium text-slate-700">2. Tree Reasoning</p>
                  <p className="text-[10px] text-slate-400 mt-0.5">
                    AI navigates the tree to find relevant sections by reasoning
                  </p>
                </div>
                <div className="text-center">
                  <div className="mx-auto w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center mb-2">
                    <Search size={14} className="text-emerald-600" />
                  </div>
                  <p className="text-xs font-medium text-slate-700">3. Smart Retrieval</p>
                  <p className="text-[10px] text-slate-400 mt-0.5">
                    Results fused with memory channels via RRF as Channel 6
                  </p>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Stat card sub-component ─────────────────────────────────────

function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  icon: any;
  label: string;
  value: string | number;
  color: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 card-hover">
      <div className="flex items-center gap-2 mb-2">
        <div className={`p-1.5 rounded-xl ${color}`}>
          <Icon size={14} />
        </div>
        <span className="text-[10px] font-medium uppercase tracking-widest text-slate-400">
          {label}
        </span>
      </div>
      <p className="text-lg font-semibold text-slate-800">{value}</p>
    </div>
  );
}
