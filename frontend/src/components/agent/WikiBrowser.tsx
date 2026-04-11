"use client";

import { useState, useEffect, useCallback } from "react";
import { listWikiPages, searchWiki, getClaimStats } from "@/lib/agent/api";
import type { WikiPageInfo } from "@/lib/types";
import ReactMarkdown from "react-markdown";

export function WikiBrowser() {
  const [pages, setPages] = useState<WikiPageInfo[]>([]);
  const [selectedPage, setSelectedPage] = useState<WikiPageInfo | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [stats, setStats] = useState<{ total_pages: number; total_topics: number; total_linked_claims: number } | null>(null);
  const [claimStats, setClaimStats] = useState<{ total: number; active: number; topics: number } | null>(null);
  const [loading, setLoading] = useState(false);

  const loadPages = useCallback(async () => {
    setLoading(true);
    try {
      const res = await listWikiPages();
      setPages(res.pages);
      setStats(res.stats);
    } catch {
      // wiki may not be initialized yet
    }
    try {
      const cs = await getClaimStats();
      setClaimStats(cs);
    } catch {
      // claim store may not be initialized
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadPages();
  }, [loadPages]);

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      loadPages();
      return;
    }
    setLoading(true);
    try {
      const res = await searchWiki(searchQuery);
      setPages(res.results);
    } catch {
      // search failed
    }
    setLoading(false);
  }, [searchQuery, loadPages]);

  return (
    <div className="flex h-full bg-zinc-950">
      {/* Sidebar */}
      <div className="w-72 border-r border-zinc-800 flex flex-col">
        <div className="p-3 border-b border-zinc-800">
          <h3 className="text-sm font-semibold text-zinc-200 mb-2">Personal Wiki</h3>
          <div className="flex gap-1">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search wiki..."
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-300 placeholder:text-zinc-500 focus:outline-none focus:border-zinc-600"
            />
            <button
              onClick={handleSearch}
              className="px-2 py-1 bg-zinc-700 text-zinc-300 rounded text-xs hover:bg-zinc-600"
            >
              Go
            </button>
          </div>

          {stats && (
            <div className="mt-2 flex gap-2 text-[10px] text-zinc-500">
              <span>{stats.total_pages} pages</span>
              <span>{stats.total_topics} topics</span>
              {claimStats && <span>{claimStats.active} claims</span>}
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto">
          {loading && (
            <div className="text-center text-zinc-500 text-xs py-4">Loading...</div>
          )}
          {!loading && pages.length === 0 && (
            <div className="text-center text-zinc-600 text-xs py-8 px-4">
              No wiki pages yet. The Wiki Agent creates pages automatically as you chat.
            </div>
          )}
          {pages.map((page) => (
            <button
              key={page.id}
              onClick={() => setSelectedPage(page)}
              className={`w-full text-left px-3 py-2 border-b border-zinc-800/50 hover:bg-zinc-800/50 transition-colors ${
                selectedPage?.id === page.id ? "bg-zinc-800/50" : ""
              }`}
            >
              <div className="text-xs font-medium text-zinc-300 truncate">{page.title}</div>
              <div className="flex gap-1 mt-1 flex-wrap">
                {page.topics.slice(0, 3).map((t) => (
                  <span key={t} className="text-[9px] px-1 py-0.5 bg-zinc-800 rounded text-zinc-500">
                    {t}
                  </span>
                ))}
              </div>
              <div className="text-[10px] text-zinc-600 mt-1">
                v{page.version} / {page.claim_ids.length} claims
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {!selectedPage ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-zinc-600 text-sm mb-2">Personal Knowledge Wiki</div>
            <div className="text-zinc-700 text-xs max-w-md">
              Your wiki is built automatically by the Wiki Agent. As you chat,
              atomic claims are extracted and organized into wiki pages.
              Select a page from the sidebar to view it.
            </div>
          </div>
        ) : (
          <div>
            <div className="mb-4">
              <h1 className="text-xl font-semibold text-zinc-200">{selectedPage.title}</h1>
              <div className="flex items-center gap-3 mt-1">
                <span className="text-[10px] text-zinc-500">
                  Version {selectedPage.version}
                </span>
                <span className="text-[10px] text-zinc-500">
                  Updated {new Date(selectedPage.updated_at).toLocaleDateString()}
                </span>
                <span className="text-[10px] text-zinc-500">
                  {selectedPage.claim_ids.length} linked claims
                </span>
              </div>
              <div className="flex gap-1 mt-2">
                {selectedPage.topics.map((t) => (
                  <span
                    key={t}
                    className="text-[10px] px-2 py-0.5 bg-zinc-800 border border-zinc-700 rounded text-zinc-400"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown>{selectedPage.content}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
