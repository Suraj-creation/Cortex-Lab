"use client";

interface WebOfflineReadinessBadgeProps {
  ready: boolean;
  details?: string;
}

export function WebOfflineReadinessBadge({ ready, details }: WebOfflineReadinessBadgeProps) {
  return (
    <div
      className={`inline-flex flex-col rounded-full border px-3 py-1 text-[11px] font-medium ${
        ready
          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
          : "bg-amber-50 text-amber-700 border-amber-200"
      }`}
    >
      <span>{ready ? "Offline Ready" : "Offline Not Ready"}</span>
      {details ? <span className="text-[10px] font-normal opacity-80 mt-0.5">{details}</span> : null}
    </div>
  );
}
