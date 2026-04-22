"use client";

import { useState, useEffect } from "react";
import {
  Fingerprint,
  Mic,
  Check,
  AlertCircle,
} from "lucide-react";
import { startEnrollment, getEnrollmentStatus } from "@/lib/api";
import { AmbientState } from "@/lib/types";

interface Props {
  status: AmbientState | null;
}

export function VoiceEnrollment({ status }: Props) {
  const [enrolled, setEnrolled] = useState(false);
  const [enrolling, setEnrolling] = useState(false);
  const [result, setResult] = useState<{
    success: boolean;
    message: string;
    samples_used?: number;
    consistency?: number;
  } | null>(null);
  const [duration, setDuration] = useState(20);
  const [countdown, setCountdown] = useState(0);

  useEffect(() => {
    getEnrollmentStatus()
      .then((data) => setEnrolled(data.enrolled))
      .catch(() => {});
  }, []);

  // Update from status
  useEffect(() => {
    if (status?.enrolled !== undefined) {
      setEnrolled(status.enrolled);
    }
  }, [status?.enrolled]);

  // Countdown timer during enrollment
  useEffect(() => {
    if (!enrolling || countdown <= 0) return;
    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [enrolling, countdown]);

  const handleEnroll = async () => {
    setEnrolling(true);
    setResult(null);
    setCountdown(duration);

    try {
      const data = await startEnrollment(duration);
      setResult(data);
      if (data.success) setEnrolled(true);
    } catch (e: unknown) {
      setResult({
        success: false,
        message: e instanceof Error ? e.message : "Enrollment failed",
      });
    }

    setEnrolling(false);
    setCountdown(0);
  };

  return (
    <div className="p-6 max-w-lg mx-auto space-y-6">
      {/* Status Card */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 text-center">
        <div
          className={`mx-auto h-16 w-16 rounded-2xl flex items-center justify-center mb-4 ${
            enrolled
              ? "bg-emerald-50 border border-emerald-200"
              : "bg-amber-50 border border-amber-200"
          }`}
        >
          {enrolled ? (
            <Check size={28} className="text-emerald-500" />
          ) : (
            <Fingerprint size={28} className="text-amber-500" />
          )}
        </div>
        <h3 className="text-sm font-semibold text-slate-700 mb-1">
          {enrolled ? "Voice Enrolled" : "Voice Not Enrolled"}
        </h3>
        <p className="text-xs text-slate-400 max-w-sm mx-auto">
          {enrolled
            ? "Your voice is enrolled. Cortex Lab can identify you in ambient conversations."
            : "Enroll your voice so Cortex Lab can distinguish you from other speakers during ambient listening."}
        </p>
      </div>

      {/* Enrollment Action */}
      <div className="rounded-xl border border-slate-200 bg-white p-6">
        <h4 className="text-xs font-semibold text-slate-600 mb-3 uppercase tracking-wider">
          {enrolled ? "Re-Enroll" : "Enroll Your Voice"}
        </h4>

        <p className="text-xs text-slate-500 mb-4 leading-relaxed">
          Speak naturally for {duration} seconds. Read a passage, have a
          conversation, or just talk about your day. The system will create a
          voiceprint from your speech patterns.
        </p>

        {/* Duration selector */}
        <div className="flex items-center gap-3 mb-4">
          <span className="text-xs text-slate-500">Duration:</span>
          <div className="flex gap-1">
            {[10, 15, 20, 30].map((d) => (
              <button
                key={d}
                onClick={() => setDuration(d)}
                disabled={enrolling}
                className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
                  duration === d
                    ? "bg-indigo-100 text-indigo-700 border border-indigo-200"
                    : "bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100"
                } disabled:opacity-50`}
              >
                {d}s
              </button>
            ))}
          </div>
        </div>

        {/* Enrollment button */}
        {enrolling ? (
          <div className="text-center py-6">
            <div className="relative mx-auto h-20 w-20 mb-3">
              <div className="absolute inset-0 rounded-full border-4 border-indigo-100" />
              <div
                className="absolute inset-0 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin"
                style={{ animationDuration: "1.5s" }}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-lg font-bold text-indigo-600">
                  {countdown}
                </span>
              </div>
            </div>
            <div className="flex items-center justify-center gap-2 text-indigo-600">
              <Mic size={14} className="animate-pulse" />
              <span className="text-xs font-medium">
                Recording... Speak naturally
              </span>
            </div>
          </div>
        ) : (
          <button
            onClick={handleEnroll}
            className="w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 px-4 py-3 text-sm font-medium text-white hover:from-indigo-600 hover:to-violet-600 transition-all shadow-sm"
          >
            <Mic size={16} />
            {enrolled ? "Re-Enroll Voice" : "Start Voice Enrollment"}
          </button>
        )}

        {/* Result */}
        {result && (
          <div
            className={`mt-4 rounded-lg p-3 border ${
              result.success
                ? "bg-emerald-50 border-emerald-200"
                : "bg-red-50 border-red-200"
            }`}
          >
            <div className="flex items-start gap-2">
              {result.success ? (
                <Check size={14} className="text-emerald-500 mt-0.5" />
              ) : (
                <AlertCircle size={14} className="text-red-500 mt-0.5" />
              )}
              <div>
                <p
                  className={`text-xs font-medium ${
                    result.success ? "text-emerald-700" : "text-red-700"
                  }`}
                >
                  {result.message}
                </p>
                {result.samples_used !== undefined && (
                  <p className="text-[10px] text-emerald-600 mt-1">
                    Samples: {result.samples_used} · Consistency:{" "}
                    {((result.consistency || 0) * 100).toFixed(1)}%
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Info */}
      <div className="rounded-xl bg-slate-50 border border-slate-200 p-4">
        <h4 className="text-xs font-semibold text-slate-600 mb-2">
          How it works
        </h4>
        <ul className="space-y-1.5 text-[11px] text-slate-500">
          <li className="flex items-start gap-2">
            <span className="text-indigo-400 mt-0.5">1.</span>
            Your voice is recorded and split into 3-second segments
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-400 mt-0.5">2.</span>
            ECAPA-TDNN extracts 192-dimensional embeddings from each segment
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-400 mt-0.5">3.</span>
            Embeddings are averaged into a single voiceprint
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-400 mt-0.5">4.</span>
            Voiceprint is saved locally — never sent to any cloud
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-400 mt-0.5">5.</span>
            During ambient listening, each speech segment is compared against
            your voiceprint (cosine similarity ≥ 0.70 = you)
          </li>
        </ul>
      </div>
    </div>
  );
}
