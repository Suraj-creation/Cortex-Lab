"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Volume2, VolumeX, Loader2 } from "lucide-react";
import { synthesizeSpeech } from "@/lib/api";

interface Props {
  text: string;
}

export function TTSPlayback({ text }: Props) {
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const textRef = useRef(text);

  // Invalidate cache if text changes
  useEffect(() => {
    if (text !== textRef.current) {
      textRef.current = text;
      // Revoke old URL
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
        audioUrlRef.current = null;
      }
      // Stop current playback
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
        setPlaying(false);
      }
    }
  }, [text]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  const handlePlay = useCallback(async () => {
    // If already playing, stop
    if (playing && audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setPlaying(false);
      return;
    }

    // If we have cached audio, replay it
    if (audioUrlRef.current) {
      const audio = new Audio(audioUrlRef.current);
      audioRef.current = audio;
      audio.onended = () => setPlaying(false);
      audio.onerror = () => setPlaying(false);
      audio.play().catch(() => setPlaying(false));
      setPlaying(true);
      return;
    }

    // Synthesize new audio
    setLoading(true);
    try {
      const wavBuffer = await synthesizeSpeech(text);
      const blob = new Blob([wavBuffer], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      audioUrlRef.current = url;

      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => setPlaying(false);
      audio.onerror = () => setPlaying(false);
      await audio.play();
      setPlaying(true);
    } catch {
      // TTS not available — silently fail
      setPlaying(false);
    }
    setLoading(false);
  }, [text, playing]);

  if (loading) {
    return (
      <button
        disabled
        className="rounded-lg p-1 text-slate-300"
        title="Synthesizing speech..."
      >
        <Loader2 size={13} className="animate-spin" />
      </button>
    );
  }

  return (
    <button
      onClick={handlePlay}
      className={`rounded-lg p-1 transition-all ${
        playing
          ? "text-indigo-500 hover:text-indigo-600"
          : "text-slate-300 hover:text-slate-500"
      }`}
      title={playing ? "Stop" : "Read aloud"}
    >
      {playing ? <VolumeX size={13} /> : <Volume2 size={13} />}
    </button>
  );
}
