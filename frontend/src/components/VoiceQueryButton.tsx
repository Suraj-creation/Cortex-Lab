"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Mic, MicOff, Loader2 } from "lucide-react";
import { voiceQuery } from "@/lib/api";
import { VoiceQueryResult } from "@/lib/types";

interface Props {
  onResult: (result: VoiceQueryResult) => void;
  onError: (error: string) => void;
  disabled?: boolean;
}

export function VoiceQueryButton({ onResult, onError, disabled }: Props) {
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupRecording();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cleanupRecording = () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
  };

  const startRecording = useCallback(async () => {
    if (recording || processing || disabled) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      // Use ScriptProcessorNode (widely supported)
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      chunksRef.current = [];

      processor.onaudioprocess = (e) => {
        const channelData = e.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(channelData));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
      setRecording(true);
    } catch (err) {
      onError(
        err instanceof Error ? err.message : "Microphone access denied"
      );
    }
  }, [recording, processing, disabled, onError]);

  const stopRecording = useCallback(async () => {
    if (!recording) return;

    setRecording(false);
    setProcessing(true);

    try {
      const chunks = chunksRef.current;
      cleanupRecording();

      // Merge Float32 chunks
      const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);

      if (totalLength < 1600) {
        // Less than 0.1s at 16kHz
        onError("Recording too short. Hold the button longer.");
        setProcessing(false);
        return;
      }

      const combined = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }

      // Float32 → Int16 PCM
      const int16 = new Int16Array(combined.length);
      for (let i = 0; i < combined.length; i++) {
        const s = Math.max(-1, Math.min(1, combined[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }

      // Convert to base64
      const bytes = new Uint8Array(int16.buffer);
      let binary = "";
      for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      const base64 = btoa(binary);

      const result = await voiceQuery(base64);
      onResult(result);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Voice query failed");
    }

    setProcessing(false);
  }, [recording, onResult, onError]);

  if (processing) {
    return (
      <button
        disabled
        className="rounded-xl p-2.5 text-violet-400 bg-violet-50 border border-violet-200"
        title="Processing voice..."
      >
        <Loader2 size={18} className="animate-spin" />
      </button>
    );
  }

  return (
    <button
      onMouseDown={startRecording}
      onMouseUp={stopRecording}
      onMouseLeave={() => {
        if (recording) stopRecording();
      }}
      onTouchStart={(e) => {
        e.preventDefault();
        startRecording();
      }}
      onTouchEnd={(e) => {
        e.preventDefault();
        stopRecording();
      }}
      disabled={disabled}
      className={`rounded-xl p-2.5 transition-all duration-200 ${
        recording
          ? "bg-red-50 text-red-500 border border-red-200 shadow-sm shadow-red-100 scale-110"
          : "text-slate-400 hover:text-indigo-500 hover:bg-indigo-50 border border-transparent hover:border-indigo-200"
      } disabled:opacity-50 disabled:cursor-not-allowed`}
      title={recording ? "Release to send" : "Hold to speak"}
    >
      {recording ? <MicOff size={18} /> : <Mic size={18} />}
    </button>
  );
}
