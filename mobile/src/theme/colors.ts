/**
 * Cortex Lab Mobile - Unified Design System
 * Replicates frontend Tailwind color system for React Native
 * Consistent with Next.js frontend implementation
 */

import { Platform } from "react-native";

export const COLORS = {
  // Primary Brand (Indigo)
  primary: {
    50: "#eef2ff",
    100: "#e0e7ff",
    200: "#c7d2fe",
    300: "#a5b4fc",
    400: "#818cf8",
    500: "#6366f1",
    600: "#4f46e5",
    700: "#4338ca",
    800: "#3730a3",
    900: "#312e81",
    950: "#1e1b4b",
  },

  // Secondary Brand (Violet)
  secondary: {
    500: "#8b5cf6",
    600: "#7c3aed",
  },

  // Surfaces (Slate/Neutral)
  surface: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
    950: "#020617",
  },

  // Semantic Colors
  success: {
    50: "#f0fdf4",
    100: "#dcfce7",
    200: "#bbf7d0",
    300: "#86efac",
    400: "#4ade80",
    500: "#22c55e",
    600: "#16a34a",
    700: "#15803d",
    800: "#166534",
    900: "#145231",
  },

  warning: {
    50: "#fffbeb",
    100: "#fef3c7",
    200: "#fde68a",
    300: "#fcd34d",
    400: "#fbbf24",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
    800: "#92400e",
    900: "#78350f",
  },

  error: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
  },

  info: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
  },

  // Semantic Aliases
  emerald: {
    500: "#10b981",
    600: "#059669",
  },

  cyan: {
    500: "#06b6d4",
    600: "#0891b2",
  },

  amber: {
    500: "#f59e0b",
    600: "#d97706",
  },

  violet: {
    500: "#a78bfa",
    600: "#8b5cf6",
  },

  // Transparent/Utilities
  transparent: "transparent",
  black: "#000000",
  white: "#ffffff",
};

/**
 * Semantic color mapping for common UI patterns
 */
export const SEMANTIC_COLORS = {
  // Text
  textPrimary: COLORS.surface[900],
  textSecondary: COLORS.surface[700],
  textTertiary: COLORS.surface[500],
  textInverse: COLORS.white,
  textMuted: COLORS.surface[400],
  textOnAccent: COLORS.white,

  // Backgrounds
  bgCanvas: "#f3f6fb",
  bgPrimary: COLORS.white,
  bgSecondary: "#f8fafc",
  bgTertiary: COLORS.surface[100],
  bgElevated: COLORS.white,
  bgOverlay: "rgba(2, 6, 23, 0.45)",
  bgHighlight: "rgba(79, 70, 229, 0.08)",

  // Borders
  borderPrimary: "rgba(15, 23, 42, 0.10)",
  borderSecondary: "rgba(15, 23, 42, 0.18)",
  borderAccent: "rgba(79, 70, 229, 0.28)",
  borderLight: "rgba(15, 23, 42, 0.08)",

  // Interactive
  buttonPrimary: COLORS.primary[600],
  buttonPrimaryHover: COLORS.primary[700],
  buttonPrimaryActive: COLORS.primary[800],
  buttonPrimaryDisabled: COLORS.surface[300],

  buttonSecondary: COLORS.surface[100],
  buttonSecondaryText: COLORS.surface[800],
  buttonSecondaryHover: COLORS.surface[200],
  navBackground: "rgba(255, 255, 255, 0.96)",

  // Status
  statusSuccess: COLORS.success[600],
  statusWarning: COLORS.warning[600],
  statusError: COLORS.error[600],
  statusInfo: COLORS.info[600],
  statusLoading: COLORS.primary[600],

  // Glass/Overlay effects
  glassBg: "rgba(255, 255, 255, 0.82)",
  glassOverlay: "rgba(255, 255, 255, 0.94)",

  // Accent
  accentPrimary: COLORS.primary[600],
  accentSecondary: COLORS.cyan[500],
};

/**
 * Spacing scale (matching Tailwind)
 */
export const SPACING = {
  xs: 3,
  sm: 6,
  md: 10,
  lg: 14,
  xl: 18,
  "2xl": 22,
  "3xl": 28,
  "4xl": 36,
  "5xl": 44,
};

/**
 * Typography scale
 * React Native fontWeight must be: "normal" | "bold" | "100" | "200" | "300" | "400" | "500" | "600" | "700" | "800" | "900"
 */
export const TYPOGRAPHY = {
  fontSize: {
    xs: 11,
    sm: 12,
    base: 14,
    md: 15,
    lg: 16,
    xl: 18,
    "2xl": 20,
    "3xl": 24,
    "4xl": 28,
  },
  fontWeight: {
    light: "300" as const,
    normal: "400" as const,
    medium: "500" as const,
    semibold: "600" as const,
    bold: "700" as const,
    extrabold: "800" as const,
  } as const,
  // Line height multipliers (actual pixel heights)
  lineHeight: {
    tight: 1.2,
    normal: 1.5,
    relaxed: 1.75,
    loose: 2,
  },
};

/**
 * Border radius scale
 */
export const BORDER_RADIUS = {
  none: 0,
  sm: 4,
  md: 6,
  lg: 8,
  xl: 12,
  "2xl": 16,
  "3xl": 20,
  full: 999,
};

/**
 * Shadows (using elevation system)
 */
const isWebPlatform = Platform.OS === "web";

type ShadowToken = Record<string, unknown>;

export const SHADOWS: Record<"none" | "sm" | "md" | "lg" | "xl", ShadowToken> = {
  none: isWebPlatform
    ? { boxShadow: "none" }
    : {
        elevation: 0,
        shadowColor: COLORS.black,
        shadowOpacity: 0,
        shadowRadius: 0,
        shadowOffset: { width: 0, height: 0 },
      },
  sm: isWebPlatform
    ? { boxShadow: "0 2px 8px rgba(15, 23, 42, 0.06)" }
    : {
        elevation: 2,
        shadowColor: "#0f172a",
        shadowOpacity: 0.08,
        shadowRadius: 4,
        shadowOffset: { width: 0, height: 2 },
      },
  md: isWebPlatform
    ? { boxShadow: "0 6px 16px rgba(15, 23, 42, 0.10)" }
    : {
        elevation: 4,
        shadowColor: "#0f172a",
        shadowOpacity: 0.12,
        shadowRadius: 8,
        shadowOffset: { width: 0, height: 4 },
      },
  lg: isWebPlatform
    ? { boxShadow: "0 10px 28px rgba(15, 23, 42, 0.14)" }
    : {
        elevation: 7,
        shadowColor: "#0f172a",
        shadowOpacity: 0.16,
        shadowRadius: 12,
        shadowOffset: { width: 0, height: 7 },
      },
  xl: isWebPlatform
    ? { boxShadow: "0 16px 40px rgba(15, 23, 42, 0.18)" }
    : {
        elevation: 10,
        shadowColor: "#0f172a",
        shadowOpacity: 0.2,
        shadowRadius: 18,
        shadowOffset: { width: 0, height: 10 },
      },
};
