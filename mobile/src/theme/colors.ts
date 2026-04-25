/**
 * Cortex Lab Mobile — "Cortex Aurora" Light Design System
 * Premium light theme with indigo/violet accents, glassmorphism, and depth.
 * Replaces Neural Dark (#060e20) → Light Aurora (#f8fafc)
 */

// ─── Core Light Palette ──────────────────────────────────────────────────────
export const NEURAL = {
  // Backgrounds / Surfaces (layered depth — light)
  background:              '#f8fafc', // root bg — cool white
  surfaceDim:              '#f1f5f9', // dimmed surface
  surfaceContainerLowest:  '#ffffff', // pure white cards
  surfaceContainerLow:     '#f8fafc', // sidebar / nav
  surfaceContainer:        '#ffffff', // cards — white
  surfaceContainerHigh:    '#f1f5f9', // elevated cards, active rows
  surfaceContainerHighest: '#e2e8f0', // inputs, floating sheets
  surfaceVariant:          '#f1f5f9', // glass elements
  surfaceBright:           '#e2e8f0', // bright surface for emphasis

  // Primary — Indigo
  primary:           '#6366f1',
  primaryContainer:  '#eef2ff',
  primaryDim:        '#4f46e5',
  primaryFixed:      '#818cf8',
  primaryFixedDim:   '#6366f1',
  inversePrimary:    '#c7d2fe',

  // Secondary — Violet
  secondary:          '#8b5cf6',
  secondaryContainer: '#ede9fe',
  secondaryDim:       '#7c3aed',
  secondaryFixed:     '#ddd6fe',

  // Tertiary — Emerald
  tertiary:          '#10b981',
  tertiaryContainer: '#d1fae5',
  tertiaryDim:       '#059669',

  // Error — Rose
  error:          '#f43f5e',
  errorContainer: '#ffe4e6',
  errorDim:       '#e11d48',

  // On-colors (text on light surfaces)
  onBackground:        '#0f172a',
  onSurface:           '#0f172a', // primary text — near black
  onSurfaceVariant:    '#475569', // secondary text — slate
  onPrimary:           '#ffffff',
  onSecondary:         '#ffffff',
  onTertiary:          '#ffffff',
  onTertiaryFixed:     '#064e3b',
  onTertiaryContainer: '#065f46',
  onError:             '#ffffff',
  inverseSurface:      '#1e293b',
  inverseOnSurface:    '#f8fafc',

  // Border / Outline
  outline:        '#94a3b8',  // medium borders
  outlineVariant: '#e2e8f0',  // subtle borders

  // Surface tint
  surfaceTint: '#6366f1',
} as const;

// ─── Semantic Aliases ─────────────────────────────────────────────────────────
export const SEMANTIC = {
  // Text
  textPrimary:   NEURAL.onSurface,         // #0f172a
  textSecondary: NEURAL.onSurfaceVariant,  // #475569
  textTertiary:  NEURAL.outline,           // #94a3b8
  textMuted:     '#cbd5e1',                // very light text
  textBrand:     NEURAL.primary,           // #6366f1

  // Surfaces
  bgRoot:     NEURAL.background,              // #f8fafc
  bgCard:     NEURAL.surfaceContainer,        // #ffffff
  bgCardHigh: NEURAL.surfaceContainerHigh,    // #f1f5f9
  bgInput:    '#f8fafc',                      // input bg
  bgSidebar:  NEURAL.surfaceContainerLow,     // #f8fafc
  bgElevated: '#ffffff',                      // elevated — pure white

  // Interactive
  brand:        NEURAL.primary,    // #6366f1
  brandDim:     NEURAL.primaryDim, // #4f46e5
  accent:       NEURAL.secondary,  // #8b5cf6
  live:         NEURAL.tertiary,   // #10b981
  error:        NEURAL.error,      // #f43f5e

  // Borders
  borderGhost:  '#f1f5f9',                    // very subtle
  borderSubtle: NEURAL.outlineVariant,        // #e2e8f0
  borderBrand:  `${NEURAL.primary}30`,        // indigo at 19%

  // Glass (for glassmorphism effects)
  glassBg:    'rgba(255, 255, 255, 0.75)',
  glassBlur:  20,

  // Gradients
  gradientPrimary: ['#6366f1', '#4f46e5'] as string[],    // indigo gradient CTA
  gradientUser:    ['#6366f1', '#8b5cf6'] as string[],     // user message
  gradientBg:      ['#f8fafc', '#eef2ff'] as string[],     // subtle bg gradient
  gradientAccent:  ['#8b5cf6', '#6366f1'] as string[],     // violet → indigo
  gradientSuccess: ['#10b981', '#059669'] as string[],     // emerald
  gradientWarm:    ['#fef3c7', '#fde68a'] as string[],     // amber warm
} as const;

// ─── Spacing Scale ────────────────────────────────────────────────────────────
export const SPACING = {
  xs:   4,
  sm:   8,
  md:   12,
  lg:   16,
  xl:   20,
  '2xl': 24,
  '3xl': 32,
  '4xl': 40,
  '5xl': 56,
} as const;

// ─── Border Radius ────────────────────────────────────────────────────────────
export const RADIUS = {
  sm:   6,
  md:   10,
  lg:   14,
  xl:   18,
  '2xl': 24,
  '3xl': 32,
  full: 9999,
} as const;

// ─── Typography Scale ─────────────────────────────────────────────────────────
export const FONT_SIZE = {
  xs:   11,
  sm:   12,
  base: 14,
  md:   15,
  lg:   17,
  xl:   20,
  '2xl': 24,
  '3xl': 28,
  '4xl': 34,
} as const;

export const FONT_WEIGHT = {
  light:     '300' as const,
  normal:    '400' as const,
  medium:    '500' as const,
  semibold:  '600' as const,
  bold:      '700' as const,
  extrabold: '800' as const,
} as const;

// ─── Shadows (Platform-adaptive) ──────────────────────────────────────────────
export const SHADOWS = {
  sm: {
    shadowColor: '#0f172a',
    shadowOpacity: 0.04,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  md: {
    shadowColor: '#0f172a',
    shadowOpacity: 0.06,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 3,
  },
  lg: {
    shadowColor: '#0f172a',
    shadowOpacity: 0.08,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 4 },
    elevation: 6,
  },
  xl: {
    shadowColor: '#0f172a',
    shadowOpacity: 0.12,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 8 },
    elevation: 10,
  },
  glow: {
    shadowColor: '#6366f1',
    shadowOpacity: 0.2,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 0 },
    elevation: 8,
  },
  glowSuccess: {
    shadowColor: '#10b981',
    shadowOpacity: 0.2,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 0 },
    elevation: 8,
  },
} as const;

// ─── Status Colors (for pipeline events, badges) ──────────────────────────────
export const STATUS_COLORS = {
  running:  '#3b82f6',    // blue
  complete: '#10b981',    // emerald
  error:    '#f43f5e',    // rose
  pending:  '#f59e0b',    // amber
  cached:   '#8b5cf6',    // violet
  queued:   '#94a3b8',    // slate
  blocked:  '#a78bfa',    // violet light
  waiting:  '#f59e0b',    // amber
} as const;

// ─── Legacy compatibility ─────────────────────────────────────────────────────
export const COLORS = {
  primary:   { 300: '#a5b4fc', 500: NEURAL.primary, 600: NEURAL.primaryDim, 700: '#4338ca' },
  secondary: { 500: NEURAL.secondary },
  success:   { 50: '#f0fdf4', 100: '#dcfce7', 300: '#86efac', 500: NEURAL.tertiary, 600: NEURAL.tertiaryDim },
  warning:   { 100: '#fef3c7', 300: '#fcd34d', 500: '#f59e0b', 600: '#d97706' },
  error:     { 50: '#fff1f2', 100: '#ffe4e6', 200: '#fecdd3', 300: '#fda4af', 500: NEURAL.error, 600: NEURAL.errorDim, 700: '#be123c' },
  info:      { 100: '#dbeafe', 300: '#93c5fd', 500: '#3b82f6', 600: '#2563eb' },
  surface:   { 400: NEURAL.onSurfaceVariant, 500: NEURAL.outline, 700: '#e2e8f0', 800: '#f1f5f9', 900: '#ffffff' },
  white:     '#ffffff',
  black:     '#000000',
  transparent: 'transparent',
  violet:    { 500: NEURAL.secondary, 600: NEURAL.secondaryDim },
  emerald:   { 500: NEURAL.tertiary, 600: NEURAL.tertiaryDim },
  amber:     { 500: '#f59e0b', 600: '#d97706' },
  cyan:      { 500: '#06b6d4', 600: '#0891b2' },
} as const;

export const SEMANTIC_COLORS = {
  textPrimary:    SEMANTIC.textPrimary,
  textSecondary:  SEMANTIC.textSecondary,
  textTertiary:   SEMANTIC.textTertiary,
  textInverse:    NEURAL.inverseOnSurface,
  textMuted:      SEMANTIC.textMuted,
  textOnAccent:   '#ffffff',
  bgCanvas:       NEURAL.background,
  bgPrimary:      '#ffffff',
  bgSecondary:    '#f8fafc',
  bgTertiary:     '#f1f5f9',
  bgElevated:     '#ffffff',
  bgOverlay:      'rgba(15, 23, 42, 0.5)',
  bgHighlight:    `${NEURAL.primary}10`,
  borderPrimary:   NEURAL.outlineVariant,
  borderSecondary: NEURAL.outline,
  borderAccent:    `${NEURAL.primary}30`,
  borderLight:     '#f1f5f9',
  buttonPrimary:          NEURAL.primary,
  buttonPrimaryHover:     NEURAL.primaryDim,
  buttonPrimaryActive:    '#4338ca',
  buttonPrimaryDisabled:  '#e2e8f0',
  buttonSecondary:        '#f1f5f9',
  buttonSecondaryText:    NEURAL.onSurface,
  buttonSecondaryHover:   '#e2e8f0',
  navBackground:          '#ffffff',
  statusSuccess: NEURAL.tertiary,
  statusWarning: '#f59e0b',
  statusError:   NEURAL.error,
  statusInfo:    '#3b82f6',
  statusLoading: NEURAL.primary,
  glassBg:     SEMANTIC.glassBg,
  glassOverlay: 'rgba(255, 255, 255, 0.6)',
  accentPrimary:   NEURAL.primary,
  accentSecondary: NEURAL.secondary,
} as const;

// Re-export spacing + radius with old names for backward compat
export { SPACING as SPACING_OLD };
export const BORDER_RADIUS = RADIUS;
export const TYPOGRAPHY = {
  fontSize:   FONT_SIZE,
  fontWeight: FONT_WEIGHT,
  lineHeight:  { tight: 1.2, normal: 1.5, relaxed: 1.75, loose: 2 },
} as const;
