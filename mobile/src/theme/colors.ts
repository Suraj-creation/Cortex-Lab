/**
 * Cortex Lab Mobile — Neural Dark Design System
 * Matches "Cortex Neural Dark" from Stitch Project 4565645347880235116
 * Philosophy: The Neural Luminary — Atmospheric Intelligence
 */

// ─── Stitch "Cortex Neural Dark" Named Colors ────────────────────────────────
export const NEURAL = {
  // Backgrounds / Surfaces (layered depth)
  background:              '#060e20', // root bg
  surfaceDim:              '#060e20', // same as bg
  surfaceContainerLowest:  '#000000',
  surfaceContainerLow:     '#091328',
  surfaceContainer:        '#0f1930', // cards
  surfaceContainerHigh:    '#141f38', // elevated cards, active rows
  surfaceContainerHighest: '#192540', // inputs, floating sheets
  surfaceVariant:          '#192540', // glass elements
  surfaceBright:           '#1f2b49',

  // Primary — Indigo
  primary:           '#a3a6ff',
  primaryContainer:  '#9396ff',
  primaryDim:        '#6063ee',
  primaryFixed:      '#9396ff',
  primaryFixedDim:   '#8387ff',
  inversePrimary:    '#494bd7',

  // Secondary — Violet
  secondary:          '#ac8aff',
  secondaryContainer: '#5516be',
  secondaryDim:       '#8455ef',
  secondaryFixed:     '#dac9ff',

  // Tertiary — Emerald
  tertiary:          '#9bffce',
  tertiaryContainer: '#69f6b8',
  tertiaryDim:       '#58e7ab',

  // Error — Coral
  error:          '#ff6e84',
  errorContainer: '#a70138',
  errorDim:       '#d73357',

  // On-colors (text on surfaces)
  onBackground:        '#dee5ff',
  onSurface:           '#dee5ff', // primary text
  onSurfaceVariant:    '#a3aac4', // secondary text / metadata
  onPrimary:           '#0f00a4',
  onSecondary:         '#280067',
  onTertiary:          '#006443',
  onTertiaryFixed:     '#00452d',
  onTertiaryContainer: '#005a3c',
  onError:             '#490013',
  inverseSurface:      '#faf8ff',
  inverseOnSurface:    '#4d556b',

  // Border / Outline
  outline:        '#6d758c',
  outlineVariant: '#40485d', // ghost borders

  // Surface tint
  surfaceTint: '#a3a6ff',
} as const;

// ─── Semantic Aliases ─────────────────────────────────────────────────────────
export const SEMANTIC = {
  // Text
  textPrimary:   NEURAL.onSurface,         // #dee5ff
  textSecondary: NEURAL.onSurfaceVariant,  // #a3aac4
  textTertiary:  NEURAL.outline,           // #6d758c
  textMuted:     NEURAL.outlineVariant,    // #40485d
  textBrand:     NEURAL.primary,           // #a3a6ff

  // Surfaces
  bgRoot:     NEURAL.background,              // #060e20
  bgCard:     NEURAL.surfaceContainer,        // #0f1930
  bgCardHigh: NEURAL.surfaceContainerHigh,    // #141f38
  bgInput:    NEURAL.surfaceContainerHighest, // #192540
  bgSidebar:  NEURAL.surfaceContainerLow,     // #091328
  bgElevated: NEURAL.surfaceBright,           // #1f2b49

  // Interactive
  brand:        NEURAL.primary,    // #a3a6ff
  brandDim:     NEURAL.primaryDim, // #6063ee
  accent:       NEURAL.secondary,  // #ac8aff
  live:         NEURAL.tertiary,   // #9bffce
  error:        NEURAL.error,      // #ff6e84

  // Borders (ghost border rule: outlineVariant at low opacity)
  borderGhost:  `${NEURAL.outlineVariant}26`, // 15% opacity
  borderSubtle: NEURAL.outlineVariant,        // #40485d full
  borderBrand:  `${NEURAL.primary}40`,        // primary at 25%

  // Glass
  glassBg:    `${NEURAL.surfaceVariant}99`,   // 60% opacity
  glassBlur:  20,

  // Gradients (used via expo-linear-gradient)
  gradientPrimary: [NEURAL.primary, NEURAL.primaryDim] as string[], // indigo gradient CTA
  gradientUser:    ['#a3a6ff', '#6063ee'] as string[],              // user bubble
  gradientBg:      ['#060e20', '#091328'] as string[],              // subtle bg gradient
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
// Stitch: ROUND_FOUR ≈ rounded-xl. No radius < 6px ("Don't use border-radius smaller than md")
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

// ─── Status Colors (for pipeline events, badges) ──────────────────────────────
export const STATUS_COLORS = {
  running:  NEURAL.primary,   // indigo pulse
  complete: NEURAL.tertiary,  // emerald
  error:    NEURAL.error,     // coral
  pending:  NEURAL.secondary, // violet
  cached:   NEURAL.tertiaryDim,
} as const;

// ─── Legacy compatibility (for shared components still using COLORS / SEMANTIC_COLORS) ──
export const COLORS = {
  primary:   { 300: '#c4c6ff', 500: NEURAL.primary, 600: NEURAL.primaryDim, 700: '#4f51c8' },
  secondary: { 500: NEURAL.secondary },
  success:   { 50: '#f0fff7', 100: '#d6f7e8', 300: NEURAL.tertiaryDim, 500: NEURAL.tertiary, 600: NEURAL.tertiaryContainer },
  warning:   { 100: '#fff3c4', 300: '#ffd86b', 500: '#f59e0b', 600: '#d97706' },
  error:     { 50: '#fff0f2', 100: '#ffd6db', 200: '#ffb3bc', 300: NEURAL.errorDim, 500: NEURAL.error, 600: NEURAL.errorContainer, 700: '#7a0020' },
  info:      { 100: '#dbeafe', 300: '#93c5fd', 500: '#60a5fa', 600: '#3b82f6' },
  surface:   { 400: NEURAL.onSurfaceVariant, 500: NEURAL.outline, 700: NEURAL.surfaceBright, 800: NEURAL.surfaceContainerHigh, 900: NEURAL.surfaceContainer },
  white:     '#ffffff',
  black:     '#000000',
  transparent: 'transparent',
  violet:    { 500: NEURAL.secondary, 600: NEURAL.secondaryDim },
  emerald:   { 500: NEURAL.tertiary, 600: NEURAL.tertiaryDim },
  amber:     { 500: '#f59e0b', 600: '#d97706' },
  cyan:      { 500: '#58e7ff', 600: '#00bcd4' },
} as const;

export const SEMANTIC_COLORS = {
  textPrimary:    SEMANTIC.textPrimary,
  textSecondary:  SEMANTIC.textSecondary,
  textTertiary:   SEMANTIC.textTertiary,
  textInverse:    NEURAL.inverseSurface,
  textMuted:      SEMANTIC.textMuted,
  textOnAccent:   NEURAL.onPrimary,
  bgCanvas:       NEURAL.background,
  bgPrimary:      NEURAL.surfaceContainer,
  bgSecondary:    NEURAL.surfaceContainerLow,
  bgTertiary:     NEURAL.surfaceContainerHigh,
  bgElevated:     NEURAL.surfaceContainerHighest,
  bgOverlay:      'rgba(6, 14, 32, 0.85)',
  bgHighlight:    `${NEURAL.primary}20`,
  borderPrimary:   NEURAL.outlineVariant,
  borderSecondary: NEURAL.outline,
  borderAccent:    `${NEURAL.primary}40`,
  borderLight:     `${NEURAL.outlineVariant}26`,
  buttonPrimary:          NEURAL.primary,
  buttonPrimaryHover:     NEURAL.primaryDim,
  buttonPrimaryActive:    NEURAL.inversePrimary,
  buttonPrimaryDisabled:  NEURAL.surfaceBright,
  buttonSecondary:        NEURAL.surfaceContainerHigh,
  buttonSecondaryText:    NEURAL.onSurface,
  buttonSecondaryHover:   NEURAL.surfaceContainerHighest,
  navBackground:          NEURAL.surfaceContainer,
  statusSuccess: NEURAL.tertiary,
  statusWarning: '#f59e0b',
  statusError:   NEURAL.error,
  statusInfo:    NEURAL.primary,
  statusLoading: NEURAL.primary,
  glassBg:     SEMANTIC.glassBg,
  glassOverlay: NEURAL.surfaceContainerHighest,
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
