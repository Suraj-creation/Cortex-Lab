/**
 * Button — Cortex Aurora gradient CTA
 * Primary: indigo gradient, light secondary, clean outline/ghost
 */
import React from 'react';
import {
  Pressable,
  Text,
  StyleSheet,
  ViewStyle,
  TextStyle,
  ActivityIndicator,
  View,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../../theme/colors';

export type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost' | 'error' | 'success';
export type ButtonSize = 'xs' | 'sm' | 'md' | 'lg';

interface ButtonProps {
  label: string;
  onPress?: () => void;
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  loading?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
  icon?: React.ReactNode;
  fullWidth?: boolean;
}

const SIZE_CONFIG: Record<ButtonSize, { px: number; py: number; fontSize: number; radius: number }> = {
  xs: { px: 10, py: 5,  fontSize: FONT_SIZE.xs,   radius: RADIUS.md },
  sm: { px: 14, py: 7,  fontSize: FONT_SIZE.sm,   radius: RADIUS.lg },
  md: { px: 18, py: 10, fontSize: FONT_SIZE.base, radius: RADIUS.xl },
  lg: { px: 24, py: 14, fontSize: FONT_SIZE.lg,   radius: RADIUS['2xl'] },
};

export function Button({
  label,
  onPress,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  style,
  textStyle,
  icon,
  fullWidth = false,
}: ButtonProps) {
  const cfg = SIZE_CONFIG[size];
  const isPrimary = variant === 'primary';

  const inner = (
    <View
      style={[
        styles.inner,
        { paddingHorizontal: cfg.px, paddingVertical: cfg.py, borderRadius: cfg.radius },
        !isPrimary && variantStyles[variant],
        (disabled || loading) && styles.disabled,
      ]}
    >
      {loading ? (
        <ActivityIndicator size="small" color={isPrimary ? '#ffffff' : '#6366f1'} />
      ) : (
        <>
          {icon && <View style={styles.iconWrap}>{icon}</View>}
          <Text
            style={[
              styles.label,
              { fontSize: cfg.fontSize },
              variantTextStyles[variant],
              textStyle,
              (disabled || loading) && styles.disabledText,
            ]}
          >
            {label}
          </Text>
        </>
      )}
    </View>
  );

  if (isPrimary && !disabled && !loading) {
    return (
      <Pressable
        onPress={onPress}
        disabled={disabled || loading}
        style={[fullWidth && styles.fullWidth, style]}
      >
        {({ pressed }) => (
          <LinearGradient
            colors={['#6d7dff', '#5463df']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={[
              styles.inner,
              { paddingHorizontal: cfg.px, paddingVertical: cfg.py, borderRadius: cfg.radius },
              pressed && styles.pressed,
              styles.primarySurface,
              SHADOWS.glow,
            ]}
          >
            {icon && <View style={styles.iconWrap}>{icon}</View>}
            <Text style={[styles.label, { fontSize: cfg.fontSize, color: '#ffffff' }, textStyle]}>
              {label}
            </Text>
          </LinearGradient>
        )}
      </Pressable>
    );
  }

  return (
    <Pressable
      onPress={onPress}
      disabled={disabled || loading}
      style={({ pressed }) => [
        fullWidth && styles.fullWidth,
        pressed && !disabled && styles.pressed,
        style,
      ]}
    >
      {inner}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  inner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: RADIUS.xl,
  },
  label: {
    fontWeight: FONT_WEIGHT.semibold,
    color: '#0f172a',
    letterSpacing: 0.2,
  },
  primarySurface: {
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.35)',
  },
  iconWrap: { marginRight: 6 },
  disabled: { opacity: 0.45 },
  disabledText: { color: '#94a3b8' },
  fullWidth: { width: '100%' },
  pressed: { opacity: 0.82, transform: [{ scale: 0.98 }] },
});

const variantStyles: Record<ButtonVariant, ViewStyle> = {
  primary:   { backgroundColor: '#6366f1' },
  secondary: { backgroundColor: '#edf2fb', borderWidth: 1, borderColor: '#ffffff', ...SHADOWS.md },
  outline:   { backgroundColor: '#eef3fb', borderWidth: 1, borderColor: '#ffffff', ...SHADOWS.md },
  ghost:     { backgroundColor: 'rgba(255,255,255,0.4)' },
  error:     { backgroundColor: '#fff4f5', borderWidth: 1, borderColor: '#ffffff', ...SHADOWS.md },
  success:   { backgroundColor: '#eefbf5', borderWidth: 1, borderColor: '#ffffff', ...SHADOWS.md },
};

const variantTextStyles: Record<ButtonVariant, TextStyle> = {
  primary:   { color: '#ffffff' },
  secondary: { color: '#334155' },
  outline:   { color: '#475569' },
  ghost:     { color: '#6366f1' },
  error:     { color: '#e11d48' },
  success:   { color: '#059669' },
};
