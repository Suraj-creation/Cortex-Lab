/**
 * Button — Neural Dark gradient CTA
 * Primary: gradient indigo → primaryDim at 135°
 * Design: "Glass & Gradient Rule" from Cortex Neural Dark
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
import { NEURAL, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../../theme/colors';

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
  xs: { px: 10, py: 4,  fontSize: FONT_SIZE.xs,   radius: RADIUS.md },
  sm: { px: 14, py: 7,  fontSize: FONT_SIZE.sm,   radius: RADIUS.lg },
  md: { px: 18, py: 10, fontSize: FONT_SIZE.base, radius: RADIUS.xl },
  lg: { px: 24, py: 14, fontSize: FONT_SIZE.lg,   radius: RADIUS.full },
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
        !isPrimary && styles.nonGradient,
        !isPrimary && variantStyles[variant],
        (disabled || loading) && styles.disabled,
      ]}
    >
      {loading ? (
        <ActivityIndicator size="small" color={isPrimary ? NEURAL.onPrimary : NEURAL.primary} />
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
            colors={[NEURAL.primary, NEURAL.primaryDim]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={[
              styles.inner,
              { paddingHorizontal: cfg.px, paddingVertical: cfg.py, borderRadius: cfg.radius },
              pressed && styles.pressed,
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
  nonGradient: {},
  label: {
    fontWeight: FONT_WEIGHT.semibold,
    color: NEURAL.onSurface,
    letterSpacing: 0.2,
  },
  iconWrap: { marginRight: 6 },
  disabled: { opacity: 0.45 },
  disabledText: { color: NEURAL.onSurfaceVariant },
  fullWidth: { width: '100%' },
  pressed: { opacity: 0.82 },
});

const variantStyles: Record<ButtonVariant, ViewStyle> = {
  primary:   { backgroundColor: NEURAL.primary },
  secondary: { backgroundColor: NEURAL.surfaceContainerHigh },
  outline:   { backgroundColor: 'transparent', borderWidth: 1, borderColor: NEURAL.outlineVariant },
  ghost:     { backgroundColor: 'transparent' },
  error:     { backgroundColor: `${NEURAL.error}22`, borderWidth: 1, borderColor: NEURAL.error },
  success:   { backgroundColor: `${NEURAL.tertiary}22`, borderWidth: 1, borderColor: NEURAL.tertiary },
};

const variantTextStyles: Record<ButtonVariant, TextStyle> = {
  primary:   { color: '#ffffff' },
  secondary: { color: NEURAL.onSurface },
  outline:   { color: NEURAL.onSurfaceVariant },
  ghost:     { color: NEURAL.primary },
  error:     { color: NEURAL.error },
  success:   { color: NEURAL.tertiary },
};
