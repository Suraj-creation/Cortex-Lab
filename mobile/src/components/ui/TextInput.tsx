/**
 * TextInput — Cortex Aurora light input
 * Clean borders, indigo focus ring, clear button support
 */
import React, { useState } from 'react';
import {
  TextInput as RNTextInput,
  View,
  StyleSheet,
  TextInputProps as RNTextInputProps,
  ViewStyle,
  Text,
  TouchableOpacity,
} from 'react-native';
import { RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../../theme/colors';

interface TextInputProps extends RNTextInputProps {
  label?: string;
  style?: ViewStyle;
  inputStyle?: object;
  pill?: boolean;
  icon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  error?: string;
  onClear?: () => void;
}

export function TextInput({
  label,
  style,
  inputStyle,
  pill = false,
  icon,
  rightIcon,
  error,
  onClear,
  multiline,
  value,
  ...rest
}: TextInputProps) {
  const [focused, setFocused] = useState(false);

  return (
    <View style={[styles.wrapper, style]}>
      {label && <Text style={styles.label}>{label}</Text>}
      <View
        style={[
          styles.container,
          pill && styles.pill,
          multiline && styles.multiline,
          focused && styles.focused,
          error ? styles.errorBorder : undefined,
        ]}
      >
        {icon && <View style={styles.icon}>{icon}</View>}
        <RNTextInput
          {...rest}
          value={value}
          multiline={multiline}
          onFocus={(e) => {
            setFocused(true);
            rest.onFocus?.(e);
          }}
          onBlur={(e) => {
            setFocused(false);
            rest.onBlur?.(e);
          }}
          style={[styles.input, multiline && styles.inputMultiline, inputStyle]}
          placeholderTextColor="#94a3b8"
          selectionColor="#6366f1"
        />
        {onClear && value ? (
          <TouchableOpacity onPress={onClear} style={styles.clearButton}>
            <Text style={styles.clearText}>✕</Text>
          </TouchableOpacity>
        ) : rightIcon ? (
          <View style={styles.rightIcon}>{rightIcon}</View>
        ) : null}
      </View>
      {error && <Text style={styles.errorText}>{error}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {},
  label: {
    fontSize: FONT_SIZE.sm,
    color: '#475569',
    fontWeight: FONT_WEIGHT.medium,
    marginBottom: 6,
  },
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#edf2fb',
    borderRadius: RADIUS.xl,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: '#ffffff',
    ...SHADOWS.md,
  },
  pill: {
    borderRadius: RADIUS.full,
  },
  multiline: {
    alignItems: 'flex-start',
    paddingVertical: 12,
  },
  focused: {
    borderColor: '#6366f1',
    backgroundColor: '#f8fbff',
    ...SHADOWS.glow,
  },
  errorBorder: {
    borderColor: '#f43f5e',
  },
  input: {
    flex: 1,
    fontSize: FONT_SIZE.base,
    color: '#0f172a',
    padding: 0,
    margin: 0,
  },
  inputMultiline: {
    minHeight: 60,
    textAlignVertical: 'top',
  },
  icon: {
    marginRight: 8,
  },
  rightIcon: {
    marginLeft: 8,
  },
  clearButton: {
    marginLeft: 8,
    padding: 4,
    backgroundColor: '#ffffff',
    borderRadius: 10,
    width: 20,
    height: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  clearText: {
    fontSize: 10,
    color: '#64748b',
    fontWeight: '600',
  },
  errorText: {
    fontSize: FONT_SIZE.xs,
    color: '#e11d48',
    marginTop: 4,
    marginLeft: 4,
  },
});
