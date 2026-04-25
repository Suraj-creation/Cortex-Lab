/**
 * SearchBar — Dedicated search component with debounce, clear, and icon
 */
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { View, TextInput, TouchableOpacity, StyleSheet, Text, ViewStyle } from 'react-native';
import { RADIUS, FONT_SIZE, FONT_WEIGHT, SPACING, SHADOWS } from '../../theme/colors';
import { AppIcon } from './AppIcon';

interface SearchBarProps {
  value: string;
  onChangeText: (text: string) => void;
  onSubmit?: () => void;
  placeholder?: string;
  debounceMs?: number;
  style?: ViewStyle;
  autoFocus?: boolean;
}

export function SearchBar({
  value,
  onChangeText,
  onSubmit,
  placeholder = 'Search...',
  debounceMs = 0,
  style,
  autoFocus = false,
}: SearchBarProps) {
  const [focused, setFocused] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleChange = useCallback((text: string) => {
    if (debounceMs > 0) {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => onChangeText(text), debounceMs);
    } else {
      onChangeText(text);
    }
  }, [onChangeText, debounceMs]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  return (
    <View style={[styles.container, focused && styles.focused, style]}>
      <AppIcon name="magnify" size={18} color="#94a3b8" />
      <TextInput
        value={value}
        onChangeText={debounceMs > 0 ? handleChange : onChangeText}
        placeholder={placeholder}
        placeholderTextColor="#94a3b8"
        style={styles.input}
        selectionColor="#6366f1"
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onSubmitEditing={onSubmit}
        returnKeyType="search"
        autoFocus={autoFocus}
        autoCorrect={false}
      />
      {value.length > 0 && (
        <TouchableOpacity onPress={() => onChangeText('')} style={styles.clearButton}>
          <AppIcon name="close-circle" size={16} color="#94a3b8" />
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f1f5f9',
    borderRadius: RADIUS.xl,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  focused: {
    backgroundColor: '#ffffff',
    borderColor: '#6366f1',
    ...SHADOWS.glow,
  },
  input: {
    flex: 1,
    fontSize: FONT_SIZE.base,
    color: '#0f172a',
    padding: 0,
    margin: 0,
  },
  clearButton: {
    padding: 2,
  },
});
