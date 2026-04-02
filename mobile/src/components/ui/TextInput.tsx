import React, { useState } from "react";
import { View, TextInput as RNTextInput, StyleSheet, ViewStyle, TextStyle } from "react-native";
import { COLORS, SEMANTIC_COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from "../../theme/colors";

interface TextInputProps {
  placeholder?: string;
  value: string;
  onChangeText: (text: string) => void;
  multiline?: boolean;
  maxHeight?: number;
  editable?: boolean;
  numberOfLines?: number;
  style?: ViewStyle;
  inputStyle?: TextStyle;
}

export function TextInput({
  placeholder,
  value,
  onChangeText,
  multiline = false,
  maxHeight = 120,
  editable = true,
  numberOfLines,
  style,
  inputStyle,
}: TextInputProps) {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <View
      style={[
        styles.container,
        isFocused && styles.containerFocused,
        style,
      ]}
    >
      <RNTextInput
        placeholder={placeholder}
        placeholderTextColor={SEMANTIC_COLORS.textTertiary}
        value={value}
        onChangeText={onChangeText}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        multiline={multiline}
        editable={editable}
        numberOfLines={numberOfLines}
        textAlignVertical={multiline ? "top" : "center"}
        style={[
          styles.input,
          multiline && { maxHeight },
          inputStyle,
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: BORDER_RADIUS.xl,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    flexDirection: "row",
    alignItems: "center",
    ...SHADOWS.none,
  },
  containerFocused: {
    borderColor: SEMANTIC_COLORS.borderAccent,
    borderWidth: 1,
    backgroundColor: COLORS.white,
    ...SHADOWS.sm,
  },
  input: {
    flex: 1,
    fontSize: TYPOGRAPHY.fontSize.md,
    color: SEMANTIC_COLORS.textPrimary,
    paddingVertical: SPACING.sm,
    minHeight: 22,
    lineHeight: TYPOGRAPHY.fontSize.md * TYPOGRAPHY.lineHeight.normal,
  },
});
