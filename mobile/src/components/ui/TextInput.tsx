/**
 * TextInput — Neural Dark command-line input
 * "The Command Line" — surface-container-highest pill, no border, primary glow on focus
 */
import React, { useState } from 'react';
import {
  TextInput as RNTextInput,
  View,
  StyleSheet,
  TextInputProps as RNTextInputProps,
  ViewStyle,
  Text,
} from 'react-native';
import { NEURAL, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../../theme/colors';

interface TextInputProps extends RNTextInputProps {
  label?: string;
  style?: ViewStyle;
  inputStyle?: object;
  pill?: boolean;
  icon?: React.ReactNode;
}

export function TextInput({
  label,
  style,
  inputStyle,
  pill = false,
  icon,
  multiline,
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
        ]}
      >
        {icon && <View style={styles.icon}>{icon}</View>}
        <RNTextInput
          {...rest}
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
          placeholderTextColor={NEURAL.outline}
          selectionColor={NEURAL.primary}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {},
  label: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurfaceVariant,
    fontWeight: FONT_WEIGHT.medium,
    marginBottom: 6,
  },
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: NEURAL.surfaceContainerHighest,
    borderRadius: RADIUS.xl,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
  },
  pill: {
    borderRadius: RADIUS.full,
  },
  multiline: {
    alignItems: 'flex-start',
    paddingVertical: 12,
  },
  focused: {
    borderColor: NEURAL.primary,
    shadowColor: NEURAL.primary,
    shadowOpacity: 0.25,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 0 },
    elevation: 4,
  },
  input: {
    flex: 1,
    fontSize: FONT_SIZE.base,
    color: NEURAL.onSurface,
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
});
