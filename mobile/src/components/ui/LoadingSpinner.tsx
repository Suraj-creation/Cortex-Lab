import React, { useEffect, useRef } from "react";
import { View, Animated, StyleSheet, Text } from "react-native";
import { COLORS, SEMANTIC_COLORS, SPACING, TYPOGRAPHY } from "../../theme/colors";

interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg";
  message?: string;
  fullscreen?: boolean;
}

export function LoadingSpinner({
  size = "md",
  message,
  fullscreen = false,
}: LoadingSpinnerProps) {
  const spinValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.loop(
      Animated.timing(spinValue, {
        toValue: 1,
        duration: 1500,
        useNativeDriver: true,
      })
    ).start();
  }, [spinValue]);

  const spin = spinValue.interpolate({
    inputRange: [0, 1],
    outputRange: ["0deg", "360deg"],
  });

  const sizeMap = {
    sm: { width: 24, height: 24, borderWidth: 2 },
    md: { width: 40, height: 40, borderWidth: 3 },
    lg: { width: 56, height: 56, borderWidth: 4 },
  };

  const spinnerSize = sizeMap[size];

  if (fullscreen) {
    return (
      <View style={styles.fullscreenContainer}>
        <Animated.View
          style={[
            styles.spinner,
            spinnerSize,
            {
              borderColor: COLORS.primary[500],
              borderTopColor: COLORS.primary[200],
              borderRightColor: COLORS.primary[200],
              borderBottomColor: COLORS.primary[200],
              transform: [{ rotate: spin }],
            },
          ]}
        />
        {message ? <Text style={styles.message}>{message}</Text> : null}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Animated.View
        style={[
          styles.spinner,
          spinnerSize,
          {
            borderColor: COLORS.primary[500],
            borderTopColor: COLORS.primary[200],
            borderRightColor: COLORS.primary[200],
            borderBottomColor: COLORS.primary[200],
            transform: [{ rotate: spin }],
          },
        ]}
      />
      {message ? <Text style={styles.message}>{message}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    justifyContent: "center",
    alignItems: "center",
    paddingVertical: SPACING["4xl"],
  },
  fullscreenContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: COLORS.white,
  },
  spinner: {
    borderRadius: 999,
    borderStyle: "solid",
  },
  message: {
    fontSize: TYPOGRAPHY.fontSize.md,
    color: SEMANTIC_COLORS.textSecondary,
    marginTop: SPACING.lg,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
});
