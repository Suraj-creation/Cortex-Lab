/**
 * LoadingSpinner — Cortex Aurora indigo spinner
 */
import React, { useEffect, useRef } from 'react';
import { Animated, View, StyleSheet, Text } from 'react-native';
import { FONT_SIZE, FONT_WEIGHT, SPACING } from '../../theme/colors';

interface LoadingSpinnerProps {
  size?: number;
  color?: string;
  message?: string;
  fullScreen?: boolean;
}

export function LoadingSpinner({
  size = 36,
  color = '#6366f1',
  message,
  fullScreen = false,
}: LoadingSpinnerProps) {
  const rotation = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.timing(rotation, {
        toValue: 1,
        duration: 900,
        useNativeDriver: true,
      }),
    );
    loop.start();
    return () => loop.stop();
  }, [rotation]);

  const spin = rotation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  const content = (
    <View style={styles.inner}>
      <Animated.View
        style={[
          styles.spinner,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            borderColor: `${color}20`,
            borderTopColor: color,
            borderWidth: size > 24 ? 3 : 2,
            transform: [{ rotate: spin }],
          },
        ]}
      />
      {message && <Text style={styles.message}>{message}</Text>}
    </View>
  );

  if (fullScreen) {
    return <View style={styles.fullScreen}>{content}</View>;
  }

  return content;
}

const styles = StyleSheet.create({
  inner: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: SPACING.lg,
  },
  fullScreen: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f8fafc',
  },
  spinner: {
    borderStyle: 'solid',
  },
  message: {
    marginTop: SPACING.md,
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
    fontWeight: FONT_WEIGHT.medium,
  },
});
