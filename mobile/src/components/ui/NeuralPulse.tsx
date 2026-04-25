/**
 * NeuralPulse — Cortex Aurora activity indicator
 * Animated pulsing dot for streaming/processing states
 */
import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, View, ViewStyle } from 'react-native';

interface NeuralPulseProps {
  active?: boolean;
  size?: number;
  color?: string;
  style?: ViewStyle;
}

export function NeuralPulse({
  active = true,
  size = 10,
  color = '#6366f1',
  style,
}: NeuralPulseProps) {
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const opacityAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (!active) {
      scaleAnim.setValue(1);
      opacityAnim.setValue(0.5);
      return;
    }

    const pulse = Animated.loop(
      Animated.sequence([
        Animated.parallel([
          Animated.timing(scaleAnim, {
            toValue: 1.6,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(opacityAnim, {
            toValue: 0.3,
            duration: 800,
            useNativeDriver: true,
          }),
        ]),
        Animated.parallel([
          Animated.timing(scaleAnim, {
            toValue: 1,
            duration: 600,
            useNativeDriver: true,
          }),
          Animated.timing(opacityAnim, {
            toValue: 1,
            duration: 600,
            useNativeDriver: true,
          }),
        ]),
      ]),
    );
    pulse.start();
    return () => pulse.stop();
  }, [active, scaleAnim, opacityAnim]);

  return (
    <View style={[styles.container, { width: size * 2, height: size * 2 }, style]}>
      <Animated.View
        style={[
          styles.outerRing,
          {
            width: size * 2,
            height: size * 2,
            borderRadius: size,
            borderColor: `${color}40`,
            transform: [{ scale: scaleAnim }],
            opacity: opacityAnim,
          },
        ]}
      />
      <View
        style={[
          styles.core,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            backgroundColor: color,
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  outerRing: {
    position: 'absolute',
    borderWidth: 1.5,
    borderStyle: 'solid',
  },
  core: {},
});
