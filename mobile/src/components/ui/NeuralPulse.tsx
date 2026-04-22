/**
 * NeuralPulse — Animated status indicator
 * 8px circle (secondary color) with scaled outer ring animation
 * Design: "The Neural Pulse" from Cortex Neural Dark design system
 */
import React, { useEffect, useRef } from 'react';
import { Animated, View, StyleSheet } from 'react-native';
import { NEURAL } from '../../theme/colors';

interface NeuralPulseProps {
  color?: string;
  size?: number;
  active?: boolean;
  style?: object;
}

export function NeuralPulse({
  color = NEURAL.secondary,
  size = 8,
  active = true,
  style,
}: NeuralPulseProps) {
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const opacityAnim = useRef(new Animated.Value(0.5)).current;

  useEffect(() => {
    if (!active) {
      scaleAnim.setValue(1);
      opacityAnim.setValue(0.3);
      return;
    }

    const loop = Animated.loop(
      Animated.parallel([
        Animated.sequence([
          Animated.timing(scaleAnim, {
            toValue: 2.2,
            duration: 1100,
            useNativeDriver: true,
          }),
          Animated.timing(scaleAnim, {
            toValue: 1,
            duration: 0,
            useNativeDriver: true,
          }),
        ]),
        Animated.sequence([
          Animated.timing(opacityAnim, {
            toValue: 0,
            duration: 1100,
            useNativeDriver: true,
          }),
          Animated.timing(opacityAnim, {
            toValue: 0.5,
            duration: 0,
            useNativeDriver: true,
          }),
        ]),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [active, scaleAnim, opacityAnim]);

  return (
    <View style={[{ width: size * 3, height: size * 3, alignItems: 'center', justifyContent: 'center' }, style]}>
      {/* Outer ring */}
      <Animated.View
        style={{
          position: 'absolute',
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor: color,
          opacity: opacityAnim,
          transform: [{ scale: scaleAnim }],
        }}
      />
      {/* Core dot */}
      <View
        style={{
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor: color,
        }}
      />
    </View>
  );
}
