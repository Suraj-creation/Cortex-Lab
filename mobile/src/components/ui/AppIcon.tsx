import React from 'react';
import type { StyleProp, TextStyle } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';

export type AppIconName = React.ComponentProps<typeof MaterialCommunityIcons>['name'];

interface AppIconProps {
  name: AppIconName;
  size?: number;
  color?: string;
  style?: StyleProp<TextStyle>;
}

export function AppIcon({
  name,
  size = 18,
  color,
  style,
}: AppIconProps) {
  return <MaterialCommunityIcons name={name} size={size} color={color} style={style} />;
}
