import React, { createContext, useContext } from 'react';
import { Text } from 'react-native';
import type { StyleProp, TextStyle } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';

export type AppIconName = React.ComponentProps<typeof MaterialCommunityIcons>['name'];

export const IconFontReadyContext = createContext(true);

const FALLBACK_GLYPHS: Record<string, string> = {
  'account-voice': 'VO',
  'alert-circle-outline': '!',
  'arrow-left': '<',
  'arrow-right': '>',
  'arrow-up': '^',
  'atom-variant': 'AT',
  'book-open-page-variant-outline': 'BK',
  'brain': 'AI',
  'cellphone-arrow-down': 'DL',
  'chart-bar': 'CB',
  'chart-timeline-variant': 'CT',
  'chat-processing-outline': 'CH',
  'check-circle-outline': 'OK',
  'check-decagram-outline': 'OK',
  'chevron-down': 'v',
  'chevron-up': '^',
  'chip': 'CP',
  'clock-outline': 'CL',
  'close': 'X',
  'close-circle': 'X',
  'cloud-check-outline': 'CL',
  'cloud-outline': 'CL',
  'cloud-upload-outline': 'UP',
  'cog-outline': 'SG',
  'content-copy': 'CP',
  'content-save-outline': 'SV',
  'database-outline': 'DB',
  'database-plus-outline': '+',
  'delete-outline': 'DL',
  'dots-horizontal': '...',
  'file-document-outline': 'DOC',
  'file-upload-outline': 'UP',
  'format-list-bulleted': 'LS',
  'graph-outline': 'KG',
  'history': 'HS',
  'information-outline': 'i',
  'lightning-bolt-outline': 'LT',
  'magnify': 'SR',
  'microphone': 'MIC',
  'microphone-message': 'MIC',
  'microphone-outline': 'MIC',
  'notebook-plus-outline': 'NB',
  'package-variant-closed': 'PK',
  'paperclip': 'PC',
  'pipe': '|',
  'play-circle-outline': 'GO',
  'plus': '+',
  'refresh': 'RF',
  'robot-outline': 'AI',
  'server-network': 'SV',
  'stop': 'ST',
  'text-search': 'TS',
  'timeline-text-outline': 'TL',
  'tune-variant': 'TN',
  'view-dashboard-outline': 'DB',
  'waveform': 'WF',
  'wifi-off': 'NO',
  'wrench-outline': 'WR',
};

function buildFallbackGlyph(name: string): string {
  const glyph = FALLBACK_GLYPHS[name];
  if (glyph) {
    return glyph;
  }

  return (
    name
      .split(/[-_]/)
      .map((part) => part.trim()[0] || '')
      .join('')
      .slice(0, 2)
      .toUpperCase() || '*'
  );
}

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
  const fontReady = useContext(IconFontReadyContext);
  const glyphMap =
    (MaterialCommunityIcons as unknown as { glyphMap?: Record<string, number | string> }).glyphMap || {};
  const hasNamedGlyph = Object.prototype.hasOwnProperty.call(glyphMap, String(name));

  if (!fontReady || !hasNamedGlyph) {
    return (
      <Text
        style={[
          {
            fontSize: Math.max(11, size * 0.72),
            lineHeight: size,
            color: color || '#475569',
            fontWeight: '700',
            textAlign: 'center',
            minWidth: size,
          },
          style,
        ]}
      >
        {buildFallbackGlyph(String(name))}
      </Text>
    );
  }

  return <MaterialCommunityIcons name={name} size={size} color={color} style={style} />;
}
