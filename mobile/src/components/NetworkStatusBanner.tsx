import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { AppIcon } from './ui/AppIcon';
import { useNetworkStatus } from '../providers/NetworkProvider';
import { FONT_SIZE, FONT_WEIGHT, SPACING } from '../theme/colors';

export function NetworkStatusBanner() {
  const { isOffline, connectionType } = useNetworkStatus();

  if (!isOffline) {
    return null;
  }

  return (
    <View style={styles.container}>
      <AppIcon name="wifi-off" size={14} color="#b45309" />
      <Text style={styles.text}>Offline mode active • Connection: {connectionType}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.xs,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.xs,
    backgroundColor: '#fef3c7',
    borderBottomWidth: 1,
    borderBottomColor: '#fde68a',
  },
  text: {
    fontSize: FONT_SIZE.xs,
    color: '#92400e',
    fontWeight: FONT_WEIGHT.semibold,
  },
});
