import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Button } from './ui/Button';
import { AppIcon } from './ui/AppIcon';
import { FONT_SIZE, FONT_WEIGHT, SPACING, RADIUS } from '../theme/colors';

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

export class ErrorBoundary extends React.Component<{ children: React.ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
    message: '',
  };

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    return {
      hasError: true,
      message: error instanceof Error ? error.message : 'Unexpected application error',
    };
  }

  componentDidCatch(error: unknown, errorInfo: unknown) {
    console.error('Mobile app runtime crash captured by ErrorBoundary', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, message: '' });
  };

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <View style={styles.container}>
        <View style={styles.card}>
          <AppIcon name="alert-circle-outline" size={28} color="#e11d48" />
          <Text style={styles.title}>Something went wrong</Text>
          <Text style={styles.message}>{this.state.message || 'An unexpected error occurred.'}</Text>
          <Button label="Try Again" onPress={this.handleReset} size="sm" />
        </View>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
    alignItems: 'center',
    justifyContent: 'center',
    padding: SPACING.xl,
  },
  card: {
    width: '100%',
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#fecdd3',
    borderRadius: RADIUS['2xl'],
    padding: SPACING.xl,
    alignItems: 'center',
    gap: SPACING.md,
  },
  title: {
    fontSize: FONT_SIZE.lg,
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
  },
  message: {
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
    textAlign: 'center',
    lineHeight: 18,
  },
});
