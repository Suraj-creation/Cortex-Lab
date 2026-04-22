# Mobile UI Components - Quick Start Guide

## Overview

This guide shows how to use the new Cortex Lab mobile UI component library and theme system.

## Theme System

### Import Theme

```typescript
import { 
  COLORS, 
  SEMANTIC_COLORS, 
  SPACING, 
  TYPOGRAPHY, 
  BORDER_RADIUS, 
  SHADOWS 
} from "@/theme/colors";
```

### Example: Using Colors

```typescript
const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.white,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    padding: SPACING.lg,
    borderRadius: BORDER_RADIUS.lg,
  },
  text: {
    fontSize: TYPOGRAPHY.fontSize.md,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textPrimary,
  },
});
```

## UI Components

### Button

```typescript
import { Button } from "@/components/ui";

<Button
  label="Click me"
  variant="primary"      // primary | secondary | outline | error
  size="md"              // sm | md | lg
  onPress={() => {}}
  disabled={false}
  loading={false}
/>
```

### Card

```typescript
import { Card } from "@/components/ui";

<Card variant="outlined" padding="lg">
  <Text>Card content here</Text>
</Card>
```

### Header

```typescript
import { Header } from "@/components/ui";

<Header
  modelStatus={modelStatus}
  title="Cortex Lab"
  subtitle="Chat"
  onSettingsPress={() => {}}
/>
```

### TextInput

```typescript
import { TextInput } from "@/components/ui";

<TextInput
  placeholder="Type here..."
  value={text}
  onChangeText={setText}
  multiline={true}
  maxHeight={120}
/>
```

### Badge

```typescript
import { Badge } from "@/components/ui";

<Badge
  label="Success"
  variant="success"      // default | success | warning | error | info | primary
  small={false}
/>
```

### BottomNav

```typescript
import { BottomNav } from "@/components/ui";

const items = [
  { key: "chat", label: "Chat" },
  { key: "memories", label: "Memory" },
  // ... more items
];

<BottomNav
  items={items}
  activeKey={activeView}
  onSelect={(key) => setActiveView(key)}
/>
```

### LoadingSpinner

```typescript
import { LoadingSpinner } from "@/components/ui";

// Inline
<LoadingSpinner size="md" message="Loading..." />

// Fullscreen
<LoadingSpinner size="lg" fullscreen message="Loading app..." />
```

### EmptyState

```typescript
import { EmptyState } from "@/components/ui";

<EmptyState
  title="No messages"
  description="Start a conversation to get started"
  icon={<Icon />}
/>
```

### Layout Helpers

```typescript
import { Screen, Spacer, Section } from "@/components/ui";

<Screen scrollable paddingHorizontal={12}>
  <Section gap="md">
    <Text>Section 1</Text>
    <Spacer size="lg" />
    <Text>Section 2</Text>
  </Section>
</Screen>
```

## Creating New Styled Components

### Pattern 1: Simple Component

```typescript
import { View, Text, StyleSheet } from "react-native";
import { COLORS, SPACING, TYPOGRAPHY } from "@/theme/colors";

interface MyComponentProps {
  title: string;
}

export function MyComponent({ title }: MyComponentProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>{title}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: SPACING.lg,
    backgroundColor: COLORS.white,
  },
  title: {
    fontSize: TYPOGRAPHY.fontSize.lg,
    fontWeight: TYPOGRAPHY.fontWeight.bold,
    color: COLORS.surface[900],
  },
});
```

### Pattern 2: Using Cards

```typescript
import { Card } from "@/components/ui";

export function MyCard() {
  return (
    <Card variant="elevated" style={styles.card}>
      <Text>Content here</Text>
    </Card>
  );
}
```

### Pattern 3: Using Color Variants

```typescript
const variantStyles = {
  primary: { bg: COLORS.primary[50], text: COLORS.primary[700] },
  success: { bg: COLORS.success[50], text: COLORS.success[700] },
};

// Then use based on variant prop
const style = variantStyles[variant];
<View style={{ backgroundColor: style.bg }}>
  <Text style={{ color: style.text }}>Text</Text>
</View>
```

## Spacing Examples

```typescript
// Use semantic spacing
padding: SPACING.lg,           // default 12px
gap: SPACING.md,               // 8px
marginBottom: SPACING.xl,      // 16px
marginHorizontal: SPACING.sm,  // 4px
```

## Typography Examples

```typescript
// Always use typography constants
fontSize: TYPOGRAPHY.fontSize.md,      // 15px
fontWeight: TYPOGRAPHY.fontWeight.semibold,  // "600"
lineHeight: TYPOGRAPHY.lineHeight.relaxed,   // 1.75x
```

## Color Usage Guidelines

### Text Colors
```typescript
color: SEMANTIC_COLORS.textPrimary       // Main text (#0f172a)
color: SEMANTIC_COLORS.textSecondary     // Muted text (#475569)
color: SEMANTIC_COLORS.textTertiary      // Even more muted (#64748b)
color: SEMANTIC_COLORS.textMuted         // Least important (#94a3b8)
```

### Background Colors
```typescript
backgroundColor: SEMANTIC_COLORS.bgPrimary     // White
backgroundColor: SEMANTIC_COLORS.bgSecondary  // Light gray (#f8fafc)
backgroundColor: SEMANTIC_COLORS.bgTertiary   // Slightly darker (#f1f5f9)
```

### Status Colors
```typescript
// Use semantic status colors
backgroundColor: SEMANTIC_COLORS.statusSuccess  // Green
backgroundColor: SEMANTIC_COLORS.statusWarning // Amber
backgroundColor: SEMANTIC_COLORS.statusError   // Red
backgroundColor: SEMANTIC_COLORS.statusLoading // Indigo
```

## Common Patterns

### View with Shadow

```typescript
const styles = StyleSheet.create({
  container: {
    ...SHADOWS.md,
  },
});
```

### Pressable Button with Feedback

```typescript
<Pressable style={({ pressed }) => [
  styles.button,
  pressed && styles.buttonPressed
]}>
  <Text>Press me</Text>
</Pressable>

const styles = StyleSheet.create({
  button: {
    opacity: 1,
  },
  buttonPressed: {
    opacity: 0.7,
  },
});
```

### Conditional Styling

```typescript
<View style={[
  styles.container,
  isActive && styles.containerActive,
  isError && styles.containerError,
]}>
  <Text>Content</Text>
</View>
```

## Accessibility Tips

1. **Button Labels**: Always provide clear, descriptive labels
   ```typescript
   <Button label="Send Message" />  // Good
   <Button label=">" />             // Bad
   ```

2. **Color Contrast**: Use heading hierarchy for text
   ```typescript
   fontSize: TYPOGRAPHY.fontSize.lg,
   fontWeight: TYPOGRAPHY.fontWeight.bold,
   ```

3. **Touch Targets**: Ensure buttons are at least 44x44pt
   ```typescript
   paddingVertical: SPACING.md,
   paddingHorizontal: SPACING.lg,
   ```

## Performance Tips

1. Use `useMemo` for complex styling calculations
2. Avoid creating styles inside render functions
3. Use `StyleSheet.create()` for static styles
4. Use `useCallback` for prop callbacks
5. Memoize animated values with `useRef`

## Color Reference

### Primary Brand Colors
- **Indigo 500**: #6366f1 (default button)
- **Indigo 600**: #4f46e5 (hover)
- **Indigo 700**: #4338ca (active)

### Semantic Colors
- **Success**: #22c55e (green)
- **Warning**: #f59e0b (amber)
- **Error**: #ef4444 (red)
- **Info**: #3b82f6 (blue)

### Neutral Colors
- **White**: #ffffff
- **Surface 50**: #f8fafc (lightest)
- **Surface 900**: #0f172a (darkest)

## Common Issues & Solutions

### Issue: Text not visible
**Solution**: Check color contrast
```typescript
// Bad: light text on light bg
color: "#ffffff",
backgroundColor: "#f1f5f9",

// Good: dark text on light bg
color: SEMANTIC_COLORS.textPrimary,
backgroundColor: SEMANTIC_COLORS.bgSecondary,
```

### Issue: Spacing looks wrong
**Solution**: Use spacing tokens consistently
```typescript
// Bad: mixing units
padding: 12,
margin: "8px",
gap: 0.5,

// Good: consistent spacing tokens
padding: SPACING.lg,      // 12px
margin: SPACING.sm,       // 4px
gap: SPACING.md,          // 8px
```

### Issue: Component not resizing
**Solution**: Add `flex: 1` or set explicit width/height
```typescript
// For flexible width
flex: 1,

// For explicit width
width: "100%",
maxWidth: 300,
```

## Documentation Links

- **React Native Docs**: https://reactnative.dev/docs/intro
- **Expo Docs**: https://docs.expo.dev/
- **TypeScript**: https://www.typescriptlang.org/docs/

## Questions?

See the main app implementation in `App.tsx` for real-world usage examples.
