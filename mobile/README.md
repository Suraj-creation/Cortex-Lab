# Cortex Mobile (Expo)

This is the React Native + Expo client for Cortex Lab.

## Current Status
- Expo TypeScript app scaffolded
- Shared core types and API client integrated from ../shared
- Chat MVP working against backend chat endpoints
- Health status indicator connected to /api/health
- Memory, graph, dashboard, observability, ambient, and documents tabs wired to backend APIs

## Prerequisites
- Node.js 18+
- Expo CLI (optional, npx is enough)
- Running Cortex backend at port 8000

## Configure API URL
API URL now auto-resolves by platform:
- Expo web: uses current browser host on port 8000
- Android emulator: defaults to 10.0.2.2:8000
- Native with EXPO_PUBLIC_API_BASE_URL set: uses explicit override

Optional explicit override in mobile/.env:

EXPO_PUBLIC_API_BASE_URL=http://YOUR_LAN_IP:8000/api

Notes:
- Android emulator default loopback is 10.0.2.2
- Physical device must use machine LAN IP

## Run
1. npm install
2. npm run start
3. press a for Android emulator, i for iOS simulator, or scan QR with Expo Go

## Next Migration Steps
1. Implement streaming transport adapter (SSE/WebSocket fallback)
2. Add conversation persistence (AsyncStorage)
3. Port Memory browser and Documents panel
4. Port Settings and provider toggle
5. Port observability dashboards and ambient voice flows
