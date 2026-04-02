# Mobile Migration Runbook

## Objective
Run web, backend, and mobile clients in parallel during migration with shared contracts and stable developer workflow.

## Services
- Backend: FastAPI server on 8000
- Web: Next.js app on 3000
- Mobile: Expo dev server

## Local Bring-Up

## 1) Backend
From repository root:
- cd backend
- python server.py

## 2) Web
From repository root:
- cd frontend
- npm install
- npm run dev

## 3) Mobile
From repository root:
- cd mobile
- npm install
- copy .env.example to .env and update EXPO_PUBLIC_API_BASE_URL
- npm run start

## API URL Guidance
- Android emulator: http://10.0.2.2:8000/api
- iOS simulator: http://localhost:8000/api
- Physical device: http://YOUR_MACHINE_LAN_IP:8000/api

## Debug Checklist
1. Health check fails in mobile:
- Verify backend is running.
- Verify EXPO_PUBLIC_API_BASE_URL is reachable from device/emulator.

2. Requests timeout:
- Ensure firewall allows inbound to 8000.
- Confirm same network for physical device.

3. Shared import resolution issues:
- Ensure mobile/metro.config.js includes ../shared watch folder.
- Restart Expo with cache clear if needed.

## Migration Order
1. Chat
2. Memory
3. Documents
4. Settings
5. Observability
6. Ambient voice
7. Knowledge graph

## Quality Gates
- Type checks pass in mobile
- Backend API health reachable from mobile
- Chat roundtrip success on emulator/device
