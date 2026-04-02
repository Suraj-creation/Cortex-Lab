# Railway Deployment (Monorepo)

This repository can deploy both backend and frontend from one GitHub repo.

## Architecture

- One repository: Cortex-Lab
- Two Railway services from same repo:
  - backend service (root directory: backend)
  - frontend service (root directory: frontend)
- Mobile app calls backend over public HTTPS URL.
- Frontend also calls same backend URL.

## Is One Repo Fine?

Yes. Monorepo is a good setup here.

Benefits:
- single source of truth
- easier API/frontend compatibility updates
- easier shared release tagging

## Service 1: Backend (Railway)

Create service in Railway:
- Source repo: this repo
- Root Directory: backend
- Build Command: pip install -r requirements.txt
- Start Command: uvicorn server:app --host 0.0.0.0 --port $PORT

Required variables:
- SKIP_LOCAL_MODEL=true
- GOOGLE_API_KEY=...

Recommended variables:
- CORS_ALLOW_ORIGINS=https://<frontend-service>.up.railway.app
- CORS_ALLOW_ORIGIN_REGEX=^https://.*\.up\.railway\.app$

Optional security:
- CORTEX_API_KEY=<strong secret>
  - If enabled, clients must send Authorization bearer token.

Important:
- Railway filesystem is ephemeral by default.
- If you want persistent memories/documents in backend/data, attach a Railway volume mounted to /app/data.

## Service 2: Frontend (Railway)

Create another service in Railway:
- Source repo: same repo
- Root Directory: frontend
- Build Command: npm install && npm run build
- Start Command: npm run start

Required variable:
- NEXT_PUBLIC_API_BASE_URL=https://<backend-service>.up.railway.app/api

## Mobile App Connection

Set in Expo/EAS environment:
- EXPO_PUBLIC_API_BASE_URL=https://<backend-service>.up.railway.app/api

Then rebuild APK so the URL is baked into app config.

## Production Checklist

1. Confirm backend health endpoint:
   - GET https://<backend-service>.up.railway.app/api/health
2. Confirm frontend loads and API calls succeed.
3. Rebuild mobile APK with EXPO_PUBLIC_API_BASE_URL set.
4. Keep secrets only in Railway/Expo envs, never in git.
5. Rotate any key immediately if accidentally exposed.
