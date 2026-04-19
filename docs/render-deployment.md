# Deploy Cortex Lab on Render (Frontend + Backend, No Paid Resources)

This repo includes a free-tier-friendly Render Blueprint at `render.yaml` that provisions:

- `cortex-backend` (FastAPI web service)
- `cortex-frontend` (Next.js web service)

It intentionally avoids managed PostgreSQL and persistent disks so Render can deploy without paid-instance requirements.

## 1. Deploy via Blueprint

1. Push this repository to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select your repo and apply `render.yaml`.
4. Render will create both services automatically.

## 2. Required Secrets

Set these in Render for the backend service:

- `GOOGLE_API_KEY` (required for Gemini LLM/voice paths)

Optional but recommended:

- `CORTEX_API_KEY` (adds bearer-token API protection)

## 3. Runtime Defaults Used

Backend defaults in blueprint:

- `SKIP_LOCAL_MODEL=true` (Gemini-first cloud mode)
- `HOST=0.0.0.0`
- `CORTEX_DATA_DIR=/tmp/cortex` (ephemeral filesystem)
- `CORS_ALLOW_ORIGINS=https://cortex-frontend.onrender.com`
- `CORS_ALLOW_ORIGIN_REGEX=^https://.*\.onrender\.com$|^https://.*\.trycloudflare\.com$|^https?://(localhost|127\.0\.0\.1)(:\d+)?$`

Frontend defaults in blueprint:

- `NEXT_PUBLIC_API_BASE_URL=https://cortex-backend.onrender.com/api`
- `NODE_VERSION=20.18.0`

## 4. Important Database Note

Current app storage is DuckDB/FAISS and writes to `CORTEX_DATA_DIR`.

That means:

- On this free-tier blueprint, cloud data is ephemeral and may reset after restart/redeploy.
- Core functionality remains available for both web and mobile clients.
- Your long-term durable/local-first data model should run on user-managed local storage.
- You can add persistent disk/Postgres later without changing application APIs.

## 5. Post-Deploy Verification

Run these checks after services are live:

1. Backend health:
   - `GET https://cortex-backend.onrender.com/api/health`
2. Frontend load:
   - Open `https://cortex-frontend.onrender.com`
3. API proxy from frontend:
   - In browser devtools, confirm `/api/*` requests return backend responses.
4. CORS:
   - Confirm no CORS errors in frontend console/network panel.
5. Persistence:
   - Add memory, restart/redeploy backend, verify app still functions.
   - Data reset across redeploy is expected on this free-tier setup.

## 6. If You Rename Services

If you change service names in Render, update:

- backend `CORS_ALLOW_ORIGINS`
- frontend `NEXT_PUBLIC_API_BASE_URL`
- any docs/scripts using default `cortex-backend` or `cortex-frontend` hostnames

## 7. Upgrade Path (When You Need Durable Cloud Data)

Later, you can upgrade with minimal impact:

1. Add Render persistent disk and set `CORTEX_DATA_DIR=/var/data/cortex`.
2. Add Render PostgreSQL and wire `DATABASE_URL`.
3. Keep frontend/mobile API URLs unchanged.
