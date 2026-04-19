# Deploy Cortex Lab on Render (Frontend + Backend + Database)

This repo now includes a Render Blueprint at `render.yaml` that provisions:

- `cortex-backend` (FastAPI web service)
- `cortex-frontend` (Next.js web service)
- `cortex-postgres` (managed PostgreSQL)
- `cortex-data` persistent disk mounted for backend state (`/var/data/cortex`)

## 1. Deploy via Blueprint

1. Push this repository to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select your repo and apply `render.yaml`.
4. Render will create all three resources automatically.

## 2. Required Secrets

Set these in Render for the backend service:

- `GOOGLE_API_KEY` (required for Gemini LLM/voice paths)

Optional but recommended:

- `CORTEX_API_KEY` (adds bearer-token API protection)

## 3. Runtime Defaults Used

Backend defaults in blueprint:

- `SKIP_LOCAL_MODEL=true` (Gemini-first cloud mode)
- `HOST=0.0.0.0`
- `CORTEX_DATA_DIR=/var/data/cortex` (persistent disk)
- `CORS_ALLOW_ORIGINS=https://cortex-frontend.onrender.com`
- `CORS_ALLOW_ORIGIN_REGEX=^https://.*\.onrender\.com$|^https://.*\.trycloudflare\.com$|^https?://(localhost|127\.0\.0\.1)(:\d+)?$`
- `DATABASE_URL` injected from `cortex-postgres`

Frontend defaults in blueprint:

- `NEXT_PUBLIC_API_BASE_URL=https://cortex-backend.onrender.com/api`
- `NODE_VERSION=20.18.0`

## 4. Important Database Note

`cortex-postgres` is provisioned and exposed as `DATABASE_URL`, but the current app metadata engine is still DuckDB/FAISS based and writes to `CORTEX_DATA_DIR`.

That means:

- Your production data persistence works immediately through the Render disk.
- PostgreSQL is ready for future migrations/integrations.
- No runtime breakage occurs if Postgres is present but unused.

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
   - Add memory, redeploy backend, verify memory still exists.

## 6. If You Rename Services

If you change service names in Render, update:

- backend `CORS_ALLOW_ORIGINS`
- frontend `NEXT_PUBLIC_API_BASE_URL`
- any docs/scripts using default `cortex-backend` or `cortex-frontend` hostnames
