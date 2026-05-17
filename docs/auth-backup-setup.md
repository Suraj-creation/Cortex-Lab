# Auth and Backup Setup

This repo now includes:

- backend Google OAuth start/callback/session endpoints
- signed app-session tokens for web and mobile
- local-first backup bundle creation
- optional Supabase Postgres backup persistence
- optional Google Drive backup upload using the signed-in user's refresh token
- Supabase schema for profiles, devices, sync cursors, memory events, backups, backup files, and realtime event fan-out

## Required environment variables

### Google OAuth

- `CORTEX_AUTH_SECRET`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI`
- `GOOGLE_SCOPES`

Recommended scopes:

```text
openid email profile https://www.googleapis.com/auth/drive.file
```

The `drive.file` scope is required for the backend to create a private `Cortex Lab Backups` folder in the user's Google Drive and upload backup bundles there. Keep the Google client secret only on the backend.

### Supabase Postgres backup target

Use either:

- `SUPABASE_DATABASE_URL`

or the split values:

- `SUPABASE_DB_HOST`
- `SUPABASE_DB_PORT`
- `SUPABASE_DB_NAME`
- `SUPABASE_DB_USER`
- `SUPABASE_DB_PASSWORD`

For Supabase Storage backup objects, also set:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_BACKUP_BUCKET`

## Google Cloud checklist

Before sign-in works end to end, configure:

1. OAuth consent screen
2. OAuth client for a web application
3. Authorized redirect URI matching `GOOGLE_REDIRECT_URI`
4. For deployed web/mobile, prefer the frontend proxy callback as `GOOGLE_REDIRECT_URI`: `https://cortex-frontend-t53t.onrender.com/api/auth/google/callback`
5. For local backend-only testing, use the localhost callback configured in Google Cloud, for example `http://localhost:8080/api/auth/google/callback`
6. If mobile deep-link sign-in is used, the backend callback should redirect to `cortexlab://auth/callback` after Google returns to the web callback

## Mobile deep link

The Expo app now declares:

```json
{
  "expo": {
    "scheme": "cortexlab"
  }
}
```

## Supabase table

Apply:

- [infra/supabase/cortex_backup_schema.sql](/C:/Users/Govin/Desktop/CortexLab/infra/supabase/cortex_backup_schema.sql)

The backend also auto-creates the minimum profile/backup tables on first write when the DSN is configured, but the SQL file is the production schema because it includes realtime events, device cursors, indexes, RLS, the private `cortex-backups` storage bucket, and backup-file metadata.

## Render environment

Add the values from [backend/.env.example](/C:/Users/Govin/Desktop/CortexLab/backend/.env.example) to the Render backend service environment. Do not commit real secrets to the repo.

If the direct Supabase host is unreachable from a deployment environment, use Supabase's Session Pooler connection string in `SUPABASE_DATABASE_URL`.

## MCP note

The project can use the Supabase MCP server when the local Claude CLI is installed and authenticated:

```bash
claude mcp add --scope project --transport http supabase "https://mcp.supabase.com/mcp?project_ref=lqqokdustkkixwgvsowo"
claude /mcp
```

This Codex environment does not currently expose an authenticated Supabase MCP tool, so the app integration is implemented through environment-based direct Postgres/Supabase configuration.

## Local model mode

The mobile app now allows in-app Gemma modelpack downloads again. Because the native LiteRT/Gemma runtime bridge is still scaffold-only, downloaded modelpacks stage `Gemma Local` in hybrid mode and requests fall back to Gemini instead of failing. Once the native runtime handler is implemented, the same modelpack install state can be used to route requests fully offline.
