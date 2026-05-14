create extension if not exists pgcrypto;

create table if not exists public.cortex_profiles (
  user_sub text primary key,
  email text not null,
  display_name text,
  avatar_url text,
  provider text not null default 'google',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.cortex_devices (
  device_id text primary key,
  user_sub text not null references public.cortex_profiles(user_sub) on delete cascade,
  platform text not null,
  app_version text,
  model text,
  last_seen_at timestamptz not null default now(),
  local_store_summary jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.cortex_sync_cursors (
  cursor_id uuid primary key default gen_random_uuid(),
  user_sub text not null references public.cortex_profiles(user_sub) on delete cascade,
  device_id text references public.cortex_devices(device_id) on delete set null,
  namespace text not null,
  high_watermark text not null default '',
  last_synced_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  unique (user_sub, device_id, namespace)
);

create table if not exists public.cortex_memory_events (
  event_id text primary key,
  user_sub text not null references public.cortex_profiles(user_sub) on delete cascade,
  device_id text references public.cortex_devices(device_id) on delete set null,
  session_id text,
  source text not null default 'mobile',
  event_type text not null,
  payload jsonb not null,
  tags text[] not null default '{}',
  created_at timestamptz not null default now(),
  ingested_at timestamptz
);

create table if not exists public.cortex_backup_snapshots (
  backup_id text primary key,
  user_sub text not null references public.cortex_profiles(user_sub) on delete cascade,
  email text not null,
  provider text not null default 'google',
  created_at timestamptz not null default now(),
  size_bytes bigint not null,
  sha256 text not null,
  manifest jsonb not null,
  bundle bytea not null,
  google_drive jsonb not null default '{}'::jsonb
);

create table if not exists public.cortex_backup_files (
  file_id uuid primary key default gen_random_uuid(),
  backup_id text not null references public.cortex_backup_snapshots(backup_id) on delete cascade,
  user_sub text not null references public.cortex_profiles(user_sub) on delete cascade,
  provider text not null,
  external_file_id text,
  folder_id text,
  web_view_link text,
  sha256 text,
  size_bytes bigint,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.cortex_realtime_events (
  realtime_id uuid primary key default gen_random_uuid(),
  user_sub text not null references public.cortex_profiles(user_sub) on delete cascade,
  device_id text references public.cortex_devices(device_id) on delete set null,
  channel text not null,
  event_type text not null,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  delivered_at timestamptz
);

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'cortex-backups',
  'cortex-backups',
  false,
  1073741824,
  array['application/zip', 'application/octet-stream']
)
on conflict (id) do update set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

create index if not exists cortex_profiles_email_idx
  on public.cortex_profiles (email);

create index if not exists cortex_devices_user_sub_idx
  on public.cortex_devices (user_sub, last_seen_at desc);

create index if not exists cortex_memory_events_user_sub_idx
  on public.cortex_memory_events (user_sub, created_at desc);

create index if not exists cortex_memory_events_tags_idx
  on public.cortex_memory_events using gin (tags);

create index if not exists cortex_backup_snapshots_user_sub_idx
  on public.cortex_backup_snapshots (user_sub, created_at desc);

create index if not exists cortex_backup_snapshots_email_idx
  on public.cortex_backup_snapshots (email, created_at desc);

create index if not exists cortex_backup_files_backup_id_idx
  on public.cortex_backup_files (backup_id);

create index if not exists cortex_realtime_events_user_sub_idx
  on public.cortex_realtime_events (user_sub, created_at desc);

alter table public.cortex_profiles enable row level security;
alter table public.cortex_devices enable row level security;
alter table public.cortex_sync_cursors enable row level security;
alter table public.cortex_memory_events enable row level security;
alter table public.cortex_backup_snapshots enable row level security;
alter table public.cortex_backup_files enable row level security;
alter table public.cortex_realtime_events enable row level security;

drop policy if exists cortex_profiles_service_role_all on public.cortex_profiles;
create policy cortex_profiles_service_role_all
  on public.cortex_profiles
  for all
  using (auth.role() = 'service_role')
  with check (auth.role() = 'service_role');

drop policy if exists cortex_devices_service_role_all on public.cortex_devices;
create policy cortex_devices_service_role_all
  on public.cortex_devices
  for all
  using (auth.role() = 'service_role')
  with check (auth.role() = 'service_role');

drop policy if exists cortex_sync_cursors_service_role_all on public.cortex_sync_cursors;
create policy cortex_sync_cursors_service_role_all
  on public.cortex_sync_cursors
  for all
  using (auth.role() = 'service_role')
  with check (auth.role() = 'service_role');

drop policy if exists cortex_memory_events_service_role_all on public.cortex_memory_events;
create policy cortex_memory_events_service_role_all
  on public.cortex_memory_events
  for all
  using (auth.role() = 'service_role')
  with check (auth.role() = 'service_role');

drop policy if exists cortex_backup_snapshots_service_role_all on public.cortex_backup_snapshots;
create policy cortex_backup_snapshots_service_role_all
  on public.cortex_backup_snapshots
  for all
  using (auth.role() = 'service_role')
  with check (auth.role() = 'service_role');

drop policy if exists cortex_backup_files_service_role_all on public.cortex_backup_files;
create policy cortex_backup_files_service_role_all
  on public.cortex_backup_files
  for all
  using (auth.role() = 'service_role')
  with check (auth.role() = 'service_role');

drop policy if exists cortex_realtime_events_service_role_all on public.cortex_realtime_events;
create policy cortex_realtime_events_service_role_all
  on public.cortex_realtime_events
  for all
  using (auth.role() = 'service_role')
  with check (auth.role() = 'service_role');

drop policy if exists cortex_backups_storage_service_role_all on storage.objects;
create policy cortex_backups_storage_service_role_all
  on storage.objects
  for all
  using (bucket_id = 'cortex-backups' and auth.role() = 'service_role')
  with check (bucket_id = 'cortex-backups' and auth.role() = 'service_role');

do $$
begin
  if not exists (
    select 1
    from pg_publication
    where pubname = 'supabase_realtime'
  ) then
    create publication supabase_realtime;
  end if;
end $$;

do $$
begin
  if not exists (
    select 1
    from pg_publication_tables
    where pubname = 'supabase_realtime'
      and schemaname = 'public'
      and tablename = 'cortex_realtime_events'
  ) then
    alter publication supabase_realtime add table public.cortex_realtime_events;
  end if;

  if not exists (
    select 1
    from pg_publication_tables
    where pubname = 'supabase_realtime'
      and schemaname = 'public'
      and tablename = 'cortex_memory_events'
  ) then
    alter publication supabase_realtime add table public.cortex_memory_events;
  end if;
end $$;
