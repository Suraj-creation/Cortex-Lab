$ErrorActionPreference = 'Continue'

$base = 'http://localhost:8000/api'
$results = @()
$ingestedId = $null
$traceId = $null

function Add-Result {
  param(
    [string]$Name,
    [string]$Method,
    [string]$Path,
    [int]$Status,
    [bool]$Pass,
    [string]$Note
  )

  $script:results += [pscustomobject]@{
    name = $Name
    method = $Method
    path = $Path
    status = $Status
    pass = $Pass
    note = $Note
  }
}

function Invoke-Test {
  param(
    [string]$Name,
    [string]$Method,
    [string]$Path,
    [object]$Body = $null,
    [int[]]$Expected = @(200)
  )

  try {
    $uri = "$base$Path"
    if ($null -ne $Body) {
      $json = $Body | ConvertTo-Json -Depth 10 -Compress
      $resp = Invoke-WebRequest -Uri $uri -Method $Method -ContentType 'application/json' -Body $json -UseBasicParsing -TimeoutSec 60
    }
    else {
      $resp = Invoke-WebRequest -Uri $uri -Method $Method -UseBasicParsing -TimeoutSec 60
    }

    $status = [int]$resp.StatusCode
    $ok = $Expected -contains $status
    Add-Result -Name $Name -Method $Method -Path $Path -Status $status -Pass $ok -Note ($resp.Content | Out-String)
    return $resp
  }
  catch {
    $status = 0
    if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
      $status = [int]$_.Exception.Response.StatusCode
    }
    $ok = $Expected -contains $status
    Add-Result -Name $Name -Method $Method -Path $Path -Status $status -Pass $ok -Note $_.Exception.Message
    return $null
  }
}

# Core health/provider
$null = Invoke-Test 'health' 'GET' '/health' $null @(200)
$null = Invoke-Test 'llm_provider_get' 'GET' '/llm/provider' $null @(200)
$null = Invoke-Test 'llm_provider_set_gemini' 'POST' '/llm/provider' @{ provider = 'gemini' } @(200)

# Memory
$null = Invoke-Test 'memories_list' 'GET' '/memories?limit=5&offset=0' $null @(200)
$null = Invoke-Test 'memories_search' 'POST' '/memories/search' @{ query = 'integration smoke test'; top_k = 3 } @(200)
$r = Invoke-Test 'memories_ingest' 'POST' '/memories/ingest' @{ content = "integration smoke $(Get-Date -Format o)"; source = 'integration-smoke' } @(200)
if ($r -and $r.Content) {
  try {
    $obj = $r.Content | ConvertFrom-Json
    $ingestedId = $obj.memory.id
  }
  catch {}
}
if ($ingestedId) {
  $null = Invoke-Test 'memories_delete_ingested' 'DELETE' "/memories/$ingestedId" $null @(200)
}

# Graph + RAG
$null = Invoke-Test 'graph' 'GET' '/graph' $null @(200)
$null = Invoke-Test 'rag_stats' 'GET' '/rag/stats' $null @(200)
$null = Invoke-Test 'rag_health' 'GET' '/rag/health' $null @(200)
$r = Invoke-Test 'rag_traces' 'GET' '/rag/traces?limit=5' $null @(200)
if ($r -and $r.Content) {
  try {
    $obj = $r.Content | ConvertFrom-Json
    if ($obj.traces.Count -gt 0) {
      $traceId = $obj.traces[0].trace_id
    }
  }
  catch {}
}
if ($traceId) {
  $null = Invoke-Test 'rag_trace_by_id' 'GET' "/rag/traces/$traceId" $null @(200)
}
$null = Invoke-Test 'observability_metrics' 'GET' '/rag/observability/metrics' $null @(200)

# Ambient + TTS
$null = Invoke-Test 'ambient_status' 'GET' '/ambient/status' $null @(200)
$null = Invoke-Test 'ambient_config' 'GET' '/ambient/config' $null @(200)
$null = Invoke-Test 'ambient_voice_providers' 'GET' '/ambient/voice-providers' $null @(200)
$null = Invoke-Test 'ambient_enrollment_status' 'GET' '/ambient/enrollment-status' $null @(200)
$null = Invoke-Test 'ambient_live_transcript' 'GET' '/ambient/live-transcript' $null @(200)
$null = Invoke-Test 'ambient_conversations' 'GET' '/ambient/conversations?limit=2&offset=0' $null @(200)
$null = Invoke-Test 'tts_status' 'GET' '/tts/status' $null @(200)

# Documents
$null = Invoke-Test 'documents_list' 'GET' '/documents' $null @(200)
$null = Invoke-Test 'documents_usage' 'GET' '/documents/usage' $null @(200)
$null = Invoke-Test 'documents_query' 'POST' '/documents/query' @{ query = 'What is Cortex Lab?'; top_k = 2 } @(200)

# Chat
$chatBody = @{
  messages = @(@{ role = 'user'; content = 'Reply with one word: OK' })
  temperature = 0.2
  top_p = 0.9
  max_tokens = 64
  stream = $false
  use_rag = $false
  llm_provider = 'gemini'
}
$null = Invoke-Test 'chat_non_rag' 'POST' '/chat' $chatBody @(200)

$ragBody = @{
  messages = @(@{ role = 'user'; content = 'What is Cortex Lab in one sentence?' })
  temperature = 0.2
  top_p = 0.9
  max_tokens = 96
  stream = $false
  use_rag = $true
  llm_provider = 'gemini'
}
$null = Invoke-Test 'chat_rag' 'POST' '/rag/chat' $ragBody @(200)

$pass = ($results | Where-Object { $_.pass -eq $true }).Count
$fail = ($results | Where-Object { $_.pass -eq $false }).Count

$outFile = Join-Path (Split-Path -Parent $PSScriptRoot) 'Cortex-Lab\api-smoke-results.json'
$results | ConvertTo-Json -Depth 6 | Out-File -FilePath $outFile -Encoding utf8

Write-Output "API_SMOKE_SUMMARY pass=$pass fail=$fail file=Cortex-Lab/api-smoke-results.json"
$results | Sort-Object pass, name | Format-Table -AutoSize

if ($fail -gt 0) {
  exit 1
}
