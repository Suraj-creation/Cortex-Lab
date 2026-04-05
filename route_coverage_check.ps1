$ErrorActionPreference = 'Stop'
$clientPath = 'mobile/shared/core/api/client.ts'
$serverPath = 'backend/server.py'

$clientRaw = Get-Content -Raw $clientPath
$clientMatches = [regex]::Matches($clientRaw, '\$\{baseUrl\}(/[^`"'']+)')
$clientEndpoints = @()
foreach ($m in $clientMatches) {
  $p = $m.Groups[1].Value
  if ($p -notmatch '^/api/') {
    $p = "/api$p"
  }
  $p = $p -replace '\$\{[^\}]+\}', '{param}'
  $clientEndpoints += $p
}
$clientEndpoints = $clientEndpoints | Sort-Object -Unique

$serverLines = Get-Content $serverPath
$routeMatches = $serverLines | Select-String -Pattern '@app\.(get|post|put|delete|patch)\("([^"]+)"' -AllMatches
$serverEndpoints = @()
foreach ($r in $routeMatches) {
  foreach ($m in $r.Matches) {
    $p = $m.Groups[2].Value -replace '\{[^\}]+\}', '{param}'
    $serverEndpoints += $p
  }
}
$serverEndpoints = $serverEndpoints | Sort-Object -Unique

$missingInServer = $clientEndpoints | Where-Object { $_ -notin $serverEndpoints }

Write-Output ('CLIENT_ENDPOINT_COUNT=' + $clientEndpoints.Count)
Write-Output ('SERVER_ENDPOINT_COUNT=' + $serverEndpoints.Count)
if ($missingInServer.Count -eq 0) {
  Write-Output 'MISSING_IN_SERVER=NONE'
  exit 0
}
Write-Output 'MISSING_IN_SERVER:'
$missingInServer | ForEach-Object { Write-Output $_ }
exit 1
