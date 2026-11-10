param([switch]$Remove)

# Registers the EmiliaGPUProfile task: runs `gpu_profile.py apply` as SYSTEM at boot (repeating every
# 5 minutes to recover from driver resets), at logon and after sleep/hibernate resume.
$ErrorActionPreference = 'Stop'
$TaskName = 'EmiliaGPUProfile'
$Python = 'D:\conda-data\envs\comfyui\python.exe'
$Script = Join-Path $PSScriptRoot 'gpu_profile.py'

$Identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $Identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script from an elevated (Administrator) PowerShell.'
}

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    & $Python $Script reset
    exit $LASTEXITCODE
}

foreach ($Path in @($Python, $Script)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file was not found: $Path"
    }
}

$Escape = [Security.SecurityElement]
$ResumeQuery = $Escape::Escape("<QueryList><Query Id=`"0`" Path=`"System`"><Select Path=`"System`">*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]</Select></Query></QueryList>")
# Authenticated users may read and start (not modify) the task, so `schtasks /Run` works unelevated.
$Xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Restore CMP 40HX power limit, clock lock, fan speed and P0 clock offsets.</Description>
    <SecurityDescriptor>D:(A;;FA;;;BA)(A;;FA;;;SY)(A;;GRGX;;;AU)</SecurityDescriptor>
  </RegistrationInfo>
  <Triggers>
    <BootTrigger>
      <Enabled>true</Enabled>
      <Delay>PT30S</Delay>
      <Repetition>
        <Interval>PT5M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </BootTrigger>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
    <EventTrigger>
      <Enabled>true</Enabled>
      <Delay>PT10S</Delay>
      <Subscription>$ResumeQuery</Subscription>
    </EventTrigger>
  </Triggers>
  <Principals>
    <Principal id="System">
      <UserId>S-1-5-18</UserId>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT2M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="System">
    <Exec>
      <Command>$($Escape::Escape($Python))</Command>
      <Arguments>"$($Escape::Escape($Script))" apply</Arguments>
    </Exec>
  </Actions>
</Task>
"@

Register-ScheduledTask -TaskName $TaskName -Xml $Xml -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 5
$Info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Host "Task $TaskName registered; last result: $($Info.LastTaskResult)"
& $Python $Script status
exit $LASTEXITCODE
