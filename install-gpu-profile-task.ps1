param(
    [string]$Python,
    [switch]$Remove,
    [switch]$Audit,
    [switch]$AllowUserWritable
)

# Registers the EmiliaGPUProfile task: runs `gpu_profile.py apply` as SYSTEM at boot (repeating every
# 5 minutes to recover from driver resets), at logon and after sleep/hibernate resume.
#
# SECURITY: the task executes $Python and gpu_profile.py as SYSTEM. If a non-administrator can modify
# either file or its directory, that account can run arbitrary code as SYSTEM. Registration therefore
# refuses such paths unless -AllowUserWritable is given. `-Audit` prints the finding without elevation.
$ErrorActionPreference = 'Stop'
$TaskName = 'EmiliaGPUProfile'
$Script = Join-Path $PSScriptRoot 'gpu_profile.py'
. (Join-Path $PSScriptRoot 'scripts\EmiliaPython.ps1')
if (-not $Python) {
    $Python = Resolve-EmiliaPython -Root $PSScriptRoot
}

# Administrators, SYSTEM, TrustedInstaller: principals that already hold SYSTEM-equivalent rights.
$TrustedSids = @(
    'S-1-5-32-544',
    'S-1-5-18',
    'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464'
)
# WriteData, AppendData, DeleteSubdirectoriesAndFiles, Delete, ChangePermissions, TakeOwnership,
# GENERIC_ALL, GENERIC_WRITE. Read/execute bits are deliberately excluded.
$WriteMask = 0x2 -bor 0x4 -bor 0x40 -bor 0x10000 -bor 0x40000 -bor 0x80000 -bor 0x10000000 -bor 0x40000000

function Get-UserWritableAccess {
    param([string[]]$Paths)
    foreach ($Path in $Paths) {
        # .NET directly: independent of the Microsoft.PowerShell.Security module being loadable.
        $Acl = (Get-Item -LiteralPath $Path -Force).GetAccessControl('Access')
        foreach ($Rule in $Acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
            if ($Rule.AccessControlType -ne 'Allow') { continue }
            # Inherit-only entries apply to children, not to the object being checked.
            if ($Rule.PropagationFlags -band [Security.AccessControl.PropagationFlags]::InheritOnly) { continue }
            if (-not ([int]$Rule.FileSystemRights -band $WriteMask)) { continue }
            if ($TrustedSids -contains $Rule.IdentityReference.Value) { continue }
            try {
                $Name = $Rule.IdentityReference.Translate([Security.Principal.NTAccount]).Value
            } catch {
                $Name = $Rule.IdentityReference.Value
            }
            [pscustomobject]@{ Path = $Path; Identity = $Name; Rights = $Rule.FileSystemRights }
        }
    }
}
foreach ($Path in @($Python, $Script)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file was not found: $Path"
    }
}
$Exposed = @(Get-UserWritableAccess -Paths @(
    $Script, (Split-Path -Parent $Script), $Python, (Split-Path -Parent $Python)
))
if ($Exposed.Count -gt 0) {
    Write-Warning 'Non-administrators can modify code that the task runs as SYSTEM (local privilege escalation):'
    $Exposed | Format-Table -AutoSize | Out-String | Write-Warning
    Write-Warning 'Harden with, for each listed directory:  icacls "<dir>" /inheritance:d  then  icacls "<dir>" /remove:g "Authenticated Users" /grant:r "Users:(OI)(CI)RX"'
}
if ($Audit) {
    if ($Exposed.Count -eq 0) { Write-Host 'No user-writable path is executed by the task.' }
    exit ([int]($Exposed.Count -gt 0))
}

$Identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $Identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script from an elevated (Administrator) PowerShell.'
}

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    & $Python $Script reset
    exit $LASTEXITCODE
}

if ($Exposed.Count -gt 0 -and -not $AllowUserWritable) {
    throw 'Refusing to register a SYSTEM task over user-writable code. Harden the listed paths, or pass -AllowUserWritable to accept the risk.'
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
