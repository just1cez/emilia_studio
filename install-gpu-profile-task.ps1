param(
    [string]$Python,
    [switch]$Remove,
    [switch]$Audit,
    [switch]$AllowUserWritable
)

# Elevated install: .\install-gpu-profile-task.ps1 [-Python <source-python.exe>]
# Inspect the registered task (no Python/config dependency): .\install-gpu-profile-task.ps1 -Audit
# Elevated uninstall: .\install-gpu-profile-task.ps1 -Remove
# The checkout and source environment remain writable. Only a new, versioned ProgramData
# runtime receives a replacement DACL; previous versions remain intact until uninstall so an
# already-running task is never stripped of its interpreter. -AllowUserWritable explicitly
# opts out of staging and is NOT a secure SYSTEM configuration (Audit reports it as unsafe).
$ErrorActionPreference = 'Stop'
$TaskName = 'EmiliaGPUProfile'
$SourceScript = Join-Path $PSScriptRoot 'gpu_profile.py'
$SafeRoot = [IO.Path]::GetFullPath((Join-Path $env:ProgramData 'EmiliaGPUProfile'))
$TrustedSids = @(
    'S-1-5-32-544', 'S-1-5-18',
    'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464'
)
$WriteMask = 0x2 -bor 0x4 -bor 0x40 -bor 0x10000 -bor 0x40000 -bor 0x80000 -bor 0x10000000 -bor 0x40000000
# Creating a sibling in ProgramData is permitted; replacing an existing protected child is not.
$AncestorWriteMask = 0x40 -bor 0x10000 -bor 0x40000 -bor 0x80000 -bor 0x10000000 -bor 0x40000000

function Get-ProfileTask {
    Get-ScheduledTask -TaskPath '\' | Where-Object { $_.TaskName -eq $TaskName }
}

function Assert-LocalPath {
    param([string]$Path)
    if (-not [IO.Path]::IsPathRooted($Path) -or $Path -notmatch '^[A-Za-z]:\\' -or
        $Path.Substring(2).Contains(':') -or $Path -match '["\r\n]') {
        throw "Expected an absolute local filesystem path: $Path"
    }
    $Full = [IO.Path]::GetFullPath($Path)
    $Part = $Full
    while ($Part) {
        if (Test-Path -LiteralPath $Part) {
            $Item = Get-Item -LiteralPath $Part -Force
            if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse points are not permitted: $Part"
            }
        } else {
            # Get-Item also detects dangling links, which Test-Path can report as absent.
            $Item = Get-Item -LiteralPath $Part -Force -ErrorAction SilentlyContinue
            if ($Item) { throw "Unusable or reparse path: $Part" }
        }
        $Parent = Split-Path -Parent $Part
        if ($Parent -eq $Part) { break }
        $Part = $Parent
    }
    return $Full
}

function Assert-ProtectedItem {
    param([string]$Path, [int]$Mask = $WriteMask)
    $Item = Get-Item -LiteralPath $Path -Force
    $Acl = $Item.GetAccessControl([Security.AccessControl.AccessControlSections]'Access,Owner')
    if ($TrustedSids -notcontains $Acl.GetOwner([Security.Principal.SecurityIdentifier]).Value) {
        throw "Untrusted owner on precreated path: $Path"
    }
    foreach ($Rule in $Acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
        if ($Rule.AccessControlType -eq 'Allow' -and
            -not ($Rule.PropagationFlags -band [Security.AccessControl.PropagationFlags]::InheritOnly) -and
            ([int]$Rule.FileSystemRights -band $Mask) -and
            $TrustedSids -notcontains $Rule.IdentityReference.Value) {
            throw "Unsafe write access on $Path for $($Rule.IdentityReference.Value): $($Rule.FileSystemRights)"
        }
    }
}

function Assert-ProtectedAncestors {
    param([string]$Path)
    $Part = Split-Path -Parent $Path
    while ($Part) {
        Assert-ProtectedItem $Part $AncestorWriteMask
        $Parent = Split-Path -Parent $Part
        if ($Parent -eq $Part) { break }
        $Part = $Parent
    }
}

function Get-PlainTree {
    param([string]$Path, [switch]$Protected)
    # Deliberately do not use -Recurse: reject a junction before traversing its children.
    $Pending = New-Object 'Collections.Generic.Stack[string]'
    $Pending.Push((Assert-LocalPath $Path))
    while ($Pending.Count) {
        $Item = Get-Item -LiteralPath ($Pending.Pop()) -Force
        if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Reparse points are not permitted: $($Item.FullName)"
        }
        if ($Protected) { Assert-ProtectedItem $Item.FullName }
        $Item
        if ($Item.PSIsContainer) {
            foreach ($Child in Get-ChildItem -LiteralPath $Item.FullName -Force) {
                if ($Child.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                    throw "Reparse points are not permitted: $($Child.FullName)"
                }
                $Pending.Push($Child.FullName)
            }
        }
    }
}

function New-ProfileAcl {
    param([bool]$Directory)
    # A fresh security descriptor, not /grant:r: all stale explicit and inherited ACEs disappear.
    $Acl = if ($Directory) { New-Object Security.AccessControl.DirectorySecurity } else { New-Object Security.AccessControl.FileSecurity }
    $Acl.SetAccessRuleProtection($true, $false)
    $Acl.SetOwner([Security.Principal.SecurityIdentifier]::new('S-1-5-32-544'))
    $Inheritance = if ($Directory) { [Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit' } else { [Security.AccessControl.InheritanceFlags]::None }
    foreach ($Sid in @('S-1-5-32-544', 'S-1-5-18', 'S-1-5-32-545')) {
        $Rights = if ($Sid -eq 'S-1-5-32-545') { 'ReadAndExecute' } else { 'FullControl' }
        $Acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
            [Security.Principal.SecurityIdentifier]::new($Sid),
            [Security.AccessControl.FileSystemRights]$Rights, $Inheritance,
            [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow))
    }
    return $Acl
}

function Set-ProfileTreeAcl {
    param([string]$Path)
    foreach ($Item in Get-PlainTree $Path) {
        $Item.SetAccessControl((New-ProfileAcl ([bool]$Item.PSIsContainer)))
    }
}

function Get-SecureTaskRuntime {
    param($Task)
    if (-not $Task) { throw "Task $TaskName is not registered." }
    $Xml = [xml](Export-ScheduledTask -TaskName $TaskName -TaskPath '\')
    $Actions = @($Xml.Task.Actions.ChildNodes | Where-Object { $_.NodeType -eq 'Element' })
    if ($Actions.Count -ne 1 -or $Actions[0].LocalName -ne 'Exec' -or
        $Xml.Task.Principals.Principal.UserId -notin @('S-1-5-18', 'SYSTEM')) {
        throw 'Expected exactly one SYSTEM Exec action.'
    }
    # Query the actual registered task DACL, not the XML registration-time descriptor.
    $Scheduler = New-Object -ComObject 'Schedule.Service'
    $Scheduler.Connect()
    $Registered = $Scheduler.GetFolder('\').GetTask($TaskName)
    $Descriptor = [Security.AccessControl.RawSecurityDescriptor]::new($Registered.GetSecurityDescriptor(7))
    if (-not $Descriptor.Owner -or $TrustedSids -notcontains $Descriptor.Owner.Value -or -not $Descriptor.DiscretionaryAcl) {
        throw 'The registered task has an untrusted owner or unrestricted DACL.'
    }
    foreach ($Ace in $Descriptor.DiscretionaryAcl) {
        if ($Ace -isnot [Security.AccessControl.CommonAce]) { throw 'Unexpected task ACL entry.' }
        if ($Ace.AceQualifier -eq [Security.AccessControl.AceQualifier]::AccessAllowed -and
            ($Ace.AccessMask -band $WriteMask) -and $TrustedSids -notcontains $Ace.SecurityIdentifier.Value) {
            throw "An untrusted principal can modify the registered task: $($Ace.SecurityIdentifier.Value)"
        }
    }
    $Exec = $Actions[0]
    $Executable = Assert-LocalPath ([string]$Exec.Command)
    if ([string]$Exec.Arguments -notmatch '^-I -S "([^"\r\n]+)" apply$') {
        throw 'The registered action must use -I -S and an explicit staged script.'
    }
    $Script = Assert-LocalPath $Matches[1]
    $Working = Assert-LocalPath ([string]$Exec.WorkingDirectory)
    $Version = Split-Path -Parent $Script
    if (-not $Version.StartsWith($SafeRoot + '\versions\', [StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path -Parent $Version) -ne (Join-Path $SafeRoot 'versions') -or
        $Working -ne $Version -or $Executable -ne (Join-Path $Version 'runtime\python.exe') -or
        (Split-Path -Leaf $Script) -ne 'gpu_profile.py') {
        throw 'The task does not use a protected, versioned ProgramData runtime and working directory.'
    }
    Assert-ProtectedAncestors $SafeRoot
    Assert-ProtectedItem $SafeRoot
    Assert-ProtectedItem (Join-Path $SafeRoot 'versions')
    $null = @(Get-PlainTree $Version -Protected)
    foreach ($Required in @($Executable, $Script, (Join-Path $Version 'runtime\Lib\site-packages\pynvml.py'))) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "Missing task dependency: $Required" }
    }
    $Runtime = Split-Path -Parent $Executable
    $PythonDll = @(Get-ChildItem -LiteralPath $Runtime -File | Where-Object { $_.Name -match '^python3\d+\.dll$' })
    if ($PythonDll.Count -ne 1) { throw 'Expected exactly one versioned Python DLL.' }
    $PathFile = Join-Path $Runtime ($PythonDll[0].BaseName + '._pth')
    $Expected = ".`nDLLs`nLib`nLib\site-packages"
    if (-not (Test-Path -LiteralPath $PathFile -PathType Leaf) -or
        (Get-Content -LiteralPath $PathFile -Raw).TrimEnd("`r", "`n").Replace("`r`n", "`n") -cne $Expected) {
        throw 'The runtime must use only its fixed local ._pth paths, without site initialization.'
    }
    [pscustomobject]@{ Python = $Executable; Script = $Script; WorkingDirectory = $Working }
}

function Invoke-ProfilePython {
    param([string]$Executable, [string]$Script, [string]$Command, [string]$Working, [bool]$Isolated = $true)
    Push-Location -LiteralPath $Working
    try {
        if ($Isolated) { & $Executable -I -S $Script $Command } else { & $Executable -I $Script $Command }
        if ($LASTEXITCODE -ne 0) { throw "Python $Command failed with exit code $LASTEXITCODE." }
    } finally { Pop-Location }
}

if ($Remove -and $Audit) { throw '-Remove and -Audit cannot be combined.' }
if ($Audit) {
    try {
        $Runtime = Get-SecureTaskRuntime (Get-ProfileTask)
        Write-Host "Protected SYSTEM task: $($Runtime.Python) -I -S `"$($Runtime.Script)`" apply"
        Write-Host "Protected working directory: $($Runtime.WorkingDirectory)"
        exit 0
    } catch {
        Write-Warning "Task audit failed: $($_.Exception.Message)"
        exit 1
    }
}
$Identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $Identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script from an elevated (Administrator) PowerShell; it never self-elevates.'
}

if ($Remove) {
    # Unregister first, without resolving Python, importing checkout helpers or requiring pynvml.
    $Task = Get-ProfileTask
    $ResetRuntime = $null
    if ($Task) {
        try { $ResetRuntime = Get-SecureTaskRuntime $Task } catch { Write-Warning "Skipping optional GPU reset: $($_.Exception.Message)" }
        Stop-ScheduledTask -TaskName $TaskName -TaskPath '\'
        $Deadline = [DateTime]::UtcNow.AddSeconds(20)
        while ((Get-ProfileTask).State -eq 'Running') {
            if ([DateTime]::UtcNow -ge $Deadline) { throw 'Task did not stop; runtime has not been removed.' }
            Start-Sleep -Milliseconds 250
        }
        Unregister-ScheduledTask -TaskName $TaskName -TaskPath '\' -Confirm:$false
    }
    if ($ResetRuntime) {
        try { Invoke-ProfilePython $ResetRuntime.Python $ResetRuntime.Script 'reset' $ResetRuntime.WorkingDirectory }
        catch { Write-Warning "Task removed; optional GPU reset failed: $($_.Exception.Message)" }
    }
    $null = Assert-LocalPath $SafeRoot
    if (Test-Path -LiteralPath $SafeRoot) {
        Assert-ProtectedAncestors $SafeRoot
        $null = @(Get-PlainTree $SafeRoot -Protected)
        Remove-Item -LiteralPath $SafeRoot -Recurse -Force
    }
    Write-Host "Task $TaskName removed. Missing dependencies do not prevent removal; GPU reset is best effort."
    exit 0
}

# Resolve install-only dependencies only after audit/removal have finished.
if (-not $Python) {
    . (Join-Path $PSScriptRoot 'scripts\EmiliaPython.ps1')
    $Python = Resolve-EmiliaPython -Root $PSScriptRoot
}
$Python = Assert-LocalPath $Python
$Script = Assert-LocalPath $SourceScript
foreach ($File in @($Python, $Script)) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Required source file was not found: $File" }
}
$Working = Split-Path -Parent $Script
if (-not $AllowUserWritable) {
    $null = Assert-LocalPath $SafeRoot
    Assert-ProtectedAncestors $SafeRoot
    if (Test-Path -LiteralPath $SafeRoot) {
        if (-not (Get-Item -LiteralPath $SafeRoot -Force).PSIsContainer) { throw "Not a directory: $SafeRoot" }
        # Do not take ownership of or traverse attacker-controlled precreated content.
        $null = @(Get-PlainTree $SafeRoot -Protected)
        Set-ProfileTreeAcl $SafeRoot
    } else {
        # Atomic protected creation: no interval with ProgramData's inherited user-create ACEs.
        [IO.Directory]::CreateDirectory($SafeRoot, (New-ProfileAcl $true)) | Out-Null
        Assert-ProtectedItem $SafeRoot
    }
    $Versions = Join-Path $SafeRoot 'versions'
    New-Item -ItemType Directory -Path $Versions -Force | Out-Null
    $Working = Join-Path $Versions ([Guid]::NewGuid().ToString('N'))
    $RuntimeDest = Join-Path $Working 'runtime'
    New-Item -ItemType Directory -Path (Join-Path $RuntimeDest 'DLLs') -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $RuntimeDest 'Lib\site-packages') -Force | Out-Null
    $SourceEnv = Split-Path -Parent $Python
    $CopyItems = @(Get-ChildItem -LiteralPath $SourceEnv -File | Where-Object { $_.Extension -eq '.dll' })
    $CopyItems += Get-Item -LiteralPath $Python
    foreach ($Item in $CopyItems) {
        $null = Assert-LocalPath $Item.FullName
        $Name = if ($Item.FullName -eq $Python) { 'python.exe' } else { $Item.Name }
        Copy-Item -LiteralPath $Item.FullName -Destination (Join-Path $RuntimeDest $Name)
    }
    foreach ($Directory in @('DLLs', 'Lib')) {
        $SourceDir = Assert-LocalPath (Join-Path $SourceEnv $Directory)
        foreach ($Item in Get-ChildItem -LiteralPath $SourceDir -Force) {
            if ($Directory -eq 'Lib' -and $Item.Name -in @('site-packages', '__pycache__', 'test')) { continue }
            $null = @(Get-PlainTree $Item.FullName)
            Copy-Item -LiteralPath $Item.FullName -Destination (Join-Path $RuntimeDest $Directory) -Recurse
        }
    }
    # pynvml is a single-module dependency; never copy arbitrary site-packages or .pth hooks.
    $NvmlSource = Assert-LocalPath (Join-Path $SourceEnv 'Lib\site-packages\pynvml.py')
    Copy-Item -LiteralPath $NvmlSource -Destination (Join-Path $RuntimeDest 'Lib\site-packages\pynvml.py')
    $PythonDll = @(Get-ChildItem -LiteralPath $RuntimeDest -File | Where-Object { $_.Name -match '^python3\d+\.dll$' })
    if ($PythonDll.Count -ne 1) { throw 'Source environment must contain exactly one versioned Python DLL.' }
    Set-Content -LiteralPath (Join-Path $RuntimeDest ($PythonDll[0].BaseName + '._pth')) -Encoding ASCII -Value @('.', 'DLLs', 'Lib', 'Lib\site-packages')
    $Script = Join-Path $Working 'gpu_profile.py'
    Copy-Item -LiteralPath $SourceScript -Destination $Script
    Set-ProfileTreeAcl $Working
    $Python = Join-Path $RuntimeDest 'python.exe'
} else {
    Write-Warning 'UNSAFE opt-out: SYSTEM will run your mutable interpreter and script. Audit will fail this configuration.'
}

$Escape = [Security.SecurityElement]
$ResumeQuery = $Escape::Escape("<QueryList><Query Id=`"0`" Path=`"System`"><Select Path=`"System`">*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]</Select></Query></QueryList>")
$Flags = if ($AllowUserWritable) { '-I' } else { '-I -S' }
$Xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Restore CMP 40HX power limit, clock lock, fan speed and P0 clock offsets.</Description>
    <SecurityDescriptor>O:BAG:BAD:P(A;;FA;;;BA)(A;;FA;;;SY)(A;;GRGX;;;AU)</SecurityDescriptor>
  </RegistrationInfo>
  <Triggers>
    <BootTrigger>
      <Enabled>true</Enabled><Delay>PT30S</Delay>
      <Repetition><Interval>PT5M</Interval><StopAtDurationEnd>false</StopAtDurationEnd></Repetition>
    </BootTrigger>
    <LogonTrigger><Enabled>true</Enabled></LogonTrigger>
    <EventTrigger><Enabled>true</Enabled><Delay>PT10S</Delay><Subscription>$ResumeQuery</Subscription></EventTrigger>
  </Triggers>
  <Principals><Principal id="System"><UserId>S-1-5-18</UserId><RunLevel>HighestAvailable</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable><ExecutionTimeLimit>PT2M</ExecutionTimeLimit><Priority>7</Priority>
  </Settings>
  <Actions Context="System"><Exec>
    <Command>$($Escape::Escape($Python))</Command>
    <Arguments>$Flags "$($Escape::Escape($Script))" apply</Arguments>
    <WorkingDirectory>$($Escape::Escape($Working))</WorkingDirectory>
  </Exec></Actions>
</Task>
"@
Register-ScheduledTask -TaskName $TaskName -TaskPath '\' -Xml $Xml -Force | Out-Null
if (-not $AllowUserWritable) { $null = Get-SecureTaskRuntime (Get-ProfileTask) }
Start-ScheduledTask -TaskName $TaskName -TaskPath '\'
Start-Sleep -Seconds 5
$Info = Get-ScheduledTaskInfo -TaskName $TaskName -TaskPath '\'
if ($Info.LastTaskResult -ne 0 -and $Info.LastTaskResult -ne 0x41301) {
    throw "The registered task failed with result $($Info.LastTaskResult)."
}
Write-Host "Task $TaskName registered; last result: $($Info.LastTaskResult). Previous runtime versions were retained for active instances."
Invoke-ProfilePython $Python $Script 'status' $Working (-not $AllowUserWritable)
exit 0
