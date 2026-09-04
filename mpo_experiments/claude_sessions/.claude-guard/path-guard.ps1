#requires -Version 5
# Path guard for Claude Code PreToolUse hook.
# Reads the hook JSON on stdin, denies any file-tool call whose target path
# falls OUTSIDE the allowed boundary folder. Allows everything inside it.
$ErrorActionPreference = 'Stop'

$boundary = 'C:\Users\User\Desktop\Docs\Academic\Tensors\code\tn_efficient_embedding\mpo_experiments\claude_sessions'

$raw = [Console]::In.ReadToEnd()
if (-not $raw) { exit 0 }                       # no input -> don't block
try { $j = $raw | ConvertFrom-Json } catch { exit 0 }  # unparsable -> don't block

$inp  = $j.tool_input
$base = if ($j.cwd) { [string]$j.cwd } else { (Get-Location).Path }

# Gather every path-like field a file tool might carry.
$cands = @()
foreach ($f in 'file_path','filePath','path','notebook_path') {
    $v = $inp.$f
    if ($v) { $cands += [string]$v }
}
if ($cands.Count -eq 0) { exit 0 }             # nothing path-like -> allow normal flow

function Resolve-Abs($p, $base) {
    if (-not [System.IO.Path]::IsPathRooted($p)) { $p = Join-Path $base $p }
    try { return [System.IO.Path]::GetFullPath($p) } catch { return $p }
}

$bn = ([System.IO.Path]::GetFullPath($boundary)).TrimEnd('\').ToLower()

foreach ($c in $cands) {
    $cn = (Resolve-Abs $c $base).TrimEnd('\').ToLower()
    $inside = ($cn -eq $bn) -or ($cn.StartsWith($bn + '\'))
    if (-not $inside) {
        $reason = "Blocked by project path-guard: '$cn' is outside the allowed folder ($boundary)."
        $out = @{ hookSpecificOutput = @{
            hookEventName            = 'PreToolUse'
            permissionDecision       = 'deny'
            permissionDecisionReason = $reason
        } } | ConvertTo-Json -Compress -Depth 6
        [Console]::Out.WriteLine($out)
        exit 0
    }
}

# All target paths are inside the boundary -> allow without prompting.
$ok = @{ hookSpecificOutput = @{
    hookEventName            = 'PreToolUse'
    permissionDecision       = 'allow'
    permissionDecisionReason = 'within allowed folder'
} } | ConvertTo-Json -Compress -Depth 6
[Console]::Out.WriteLine($ok)
exit 0
