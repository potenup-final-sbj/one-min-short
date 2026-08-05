param(
    [Parameter(Mandatory = $true)][string]$TextPath,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [string]$VoiceName = "Microsoft Heami Desktop",
    [int]$Rate = 0
)

$ErrorActionPreference = "Stop"
$voice = New-Object -ComObject SAPI.SpVoice
$stream = New-Object -ComObject SAPI.SpFileStream

try {
    $matchingVoice = $null
    $voices = $voice.GetVoices()
    for ($index = 0; $index -lt $voices.Count; $index++) {
        $candidate = $voices.Item($index)
        if ($candidate.GetDescription() -like "$VoiceName*") {
            $matchingVoice = $candidate
            break
        }
    }

    if ($null -ne $matchingVoice) {
        $voice.Voice = $matchingVoice
    }

    $voice.Rate = $Rate
    $text = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $TextPath))
    $stream.Open($OutputPath, 3, $false)
    $voice.AudioOutputStream = $stream
    [void]$voice.Speak($text)
}
finally {
    $stream.Close()
    [Runtime.InteropServices.Marshal]::ReleaseComObject($stream) | Out-Null
    [Runtime.InteropServices.Marshal]::ReleaseComObject($voice) | Out-Null
}

