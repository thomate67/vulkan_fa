param(
    [string]$Folder = "",
    [string]$Compiler = "glslangvalidator",
    [string]$Flags = ""
)
$startoutput = "Compiling shaders using " + $Compiler
Write-Output $startoutput

$types = @("*.vert", "*.frag", "*.tesc", "*.tese", "*.geom", "*.comp")
$files = Get-Childitem $Folder -Include $types -Recurse -File


foreach($file in $files)
{
    if($Compiler -eq "glslangvalidator")
    {
        $command = "C:\VulkanSDK\1.1.101.0\Bin\glslangvalidator.exe -V $file -o $file.spv"
        Invoke-Expression $command
    }
    else
    {
        Write-Output "Please choose glslc or glslangvalidator as compiler"
        break
    }
}

$date = Get-Date -Format g
$finishOutput = "Done: " + $date
Write-Output $finishOutput 

pause