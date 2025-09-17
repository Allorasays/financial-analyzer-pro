# PowerShell script to set up environment variables for Android development
# Run as Administrator for best results

Write-Host "🔧 Setting up Environment Variables for Financial Analyzer Pro..." -ForegroundColor Green
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "⚠️  Warning: Not running as Administrator" -ForegroundColor Yellow
    Write-Host "Some environment variables might not be set system-wide" -ForegroundColor Yellow
    Write-Host ""
}

# Function to set environment variable
function Set-EnvVar {
    param(
        [string]$Name,
        [string]$Value,
        [string]$Scope = "User"
    )
    
    try {
        [Environment]::SetEnvironmentVariable($Name, $Value, $Scope)
        Write-Host "✅ Set $Name = $Value" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to set $Name" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Function to find Java installation
function Find-JavaHome {
    $javaPaths = @(
        "C:\Program Files\Java\jdk-*",
        "C:\Program Files\Eclipse Adoptium\jdk-*",
        "C:\Program Files\OpenJDK\jdk-*"
    )
    
    foreach ($path in $javaPaths) {
        $dirs = Get-ChildItem -Path $path -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        if ($dirs) {
            return $dirs[0].FullName
        }
    }
    
    return $null
}

# Function to find Android SDK
function Find-AndroidHome {
    $androidPaths = @(
        "$env:USERPROFILE\AppData\Local\Android\Sdk",
        "C:\Android\Sdk",
        "C:\Users\$env:USERNAME\AppData\Local\Android\Sdk"
    )
    
    foreach ($path in $androidPaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    
    return $null
}

Write-Host "🔍 Detecting Java installation..." -ForegroundColor Cyan
$javaHome = Find-JavaHome

if ($javaHome) {
    Write-Host "✅ Found Java at: $javaHome" -ForegroundColor Green
    Set-EnvVar "JAVA_HOME" $javaHome
} else {
    Write-Host "❌ Java not found" -ForegroundColor Red
    Write-Host "Please install JDK from: https://adoptium.net/" -ForegroundColor Yellow
    Write-Host "After installation, run this script again" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🔍 Detecting Android SDK..." -ForegroundColor Cyan
$androidHome = Find-AndroidHome

if ($androidHome) {
    Write-Host "✅ Found Android SDK at: $androidHome" -ForegroundColor Green
    Set-EnvVar "ANDROID_HOME" $androidHome
} else {
    Write-Host "❌ Android SDK not found" -ForegroundColor Red
    Write-Host "Please install Android Studio from: https://developer.android.com/studio" -ForegroundColor Yellow
    Write-Host "After installation, run this script again" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🔧 Setting up PATH variables..." -ForegroundColor Cyan

# Add Java to PATH
if ($javaHome) {
    $javaBin = Join-Path $javaHome "bin"
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$javaBin*") {
        $newPath = "$currentPath;$javaBin"
        Set-EnvVar "PATH" $newPath
    }
}

# Add Android SDK tools to PATH
if ($androidHome) {
    $androidTools = Join-Path $androidHome "platform-tools"
    $androidBuildTools = Join-Path $androidHome "build-tools"
    
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    
    if ($currentPath -notlike "*$androidTools*") {
        $newPath = "$currentPath;$androidTools"
        Set-EnvVar "PATH" $newPath
    }
    
    if ($currentPath -notlike "*$androidBuildTools*") {
        $newPath = "$currentPath;$androidBuildTools"
        Set-EnvVar "PATH" $newPath
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "📊 SUMMARY" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if ($javaHome) {
    Write-Host "✅ JAVA_HOME set to: $javaHome" -ForegroundColor Green
} else {
    Write-Host "❌ JAVA_HOME not set" -ForegroundColor Red
}

if ($androidHome) {
    Write-Host "✅ ANDROID_HOME set to: $androidHome" -ForegroundColor Green
} else {
    Write-Host "❌ ANDROID_HOME not set" -ForegroundColor Red
}

Write-Host ""
Write-Host "🔄 Please restart your command prompt for changes to take effect" -ForegroundColor Yellow
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Restart command prompt" -ForegroundColor White
Write-Host "2. Run: check_prerequisites.bat" -ForegroundColor White
Write-Host "3. Run: setup_android.bat" -ForegroundColor White
Write-Host "4. Run: run_android.bat" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to continue"










