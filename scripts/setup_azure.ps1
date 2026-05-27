# Idempotent Azure provisioning for the en-es-mt experiment.
# Usage:
#   pwsh -File scripts/setup_azure.ps1 `
#       -ResourceGroup rg-en-es-mt `
#       -Location eastus `
#       -Workspace aml-en-es-mt `
#       -StorageAccount enesmtek09 `
#       -ComputeName gpu-a10-1x `
#       -VmSize Standard_NV36ads_A10_v5
#
# Re-running is safe — every step uses az's "create if not exists" semantics.

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ResourceGroup,
    [Parameter(Mandatory=$true)][string]$Location,
    [Parameter(Mandatory=$true)][string]$Workspace,
    [Parameter(Mandatory=$true)][string]$StorageAccount,
    [string]$Container       = 'en-es-mt',
    [string]$ComputeName     = 'gpu-a10-1x',
    [string]$VmSize          = 'Standard_NV36ads_A10_v5',
    [int]$MaxInstances       = 1,
    [int]$IdleScaleDownSec   = 1800,
    [string]$EnvironmentName = 'en-es-mt-env',
    [switch]$SkipCompute,
    [switch]$SkipEnvironment
)

$ErrorActionPreference = 'Stop'

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "  OK $msg" -ForegroundColor Green }
function Warn($msg)     { Write-Host "  ! $msg" -ForegroundColor Yellow }

# --- 0. az CLI present + logged in ---
Step 0 "az CLI + login"
$null = az --version 2>$null
if ($LASTEXITCODE -ne 0) { throw "az CLI not on PATH. Install: https://aka.ms/install-azure-cli" }

$account = az account show 2>$null | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $account) {
    Warn "not logged in — running az login"
    az login | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "az login failed" }
    $account = az account show | ConvertFrom-Json
}
$SubscriptionId = $account.id
$SubscriptionName = $account.name
Ok "subscription: '$SubscriptionName' ($SubscriptionId)"

# --- 1. ML extension ---
Step 1 "az ml extension"
$mlExt = az extension show -n ml 2>$null | ConvertFrom-Json
if (-not $mlExt) {
    az extension add -n ml --yes | Out-Null
    Ok "installed az ml extension"
} else {
    Ok "az ml extension already installed (v$($mlExt.version))"
}

# --- 2. Resource group ---
Step 2 "resource group: $ResourceGroup"
$rg = az group show -n $ResourceGroup 2>$null | ConvertFrom-Json
if (-not $rg) {
    az group create -n $ResourceGroup -l $Location | Out-Null
    Ok "created $ResourceGroup in $Location"
} else {
    Ok "exists in $($rg.location)"
}

# --- 3. AML workspace ---
Step 3 "AML workspace: $Workspace"
$ws = az ml workspace show -n $Workspace --resource-group $ResourceGroup 2>$null | ConvertFrom-Json
if (-not $ws) {
    Write-Host "  (this takes ~2-3 min — creates Key Vault + App Insights + default storage)"
    az ml workspace create -n $Workspace --resource-group $ResourceGroup --location $Location | Out-Null
    Ok "created $Workspace"
} else {
    Ok "exists"
}

# --- 4. Storage account + container ---
Step 4 "storage account: $StorageAccount + container: $Container"
$sa = az storage account show -n $StorageAccount --resource-group $ResourceGroup 2>$null | ConvertFrom-Json
if (-not $sa) {
    # nameAvailability check up front for a clearer error than create's failure mode
    $check = az storage account check-name --name $StorageAccount | ConvertFrom-Json
    if (-not $check.nameAvailable) {
        throw "storage account name '$StorageAccount' not available: $($check.reason) - $($check.message)"
    }
    az storage account create -n $StorageAccount --resource-group $ResourceGroup `
        --location $Location --sku Standard_LRS --kind StorageV2 | Out-Null
    Ok "created $StorageAccount"
} else {
    Ok "exists"
}

# Use AAD auth to create the container (no key extraction)
$ctx = az storage container show --account-name $StorageAccount --name $Container --auth-mode login 2>$null
if ($LASTEXITCODE -ne 0) {
    az storage container create --account-name $StorageAccount --name $Container --auth-mode login | Out-Null
    Ok "created container $Container"
} else {
    Ok "container $Container exists"
}

# --- 5. GPU compute cluster ---
if (-not $SkipCompute) {
    Step 5 "compute cluster: $ComputeName ($VmSize)"
    $compute = az ml compute show -n $ComputeName --workspace-name $Workspace --resource-group $ResourceGroup 2>$null | ConvertFrom-Json
    if (-not $compute) {
        az ml compute create -n $ComputeName --type AmlCompute --size $VmSize `
            --min-instances 0 --max-instances $MaxInstances `
            --idle-time-before-scale-down $IdleScaleDownSec `
            --workspace-name $Workspace --resource-group $ResourceGroup | Out-Null
        Ok "submitted creation (state: provisioning — async)"
    } else {
        Ok "exists (state: $($compute.provisioning_state))"
    }
} else {
    Warn "skipping compute (-SkipCompute)"
}

# --- 6. AML environment ---
if (-not $SkipEnvironment) {
    Step 6 "environment: $EnvironmentName"
    $envYaml = Join-Path (Split-Path $PSScriptRoot -Parent) 'azure/environment.yml'
    if (-not (Test-Path $envYaml)) { throw "env yaml not found at $envYaml" }
    $existingEnv = az ml environment show -n $EnvironmentName --workspace-name $Workspace --resource-group $ResourceGroup --label latest 2>$null | ConvertFrom-Json
    if (-not $existingEnv) {
        az ml environment create -f $envYaml --workspace-name $Workspace --resource-group $ResourceGroup | Out-Null
        Ok "registered $EnvironmentName"
    } else {
        Ok "exists (latest version: $($existingEnv.version))"
    }
} else {
    Warn "skipping environment (-SkipEnvironment)"
}

# --- 7. Print .env block ---
Write-Host "`n====== PASTE INTO .env ======" -ForegroundColor Magenta
Write-Host @"
AZURE_SUBSCRIPTION_ID=$SubscriptionId
AZURE_RESOURCE_GROUP=$ResourceGroup
AZURE_ML_WORKSPACE=$Workspace
AZURE_REGION=$Location

AZURE_STORAGE_ACCOUNT=$StorageAccount
AZURE_STORAGE_CONTAINER=$Container

AZURE_COMPUTE_NAME=$ComputeName
AZURE_VM_SIZE=$VmSize
"@
Write-Host "==============================`n" -ForegroundColor Magenta

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Paste the block above into a fresh .env (copy .env.example as the starting point)."
Write-Host "  2. Add WANDB_API_KEY from https://wandb.ai/authorize"
Write-Host "  3. Run: uv run python scripts/check_azure.py"
