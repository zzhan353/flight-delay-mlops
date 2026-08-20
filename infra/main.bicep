// Infrastructure for the flight-delay service.
//
// Everything here is shaped by one constraint: the whole stack must cost under
// $1/month while still being a real deployment rather than a toy. That rules out an
// Azure ML managed online endpoint (~$100/month for the smallest instance) and an
// Azure Container Registry (Basic is ~$5/month on its own — more than the entire
// budget). The replacements are Container Apps on the consumption plan, which scales
// to zero and has a monthly free grant, and GitHub Container Registry, which is free
// for public images.
//
// The Azure ML workspace is still here, used for what it is uniquely good at — model
// registry, lineage and experiment tracking — while training itself runs on GitHub
// Actions runners. Nothing in this template provisions compute for training, which is
// why there is no ACR: no environment images are ever built in Azure.

targetScope = 'resourceGroup'

@description('Region for all resources')
param location string = resourceGroup().location

@description('Prefix for generated resource names')
@minLength(3)
@maxLength(11)
param namePrefix string = 'flightdelay'

@description('Container image to run, including tag')
param containerImage string = 'mcr.microsoft.com/k8se/quickstart:latest'

@description('Scale to zero when idle. Raise only if cold starts become unacceptable.')
@minValue(0)
@maxValue(1)
param minReplicas int = 0

@description('Ceiling on concurrent replicas — the guard against a surprise bill')
@minValue(1)
@maxValue(3)
param maxReplicas int = 2

var suffix = uniqueString(resourceGroup().id)
var tags = {
  project: 'flight-delay-mlops'
  managedBy: 'bicep'
  costCenter: 'portfolio'
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${namePrefix}-${suffix}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    // The free grant is 5 GB/month; a demo service will not approach it, but a
    // 30-day retention keeps the blast radius of a traffic spike small.
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${namePrefix}-${suffix}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ---------------------------------------------------------------------------
// Azure ML: registry and tracking only, no compute
// ---------------------------------------------------------------------------

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'st${namePrefix}${substring(suffix, 0, 6)}'
  location: location
  tags: tags
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${namePrefix}-${substring(suffix, 0, 8)}'
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-10-01' = {
  name: 'mlw-${namePrefix}'
  location: location
  tags: tags
  identity: { type: 'SystemAssigned' }
  properties: {
    friendlyName: 'Flight delay model registry'
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    // No containerRegistry: training never builds environment images in Azure, and
    // omitting it avoids the ~$5/month ACR Basic charge.
  }
}

// ---------------------------------------------------------------------------
// Serving
// ---------------------------------------------------------------------------

resource containerEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${namePrefix}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-${namePrefix}'
  location: location
  tags: tags
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      // No registries block: the image is public on ghcr.io, so there are no pull
      // credentials to store, rotate or leak.
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          resources: {
            // Smallest billable size. The model is ~4 MB and inference is single-row,
            // so anything larger is paying for idle memory.
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          probes: [
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 3
              periodSeconds: 5
            }
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        // Stay warm for 30 minutes after the last request instead of the 5-minute
        // default. Idle replicas are billed, but a visitor who arrives while the
        // service is asleep waits through a cold start, and someone reading a resume
        // rarely waits. 30 minutes covers a browsing session and an interview demo
        // without paying to run continuously.
        cooldownPeriod: 1800
        rules: [
          {
            name: 'http-concurrency'
            http: { metadata: { concurrentRequests: '20' } }
          }
        ]
      }
    }
  }
}

output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output containerAppName string = containerApp.name
output mlWorkspaceName string = mlWorkspace.name
output appInsightsConnectionString string = appInsights.properties.ConnectionString
