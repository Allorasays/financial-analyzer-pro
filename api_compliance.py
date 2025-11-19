"""
API Compliance Checker and Environment Configuration
Validates API usage against terms of service and manages feature flags
"""
import os
from enum import Enum
from typing import Dict, Optional

class APITier(Enum):
    """Application deployment tier"""
    DEVELOPMENT = "development"
    BETA = "beta"
    PRODUCTION = "production"

class APIStatus(Enum):
    """API availability status"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    REQUIRES_UPGRADE = "requires_upgrade"

# Get current tier from environment
APP_TIER = APITier(os.getenv("APP_TIER", "development").lower())

# API compliance matrix
API_COMPLIANCE = {
    "newsapi": {
        APITier.DEVELOPMENT: APIStatus.ENABLED,  # OK for local dev
        APITier.BETA: APIStatus.REQUIRES_UPGRADE,  # Must upgrade to Business plan
        APITier.PRODUCTION: APIStatus.REQUIRES_UPGRADE,  # Must upgrade to Business plan
    },
    "tiingo": {
        APITier.DEVELOPMENT: APIStatus.ENABLED,
        APITier.BETA: APIStatus.ENABLED,  # Free tier OK for limited beta
        APITier.PRODUCTION: APIStatus.REQUIRES_UPGRADE,  # Should upgrade for reliability
    },
    "fmp": {
        APITier.DEVELOPMENT: APIStatus.ENABLED,
        APITier.BETA: APIStatus.REQUIRES_UPGRADE,  # Free tier prohibits public use
        APITier.PRODUCTION: APIStatus.REQUIRES_UPGRADE,
    },
    "alpha_vantage": {
        APITier.DEVELOPMENT: APIStatus.ENABLED,
        APITier.BETA: APIStatus.ENABLED,  # Free tier OK with rate limiting
        APITier.PRODUCTION: APIStatus.REQUIRES_UPGRADE,
    },
    "fred": {
        APITier.DEVELOPMENT: APIStatus.ENABLED,
        APITier.BETA: APIStatus.ENABLED,  # Government data, free for public use
        APITier.PRODUCTION: APIStatus.ENABLED,
    },
    "yahoo": {
        APITier.DEVELOPMENT: APIStatus.ENABLED,
        APITier.BETA: APIStatus.ENABLED,  # Public scraping, acceptable with attribution
        APITier.PRODUCTION: APIStatus.ENABLED,
    },
}

def get_api_status(api_name: str) -> APIStatus:
    """Get the current status of an API based on app tier"""
    compliance = API_COMPLIANCE.get(api_name.lower(), {})
    return compliance.get(APP_TIER, APIStatus.DISABLED)

def is_api_enabled(api_name: str) -> bool:
    """Check if an API is enabled for the current tier"""
    status = get_api_status(api_name)
    
    # Check for override environment variable (e.g., NEWSAPI_ENABLED=true)
    override_env = os.getenv(f"{api_name.upper()}_ENABLED")
    if override_env:
        return override_env.lower() in ("true", "1", "yes")
    
    # Check for upgrade keys (e.g., NEWSAPI_BUSINESS_KEY means upgraded)
    upgrade_key_env = os.getenv(f"{api_name.upper()}_BUSINESS_KEY") or os.getenv(f"{api_name.upper()}_PREMIUM_KEY")
    if upgrade_key_env:
        return True  # If they have a premium key, allow it
    
    return status == APIStatus.ENABLED

def get_compliance_message(api_name: str) -> Optional[str]:
    """Get a compliance message for an API"""
    status = get_api_status(api_name)
    
    if status == APIStatus.REQUIRES_UPGRADE:
        upgrade_info = {
            "newsapi": "NewsAPI Business plan required for beta/production ($449/month)",
            "tiingo": "Tiingo Starter recommended for production ($10/month)",
            "fmp": "FMP Starter plan required for beta/production ($14/month)",
            "alpha_vantage": "Alpha Vantage Premium recommended for production ($49.99/month)",
        }
        return upgrade_info.get(api_name.lower(), f"{api_name} requires upgrade for {APP_TIER.value} tier")
    
    return None

def get_all_api_statuses() -> Dict[str, Dict]:
    """Get status of all APIs"""
    results = {}
    for api_name in API_COMPLIANCE.keys():
        status = get_api_status(api_name)
        enabled = is_api_enabled(api_name)
        message = get_compliance_message(api_name)
        
        results[api_name] = {
            "status": status.value,
            "enabled": enabled,
            "tier": APP_TIER.value,
            "message": message,
        }
    return results

def validate_beta_readiness() -> Dict[str, any]:
    """Validate if the app is ready for beta deployment"""
    issues = []
    warnings = []
    
    # Check critical APIs
    critical_apis = {
        "newsapi": {
            "critical": False,  # Can use alternative
            "message": "NewsAPI free tier not allowed for beta. Use Yahoo Finance or upgrade to Business plan.",
        },
        "tiingo": {
            "critical": True,  # Primary data source
            "message": "Tiingo free tier OK but may hit rate limits. Consider upgrading to Starter.",
        },
        "fmp": {
            "critical": False,  # Can use SEC EDGAR alternative
            "message": "FMP free tier not allowed for beta. Use SEC EDGAR or upgrade to Starter plan.",
        },
    }
    
    for api_name, info in critical_apis.items():
        status = get_api_status(api_name)
        enabled = is_api_enabled(api_name)
        
        if info["critical"] and not enabled and status == APIStatus.REQUIRES_UPGRADE:
            issues.append(f"Critical API '{api_name}' disabled: {info['message']}")
        elif not enabled and status == APIStatus.REQUIRES_UPGRADE:
            warnings.append(f"API '{api_name}' disabled: {info['message']}")
    
    return {
        "ready": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "app_tier": APP_TIER.value,
        "api_statuses": get_all_api_statuses(),
    }

if __name__ == "__main__":
    # Test compliance checker
    print(f"Current App Tier: {APP_TIER.value}")
    print("\nAPI Statuses:")
    for api, info in get_all_api_statuses().items():
        print(f"  {api}: {info['status']} (enabled: {info['enabled']})")
        if info['message']:
            print(f"    -> {info['message']}")
    
    print("\nBeta Readiness Check:")
    readiness = validate_beta_readiness()
    if readiness["ready"]:
        print("✅ App is ready for beta deployment")
    else:
        print("❌ App is NOT ready for beta deployment")
        print("Issues:")
        for issue in readiness["issues"]:
            print(f"  - {issue}")
    
    if readiness["warnings"]:
        print("\nWarnings:")
        for warning in readiness["warnings"]:
            print(f"  - {warning}")

