"""
Shared connector enums and capability flags.

Phase 1 only defines the architecture. Future provider-specific
connectors should use these types instead of scattered string literals.
"""

from enum import Enum, Flag, auto


class ConnectorType(str, Enum):
    generic_http = "generic_http"
    generic_rest = "generic_rest"
    provider_api = "provider_api"
    official_sdk = "official_sdk"
    stac = "stac"
    ogc_api = "ogc_api"
    thredds = "thredds"
    erddap = "erddap"
    opendap = "opendap"
    ckan = "ckan"
    ftp = "ftp"
    s3 = "s3"
    browser = "browser"


class AuthenticationType(str, Enum):
    none = "none"
    api_key = "api_key"
    basic_auth = "basic_auth"
    oauth = "oauth"
    session_cookie = "session_cookie"
    user_credentials = "user_credentials"
    unknown = "unknown"


class AccessType(str, Enum):
    public = "public"
    self_registration = "self_registration"
    user_credentials_required = "user_credentials_required"
    organisation_account_required = "organisation_account_required"
    paid = "paid"
    unknown = "unknown"


class DatasetType(str, Enum):
    tabular = "tabular"
    raster = "raster"
    vector = "vector"
    gridded = "gridded"
    time_series = "time_series"
    document = "document"
    unknown = "unknown"


class CapabilityFlags(Flag):
    none = 0
    supports_metadata = auto()
    supports_download = auto()
    supports_authentication = auto()
    supports_registration = auto()
    supports_api = auto()
    supports_browser = auto()
    supports_subsetting = auto()
    supports_resume = auto()
    supports_checksum = auto()
    supports_parallel_download = auto()
    supports_dataset_search = auto()


class RetryStrategy(str, Enum):
    none = "none"
    fixed = "fixed"
    exponential_backoff = "exponential_backoff"
    provider_defined = "provider_defined"


class ValidationStrategy(str, Enum):
    basic_file = "basic_file"
    content_type = "content_type"
    checksum = "checksum"
    provider_defined = "provider_defined"
