"""Utility functions for the initiation pipeline."""

from .data_processing import (
    clean_text,
    extract_year_from_date,
    extract_name_from_linkedin_url,
    validate_email,
    validate_url,
    deduplicate_by_key,
    deduplicate_by_similarity,
    calculate_similarity,
    chunk_list,
    safe_float,
    safe_int,
    format_currency,
    parse_funding_amount,
    extract_emails_from_text,
    extract_urls_from_text,
    normalize_company_name,
    calculate_text_similarity,
    get_domain_from_url,
    is_recent_date,
    create_slug,
    format_phone_number,
    DataValidator,
    batch_process,
    retry_with_backoff,
    merge_dicts_safely
)

# Import centralized validation functions
from ..validators import (
    validate_linkedin_url,
    validate_url as validate_url_v2,
    validate_email as validate_email_v2,
    validate_phone,
    validate_year,
    validate_funding_amount,
    validate_company_name,
    validate_person_name,
    validate_text_length
)

__all__ = [
    # Data processing
    "clean_text",
    "extract_year_from_date",
    "extract_name_from_linkedin_url",
    "validate_email",  # Keep for backward compatibility
    "validate_url",    # Keep for backward compatibility
    "deduplicate_by_key",
    "deduplicate_by_similarity",
    "calculate_similarity",
    "chunk_list",
    "safe_float",
    "safe_int",
    "format_currency",
    "parse_funding_amount",
    "extract_emails_from_text",
    "extract_urls_from_text",
    "normalize_company_name",
    "calculate_text_similarity",
    "get_domain_from_url",
    "is_recent_date",
    "create_slug",
    "format_phone_number",
    "DataValidator",
    "batch_process",
    "retry_with_backoff",
    "merge_dicts_safely",
    
    # Centralized validators (preferred)
    "validate_linkedin_url",
    "validate_url_v2",
    "validate_email_v2",
    "validate_phone",
    "validate_year",
    "validate_funding_amount",
    "validate_company_name",
    "validate_person_name",
    "validate_text_length",
    
    # File handlers
    "read_json_async",
    "write_json_async",
    "read_csv_safe",
    "write_csv_safe",
    "read_excel_safe",
    "write_excel_safe",
    "get_file_size",
    "ensure_directory",
    "list_files_by_extension",
    "backup_file",
    "clean_filename"
]
