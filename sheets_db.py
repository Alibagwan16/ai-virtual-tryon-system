"""Google Sheets database layer for the AI Virtual Try-On app."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: Dict[str, tuple] = {}
_CACHE_TTL_SECONDS = 8

_client = None
_sheet = None
_init_attempted = False
_init_error: Optional[str] = None


# --------------------------------------------------------------------------
# Credentials / configuration
# --------------------------------------------------------------------------

def _get_credentials_info() -> Optional[dict]:
    """Get Google service-account credentials."""

    # Streamlit Cloud secrets
    try:
        import streamlit as st

        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])

    except Exception as e:
        logger.warning(
            "Could not read Streamlit secrets: %s",
            e,
        )

    # Environment variable containing complete JSON
    raw = os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "",
    ).strip()

    if raw:
        try:
            return json.loads(raw)

        except Exception as e:
            logger.warning(
                "GOOGLE_SERVICE_ACCOUNT_JSON is invalid: %s",
                e,
            )

    # Environment variable containing JSON file path
    path = os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        "",
    ).strip()

    if path and os.path.exists(path):
        try:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:
                return json.load(f)

        except Exception as e:
            logger.warning(
                "Could not read GOOGLE_SERVICE_ACCOUNT_FILE: %s",
                e,
            )

    return None


def _get_sheet_identifier() -> Optional[dict]:
    """Get Google Sheet ID or name."""

    # ------------------------------------------------------------------
    # Streamlit Cloud secrets
    # ------------------------------------------------------------------
    try:
        import streamlit as st

        # IMPORTANT:
        # GOOGLE_SHEET_ID is a ROOT-LEVEL Streamlit secret.
        # It is NOT inside [gcp_service_account].
        if "GOOGLE_SHEET_ID" in st.secrets:
            sheet_id = str(
                st.secrets["GOOGLE_SHEET_ID"]
            ).strip()

            if sheet_id:
                return {
                    "by": "id",
                    "value": sheet_id,
                }

        # Optional spreadsheet name
        if "GOOGLE_SHEET_NAME" in st.secrets:
            sheet_name = str(
                st.secrets["GOOGLE_SHEET_NAME"]
            ).strip()

            if sheet_name:
                return {
                    "by": "name",
                    "value": sheet_name,
                }

    except Exception as e:
        logger.warning(
            "Could not read sheet settings from Streamlit secrets: %s",
            e,
        )

    # ------------------------------------------------------------------
    # Environment variables
    # ------------------------------------------------------------------
    sheet_id = os.getenv(
        "GOOGLE_SHEET_ID",
        "",
    ).strip()

    if sheet_id:
        return {
            "by": "id",
            "value": sheet_id,
        }

    sheet_name = os.getenv(
        "GOOGLE_SHEET_NAME",
        "",
    ).strip()

    if sheet_name:
        return {
            "by": "name",
            "value": sheet_name,
        }

    return None


def is_configured() -> bool:
    return (
        _get_credentials_info() is not None
        and _get_sheet_identifier() is not None
    )


def init_error() -> Optional[str]:
    return _init_error


# --------------------------------------------------------------------------
# Google Sheets connection
# --------------------------------------------------------------------------

def _init() -> bool:
    """Connect to Google Sheets. Retries if an earlier attempt failed."""
    global _client, _sheet, _init_attempted, _init_error

    if _sheet is not None:
        return True

    creds_info = _get_credentials_info()
    sheet_ref = _get_sheet_identifier()

    if not creds_info:
        _init_error = "Google service-account credentials not found."
        _init_attempted = True
        print(f"[Google Sheets] {_init_error}", flush=True)
        return False

    if not sheet_ref:
        _init_error = "GOOGLE_SHEET_ID or GOOGLE_SHEET_NAME not found."
        _init_attempted = True
        print(f"[Google Sheets] {_init_error}", flush=True)
        return False

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        creds = Credentials.from_service_account_info(
            creds_info,
            scopes=scopes,
        )

        client = gspread.authorize(creds)

        if sheet_ref["by"] == "id":
            sheet = client.open_by_key(sheet_ref["value"])
        else:
            sheet = client.open(sheet_ref["value"])

        _client = client
        _sheet = sheet
        _init_attempted = True
        _init_error = None

        print(
            f"[Google Sheets] Connected successfully: {sheet.title}",
            flush=True,
        )

        return True

    except Exception as e:
        _init_error = f"{type(e).__name__}: {e}"
        _init_attempted = True

        print(
            f"[Google Sheets] CONNECTION FAILED: {_init_error}",
            flush=True,
        )

        return False
# --------------------------------------------------------------------------
# Worksheet handling
# --------------------------------------------------------------------------

def _get_or_create_worksheet(
    table_name: str,
    headers: List[str],
):
    """Get worksheet or create it if missing."""

    if not _init():
        return None

    try:
        worksheet = _sheet.worksheet(
            table_name
        )

    except Exception:
        # Worksheet doesn't exist -> create it
        try:
            worksheet = _sheet.add_worksheet(
                title=table_name,
                rows=1000,
                cols=max(
                    len(headers),
                    10,
                ),
            )

            worksheet.append_row(
                headers,
                value_input_option="USER_ENTERED",
            )

        except Exception:
            logger.exception(
                "Could not create worksheet '%s'",
                table_name,
            )

            return None

    # If worksheet exists but is empty, add headers
    try:
        values = worksheet.get_all_values()

        if not values:
            worksheet.append_row(
                headers,
                value_input_option="USER_ENTERED",
            )

    except Exception:
        pass

    return worksheet


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _records_from_worksheet(
    worksheet,
    headers: List[str],
) -> List[Dict[str, Any]]:

    try:
        rows = worksheet.get_all_records(
            expected_headers=headers
        )

        return [
            dict(row)
            for row in rows
        ]

    except Exception:
        # Fallback for sheets where headers are not
        # interpreted correctly by gspread.
        try:
            values = worksheet.get_all_values()

            if not values:
                return []

            actual_headers = values[0]

            records = []

            for row in values[1:]:
                padded = list(row)

                while len(padded) < len(actual_headers):
                    padded.append("")

                records.append(
                    dict(
                        zip(
                            actual_headers,
                            padded,
                        )
                    )
                )

            return records

        except Exception:
            logger.exception(
                "Could not read worksheet"
            )

            return []


# --------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------

def get_rows(
    table_name: str,
    headers: List[str],
) -> Optional[List[Dict[str, Any]]]:
    """Return rows from a Google Sheet table.

    None means Google Sheets is unavailable.
    [] means the sheet is available but contains no records.
    """

    if not _init():
        return None

    now = time.time()

    cached = _cache.get(
        table_name
    )

    if cached:
        fetched_at, records = cached

        if (
            now - fetched_at
            < _CACHE_TTL_SECONDS
        ):
            return records

    worksheet = _get_or_create_worksheet(
        table_name,
        headers,
    )

    if worksheet is None:
        return None

    try:
        records = _records_from_worksheet(
            worksheet,
            headers,
        )

        _cache[table_name] = (
            now,
            records,
        )

        return records

    except Exception:
        logger.exception(
            "Could not read Google Sheet table '%s'",
            table_name,
        )

        return None


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------

def append_row(
    table_name: str,
    headers: List[str],
    row: Dict[str, Any],
) -> bool:
    """Append one record to a Google Sheet."""

    if not _init():
        return False

    worksheet = _get_or_create_worksheet(
        table_name,
        headers,
    )

    if worksheet is None:
        return False

    try:
        values = [
            row.get(
                header,
                "",
            )
            for header in headers
        ]

        worksheet.append_row(
            values,
            value_input_option="USER_ENTERED",
        )

        # Clear cache so next read sees the new row.
        _cache.pop(
            table_name,
            None,
        )

        logger.info(
            "Google Sheets row appended successfully: %s",
            table_name,
        )

        return True

    except Exception as e:
        logger.exception(
            "Could not append row to '%s': %s",
            table_name,
            e,
        )

        return False

def delete_row(
    table_name: str,
    key_column: str,
    key_value: str,
) -> bool:
    """Delete the first record matching key_column/key_value."""

    if not _init():
        return False

    worksheet = _get_or_create_worksheet(
        table_name,
        [],
    )

    if worksheet is None:
        return False

    try:
        records = worksheet.get_all_records()
        headers = worksheet.row_values(1)

        if key_column not in headers:
            return False

        for row_number, record in enumerate(records, start=2):
            if str(record.get(key_column, "")).strip() == str(key_value).strip():
                worksheet.delete_rows(row_number)

                _cache.pop(
                    table_name,
                    None,
                )

                logger.info(
                    "Google Sheets row deleted successfully: %s",
                    table_name,
                )

                return True

        return False

    except Exception as e:
        logger.exception(
            "Could not delete row from '%s': %s",
            table_name,
            e,
        )

        return False

    


# --------------------------------------------------------------------------
# Update / upsert
# --------------------------------------------------------------------------

def upsert_row(
    table_name: str,
    headers: List[str],
    match: Dict[str, Any],
    updates: Dict[str, Any],
) -> bool:
    """Update an existing row matching `match`, or append it."""

    if not _init():
        return False

    worksheet = _get_or_create_worksheet(
        table_name,
        headers,
    )

    if worksheet is None:
        return False

    try:
        values = worksheet.get_all_values()

        if not values:
            worksheet.append_row(
                headers,
                value_input_option="USER_ENTERED",
            )

            values = [headers]

        actual_headers = values[0]

        # --------------------------------------------------------------
        # Find matching row
        # --------------------------------------------------------------
        match_row_number = None

        for row_number, row_values in enumerate(
            values[1:],
            start=2,
        ):
            row_dict = {}

            for index, header in enumerate(
                actual_headers
            ):
                row_dict[header] = (
                    row_values[index]
                    if index < len(row_values)
                    else ""
                )

            matched = True

            for key, expected in match.items():
                if str(
                    row_dict.get(
                        key,
                        "",
                    )
                ) != str(expected):
                    matched = False
                    break

            if matched:
                match_row_number = row_number
                break

        # --------------------------------------------------------------
        # Append new row
        # --------------------------------------------------------------
        if match_row_number is None:
            new_row = dict(match)
            new_row.update(updates)

            worksheet.append_row(
                [
                    new_row.get(
                        header,
                        "",
                    )
                    for header in headers
                ],
                value_input_option="USER_ENTERED",
            )

        # --------------------------------------------------------------
        # Update existing row
        # --------------------------------------------------------------
        else:
            current_values = list(
                values[
                    match_row_number - 1
                ]
            )

            while len(current_values) < len(headers):
                current_values.append("")

            header_indexes = {
                header: index
                for index, header in enumerate(
                    headers
                )
            }

            for key, value in updates.items():
                if key in header_indexes:
                    current_values[
                        header_indexes[key]
                    ] = value

            worksheet.update(
                f"A{match_row_number}:"
                f"{_column_letter(len(headers))}"
                f"{match_row_number}",
                [
                    current_values[
                        :len(headers)
                    ]
                ],
                value_input_option="USER_ENTERED",
            )

        _cache.pop(
            table_name,
            None,
        )

        return True

    except Exception:
        logger.exception(
            "Could not upsert row in '%s'",
            table_name,
        )

        return False


# --------------------------------------------------------------------------
# Column helper
# --------------------------------------------------------------------------

def _column_letter(number: int) -> str:
    """Convert 1 -> A, 2 -> B, ..."""

    result = ""

    while number:
        number, remainder = divmod(
            number - 1,
            26,
        )

        result = (
            chr(65 + remainder)
            + result
        )

    return result