
        _init_error = "Missing configuration: " + ", ".join(missing)
        logger.warning(_init_error)

        _init_attempted = True
        return False

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = (
            client.open_by_key(sheet_ref["value"])
            if sheet_ref["by"] == "id"
            else client.open(sheet_ref["value"])
        )
        _client = client
        _sheet = sheet
        _init_attempted = True
        return True
    except Exception as e:  # noqa: BLE001
        _init_error = str(e)
        _init_attempted = True
        logger.warning(
            "Google Sheets storage is configured but couldn't connect (%s) — "
            "falling back to local JSON files for this run.", e,
        )
        return False


def _get_or_create_worksheet(name: str, headers: List[str]):
    assert _sheet is not None
    try:
        ws = _sheet.worksheet(name)
    except Exception:  # noqa: BLE001
        ws = _sheet.add_worksheet(title=name, rows=200, cols=max(10, len(headers)))
        ws.append_row(headers)
        return ws
    if not ws.row_values(1):
        ws.append_row(headers)
    return ws


def _invalidate(table: str) -> None:
    with _lock:
        _cache.pop(table, None)


# --------------------------------------------------------------------------
# Public read/write API — generic, table-name + headers driven
# --------------------------------------------------------------------------
def get_rows(table: str, headers: List[str], force_refresh: bool = False) -> Optional[List[Dict[str, Any]]]:
    """Returns every row of `table` as a list of {column: value} dicts, or
    None if Sheets isn't configured/reachable — callers should fall back to
    their local JSON file in that case. Cached briefly so a page full of
    Streamlit widgets re-running doesn't re-hit the Sheets API every time."""
    if not _init():
        return None

    if not force_refresh:
        with _lock:
            cached = _cache.get(table)
        if cached and (time.time() - cached[0] < _CACHE_TTL_SECONDS):
            return [dict(r) for r in cached[1]]

    try:
        ws = _get_or_create_worksheet(table, headers)
        records = ws.get_all_records()
    except Exception as e:  # noqa: BLE001
        logger.warning("Sheets read failed for table %r: %s", table, e)
        return None

    with _lock:
        _cache[table] = (time.time(), records)
    return [dict(r) for r in records]


def append_row(table: str, headers: List[str], row: Dict[str, Any]) -> bool:
    """Appends one new row. Returns True on success, False if Sheets isn't
    configured/reachable (caller should fall back to its local JSON file)."""
    if not _init():
        return False
    try:
        ws = _get_or_create_worksheet(table, headers)
        ws.append_row([row.get(h, "") for h in headers])
        _invalidate(table)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Sheets append failed for table %r: %s", table, e)
        return False


def upsert_row(table: str, headers: List[str], match: Dict[str, Any], updates: Dict[str, Any]) -> bool:
    """Finds the first row where EVERY column in `match` equals its given
    value, and merges `updates` into it; appends a brand-new row
    (match + updates) if no row matched. Returns True on success, False if
    Sheets isn't configured/reachable."""
    if not _init():
        return False
    try:
        ws = _get_or_create_worksheet(table, headers)
        records = ws.get_all_records()
        row_index = None
        for i, r in enumerate(records):
            if all(str(r.get(k, "")) == str(v) for k, v in match.items()):
                row_index = i + 2  # +1 header row, +1 for 1-indexing
                break
        if row_index is None:
            new_row = dict(match)
            new_row.update(updates)
            ws.append_row([new_row.get(h, "") for h in headers])
        else:
            current = dict(records[row_index - 2])
            current.update(updates)
            ws.update(f"A{row_index}", [[current.get(h, "") for h in headers]])
        _invalidate(table)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Sheets upsert failed for table %r: %s", table, e)
        return False
