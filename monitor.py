#!/usr/bin/env python3
"""Roblox AFK Monitor.

Polls the Roblox Presence API for a list of tracked user IDs, writes a JSON
snapshot to ``data/status.json`` (consumed by the GitHub Pages dashboard in
``index.html``) and sends a red Discord webhook embed for every user that is NOT
currently in a game (``presenceType != 2``) - that is the "AFK" signal.

Runtime dependencies: ``requests`` (standard library otherwise).

Environment variables (required):
    ROBLOX_USER_IDS      Comma-separated Roblox user IDs, e.g. "123,456".
    DISCORD_WEBHOOK_URL  Full Discord webhook URL. Never printed to the log.

Exit codes:
    0  status.json was written successfully. Discord delivery failures are
       reported as warnings only, so the workflow can still commit the snapshot.
    1  Bad configuration, or an unrecoverable Roblox API / network error.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - dependency guard
    print("[monitor] FATAL: the 'requests' package is missing. Run: pip install requests")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATUS_FILE = DATA_DIR / "status.json"

PRESENCE_API_URL = "https://presence.roblox.com/v1/presence/users"
THUMBNAIL_API_URL = "https://thumbnails.roblox.com/v1/users/avatar-headshot"

REQUEST_TIMEOUT = 15  # seconds
POLL_INTERVAL_SECONDS = 300  # 5 minutes, must match the workflow cron + UI copy

# Retry policy for the Roblox endpoints (presence + thumbnails). Discord is
# intentionally excluded: a failed alert must never delay the snapshot.
RETRY_MAX_ATTEMPTS = 3  # 1 initial attempt + 2 retries
RETRY_BASE_DELAY_SECONDS = 5.0  # 5s, then 10s (exponential backoff)
RETRY_MAX_DELAY_SECONDS = 60.0  # hard cap, also honours a sane Retry-After
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

AVATAR_SIZE = "48x48"
AVATAR_FORMAT = "Png"
THUMBNAIL_BATCH_SIZE = 100  # Roblox thumbnails API accepts at most 100 ids / call

IN_GAME_PRESENCE_TYPE = 2
EMBED_COLOR_ALERT = 16711680  # decimal red

# Discord webhook limits (safe, slightly below the hard caps).
DISCORD_MAX_EMBEDS_PER_MESSAGE = 10
DISCORD_MAX_EMBED_CHARS = 5500  # hard cap is 6000, keep a margin

EXIT_OK = 0
EXIT_ERROR = 1

PRESENCE_TYPE_LABELS: dict[int, str] = {
    0: "Offline",
    1: "Online",
    2: "InGame",
    3: "In Studio",
}
UNKNOWN_PRESENCE_LABEL = "Unknown"

# Values registered here are scrubbed from every line we emit, so the Discord
# webhook URL can never leak into GitHub Actions logs.
_SECRET_VALUES: list[str] = []

# Any Discord webhook URL, regardless of casing, is masked even when it was never
# registered as a secret (e.g. it came from an error message).
WEBHOOK_URL_RE = re.compile(
    r"https://discord(?:app)?\.com/api/webhooks/[^/\s]+/[^/\s]+",
    re.IGNORECASE,
)
REDACTION_PLACEHOLDER = "***REDACTED***"


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #


def register_secret(value: str | None) -> None:
    """Register a value (e.g. the webhook URL) for log redaction."""
    if value and len(value.strip()) >= 8:
        _SECRET_VALUES.append(value.strip())


def redact(text: Any) -> str:
    """Mask webhook-looking URLs and every registered secret in a string."""
    cleaned = WEBHOOK_URL_RE.sub(REDACTION_PLACEHOLDER, str(text))
    for secret in _SECRET_VALUES:
        cleaned = cleaned.replace(secret, REDACTION_PLACEHOLDER)
        # Also hide the token part of the webhook path on its own.
        tail = secret.rstrip("/").rsplit("/", 1)[-1]
        if len(tail) >= 16:
            cleaned = cleaned.replace(tail, REDACTION_PLACEHOLDER)
    return cleaned


def log(message: str) -> None:
    """Print a timestamped, redacted line to stdout."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {redact(message)}", flush=True)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string ending in Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# HTTP request helper (retry + exponential backoff)
# --------------------------------------------------------------------------- #


def _retry_delay(attempt: int, response: requests.Response | None) -> float:
    """Backoff for retry ``attempt`` (1-based), honouring a sane Retry-After."""
    delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))

    if response is not None:
        raw_retry_after = str(response.headers.get("Retry-After", "")).strip()
        if raw_retry_after:
            hinted: float | None = None
            if raw_retry_after.isdigit():
                hinted = float(raw_retry_after)
            else:
                # RFC 7231 also allows an HTTP-date.
                try:
                    when = parsedate_to_datetime(raw_retry_after)
                except (TypeError, ValueError):
                    hinted = None
                else:
                    if when is not None:
                        if when.tzinfo is None:
                            when = when.replace(tzinfo=timezone.utc)
                        hinted = (when - datetime.now(timezone.utc)).total_seconds()
            if hinted is not None:
                delay = max(delay, hinted)

    return min(max(delay, 0.0), RETRY_MAX_DELAY_SECONDS)


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    label: str,
    **kwargs: Any,
) -> tuple[requests.Response | None, str | None]:
    """Send a request, retrying transient failures with exponential backoff.

    Retries on transport errors, HTTP 429 and HTTP 5xx, up to
    ``RETRY_MAX_ATTEMPTS`` total attempts (2 retries, 5s then 10s).

    Returns ``(response, error)``. ``error`` is set only when every attempt
    failed at the transport level (in which case ``response`` is ``None``);
    otherwise ``response`` is the final response, which may still carry a
    retryable status code if the attempts were exhausted.
    """
    last_error: str | None = None

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        response: requests.Response | None = None
        try:
            response = session.request(method, url, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_error = redact(exc)
        else:
            if response.status_code not in RETRYABLE_STATUS_CODES:
                return response, None
            last_error = f"HTTP {response.status_code}"

        if attempt >= RETRY_MAX_ATTEMPTS:
            log(
                f"WARN: {label} still failing after {RETRY_MAX_ATTEMPTS} attempt(s) "
                f"({last_error}); giving up."
            )
            break

        delay = _retry_delay(attempt, response)
        log(
            f"WARN: {label} failed ({last_error}); "
            f"retry {attempt}/{RETRY_MAX_ATTEMPTS - 1} in {delay:.0f}s."
        )
        time.sleep(delay)

    # ``error`` is only meaningful for transport-level exhaustion; a response
    # (even a 429/5xx one) is returned so the caller can inspect the status.
    return response, last_error if response is None else None


# --------------------------------------------------------------------------- #
# Input parsing / configuration
# --------------------------------------------------------------------------- #


def parse_user_ids(raw: str) -> tuple[list[int], list[str]]:
    """Parse a comma-separated list of Roblox user IDs.

    Returns ``(valid_ids, invalid_tokens)``, preserving order and dropping
    duplicates.
    """
    valid: list[int] = []
    invalid: list[str] = []
    seen: set[int] = set()

    for chunk in raw.replace("\n", ",").replace(";", ",").split(","):
        token = chunk.strip()
        if not token:
            continue
        try:
            user_id = int(token)
        except ValueError:
            invalid.append(token)
            continue
        if user_id <= 0:
            invalid.append(token)
            continue
        if user_id in seen:
            continue
        seen.add(user_id)
        valid.append(user_id)

    return valid, invalid


def load_configuration() -> tuple[list[int], str]:
    """Read and validate the required environment variables, or exit(1)."""
    raw_ids = os.environ.get("ROBLOX_USER_IDS", "").strip()
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

    register_secret(webhook_url)

    missing = [
        name
        for name, value in (("ROBLOX_USER_IDS", raw_ids), ("DISCORD_WEBHOOK_URL", webhook_url))
        if not value
    ]
    if missing:
        for name in missing:
            log(f"ERROR: environment variable {name} is empty or not set.")
        log(
            "Hint: set ROBLOX_USER_IDS='123,456' and "
            "DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'"
        )
        sys.exit(EXIT_ERROR)

    if not webhook_url.lower().startswith(("http://", "https://")):
        log("ERROR: DISCORD_WEBHOOK_URL is not a valid http(s) URL.")
        sys.exit(EXIT_ERROR)

    user_ids, invalid = parse_user_ids(raw_ids)
    if invalid:
        log(f"WARN: ignoring non-numeric ROBLOX_USER_IDS entries: {', '.join(invalid)}")
    if not user_ids:
        log("ERROR: ROBLOX_USER_IDS did not contain any valid numeric user IDs.")
        sys.exit(EXIT_ERROR)

    return user_ids, webhook_url


# --------------------------------------------------------------------------- #
# Roblox Presence API
# --------------------------------------------------------------------------- #


def fetch_presences(session: requests.Session, user_ids: list[int]) -> list[dict[str, Any]]:
    """POST to the Roblox presence API and return the ``userPresences`` list."""
    payload = {"userIds": user_ids}
    headers = {"Content-Type": "application/json"}

    log(f"Querying Roblox presence API for {len(user_ids)} user ID(s)...")
    response, transport_error = request_with_retry(
        session,
        "POST",
        PRESENCE_API_URL,
        label="Roblox presence API",
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    if transport_error is not None or response is None:
        log(f"ERROR: request to the Roblox presence API failed: {transport_error}")
        log("       No snapshot written; the next scheduled run will retry.")
        sys.exit(EXIT_ERROR)

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "unknown")
        log("ERROR: rate limited by the Roblox presence API (HTTP 429).")
        log(
            f"       Retry-After: {retry_after}; retries exhausted after "
            f"{RETRY_MAX_ATTEMPTS} attempt(s). Exiting so the next scheduled run retries."
        )
        sys.exit(EXIT_ERROR)

    if response.status_code != 200:
        log(f"ERROR: Roblox presence API returned HTTP {response.status_code}.")
        log(f"       Response body: {redact(response.text[:300])}")
        sys.exit(EXIT_ERROR)

    try:
        data = response.json()
    except ValueError:
        log("ERROR: Roblox presence API returned a non-JSON body.")
        log(f"       Response body: {redact(response.text[:300])}")
        sys.exit(EXIT_ERROR)

    presences = data.get("userPresences") if isinstance(data, dict) else None
    if not isinstance(presences, list):
        log("ERROR: Roblox presence API response is missing the 'userPresences' list.")
        sys.exit(EXIT_ERROR)

    log(f"Received presence data for {len(presences)} user(s).")
    return [item for item in presences if isinstance(item, dict)]


# --------------------------------------------------------------------------- #
# Roblox Thumbnails API (avatars)
# --------------------------------------------------------------------------- #


def fetch_avatars(session: requests.Session, user_ids: list[int]) -> dict[int, str]:
    """Return a ``{userId: imageUrl}`` map for avatar headshots.

    Failures are non-fatal: any user without a usable avatar maps to ``""`` so
    the dashboard can render a placeholder.
    """
    avatars: dict[int, str] = {user_id: "" for user_id in user_ids}
    if not user_ids:
        return avatars

    for start in range(0, len(user_ids), THUMBNAIL_BATCH_SIZE):
        batch = user_ids[start : start + THUMBNAIL_BATCH_SIZE]
        params = {
            "userIds": ",".join(str(user_id) for user_id in batch),
            "size": AVATAR_SIZE,
            "format": AVATAR_FORMAT,
        }

        response, transport_error = request_with_retry(
            session,
            "GET",
            THUMBNAIL_API_URL,
            label="Roblox thumbnails API",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        if transport_error is not None or response is None:
            log(f"WARN: avatar request failed ({transport_error}); continuing without avatars.")
            continue

        if response.status_code != 200:
            log(
                f"WARN: Roblox thumbnails API returned HTTP {response.status_code}; "
                "avatars for this batch fall back to an empty value."
            )
            continue

        try:
            payload = response.json()
        except ValueError:
            log("WARN: Roblox thumbnails API returned a non-JSON body; skipping avatars.")
            continue

        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            continue

        resolved = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            target = entry.get("targetId")
            image_url = entry.get("imageUrl")
            if not isinstance(target, int) or target not in avatars:
                continue
            if entry.get("state") not in (None, "Completed"):
                continue
            if isinstance(image_url, str) and image_url.strip():
                avatars[target] = image_url.strip()
                resolved += 1

        log(f"Resolved {resolved}/{len(batch)} avatar URL(s) for this batch.")

    found = sum(1 for url in avatars.values() if url)
    log(f"Avatars available for {found}/{len(user_ids)} account(s).")
    return avatars


# --------------------------------------------------------------------------- #
# status.json
# --------------------------------------------------------------------------- #


def presence_label(presence_type: Any) -> str:
    """Map a Roblox presenceType to its human label."""
    try:
        key = int(presence_type)
    except (TypeError, ValueError):
        return UNKNOWN_PRESENCE_LABEL
    return PRESENCE_TYPE_LABELS.get(key, f"{UNKNOWN_PRESENCE_LABEL} ({presence_type})")


def coerce_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_accounts(
    presences: list[dict[str, Any]], avatars: dict[int, str]
) -> list[dict[str, Any]]:
    """Normalise presence records into the status.json ``accounts`` shape."""
    accounts: list[dict[str, Any]] = []

    for presence in presences:
        user_id = coerce_int(presence.get("userId"))
        if user_id is None:
            log("WARN: skipping a presence record without a usable userId.")
            continue

        presence_type = coerce_int(presence.get("userPresenceType"), 0)

        accounts.append(
            {
                "userId": user_id,
                "status": presence_label(presence_type),
                "presenceType": presence_type,
                "placeId": coerce_int(presence.get("placeId")),
                "lastLocation": clean_text(presence.get("lastLocation")),
                "lastOnline": clean_text(presence.get("lastOnline")),
                "avatar": avatars.get(user_id, ""),
            }
        )

    accounts.sort(key=lambda account: account["userId"])
    return accounts


def build_summary(accounts: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate per-status counters for the dashboard stat cards."""
    summary = {"total": len(accounts), "inGame": 0, "online": 0, "offline": 0, "inStudio": 0}

    for account in accounts:
        presence_type = account["presenceType"]
        if presence_type == 2:
            summary["inGame"] += 1
        elif presence_type == 1:
            summary["online"] += 1
        elif presence_type == 3:
            summary["inStudio"] += 1
        else:
            summary["offline"] += 1

    return summary


def write_status_file(accounts: list[dict[str, Any]]) -> Path:
    """Write data/status.json, creating the data directory when needed."""
    payload = {
        "updatedAt": utc_now_iso(),
        "pollIntervalSeconds": POLL_INTERVAL_SECONDS,
        "summary": build_summary(accounts),
        "accounts": accounts,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(f"Wrote {STATUS_FILE.relative_to(BASE_DIR)} ({payload['summary']['total']} account(s)).")
    return STATUS_FILE


# --------------------------------------------------------------------------- #
# Discord notifications
# --------------------------------------------------------------------------- #


def build_alert_embed(account: dict[str, Any], checked_at: str) -> dict[str, Any]:
    """Build the red embed describing one non-InGame account."""
    user_id = account["userId"]
    status = account["status"]

    fields: list[dict[str, Any]] = [
        {"name": "User ID", "value": str(user_id), "inline": True},
        {"name": "Status", "value": status, "inline": True},
    ]

    place_id = account.get("placeId")
    if place_id:
        fields.append({"name": "Place ID", "value": str(place_id), "inline": True})

    last_location = account.get("lastLocation") or "Unknown"
    # Redact defensively: never forward a secret-looking string to Discord.
    fields.append(
        {
            "name": "Last Location",
            "value": f"`{redact(last_location)[:180]}`",
            "inline": False,
        }
    )

    last_online = account.get("lastOnline")
    if last_online:
        fields.append({"name": "Last Online", "value": f"`{redact(last_online)[:60]}`", "inline": True})

    fields.append({"name": "Timestamp", "value": checked_at, "inline": True})

    return {
        "title": f"AFK Alert - User {user_id}",
        "description": f"User **{user_id}** is **{status}** (not in a game).",
        "color": EMBED_COLOR_ALERT,
        "fields": fields,
        "footer": {"text": f"Roblox AFK Monitor | presenceType={account['presenceType']}"},
        "timestamp": checked_at,
    }


def send_discord_message(
    session: requests.Session, webhook_url: str, embeds: list[dict[str, Any]]
) -> bool:
    """Deliver one batch of embeds to the Discord webhook. Never raises."""
    body = {"username": "Roblox AFK Monitor", "embeds": embeds}

    try:
        response = session.post(webhook_url, json=body, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        log(f"WARN: failed to deliver Discord notification: {redact(exc)}")
        return False

    if response.status_code == 429:
        log("WARN: Discord rate limited the webhook (HTTP 429); notification dropped, not retrying.")
        return False

    if response.status_code not in (200, 204):
        log(f"WARN: Discord webhook returned HTTP {response.status_code}: {redact(response.text[:200])}")
        return False

    return True


def chunk_embeds(embeds: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group embeds into webhook-sized batches (<=10 embeds, <=~5500 chars)."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_len = 0

    for embed in embeds:
        size = len(json.dumps(embed, ensure_ascii=False))
        too_many = len(current) >= DISCORD_MAX_EMBEDS_PER_MESSAGE
        too_long = bool(current) and current_len + size > DISCORD_MAX_EMBED_CHARS
        if too_many or too_long:
            batches.append(current)
            current, current_len = [], 0
        current.append(embed)
        current_len += size

    if current:
        batches.append(current)

    return batches


def notify_non_in_game(
    session: requests.Session, webhook_url: str, accounts: list[dict[str, Any]], checked_at: str
) -> tuple[int, int]:
    """Send red embeds for every account with presenceType != 2.

    Returns ``(sent_count, failed_count)`` measured in accounts.
    """
    offenders = [
        account for account in accounts if account["presenceType"] != IN_GAME_PRESENCE_TYPE
    ]

    if not offenders:
        log("No alerts needed - every tracked account is in a game.")
        return 0, 0

    log(f"{len(offenders)} account(s) not in a game - preparing Discord alert(s)...")
    embeds = [build_alert_embed(account, checked_at) for account in offenders]
    batches = chunk_embeds(embeds)

    sent = 0
    failed = 0
    for index, batch in enumerate(batches, start=1):
        log(f"Sending Discord batch {index}/{len(batches)} ({len(batch)} embed(s))...")
        if send_discord_message(session, webhook_url, batch):
            sent += len(batch)
        else:
            failed += len(batch)

    return sent, failed


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    user_ids, webhook_url = load_configuration()
    checked_at = utc_now_iso()

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Roblox-AFK-Monitor/2.0"})

        presences = fetch_presences(session, user_ids)
        avatars = fetch_avatars(session, user_ids)
        accounts = build_accounts(presences, avatars)
        write_status_file(accounts)

        sent, failed = notify_non_in_game(session, webhook_url, accounts, checked_at)

    summary = build_summary(accounts)
    not_in_game = summary["total"] - summary["inGame"]

    log("-" * 66)
    log(
        f"SUMMARY: total={summary['total']} | inGame={summary['inGame']} | checkedAt={checked_at}"
    )
    log(
        f"         online={summary['online']} | offline={summary['offline']} "
        f"| inStudio={summary['inStudio']} | notInGame={not_in_game}"
    )
    log(f"         discord: {sent} alert(s) sent, {failed} delivery failure(s)")

    if failed:
        # status.json is already on disk, so the workflow must keep going and
        # commit/push the snapshot. A flaky Discord webhook is only a warning.
        log(
            f"WARNING: {failed} Discord alert(s) could not be delivered (see warnings above). "
            "status.json was written successfully, so exiting 0 to let the workflow commit it."
        )
    else:
        log("RESULT: finished successfully.")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())