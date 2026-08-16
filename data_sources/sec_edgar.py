"""SEC EDGAR client - free, no API key, full historical depth.

Used to extend stage 2 (drop-reason classification) beyond Finnhub's
~1-year free-tier news retention: 8-K filings are the primary-source,
company-filed disclosure of quarterly results and guidance changes, and
EDGAR's submissions API has this going back years for essentially every
current S&P 500 ticker (verified: AAPL's "recent" filings block alone
reaches back to 2015 without needing pagination).

SEC's fair-access policy asks for a descriptive User-Agent (contact info
included) and requests capped around 10/sec; this client stays under
that with a fixed per-request delay rather than a token bucket, since
there's no rate-limit header to react to like Finnhub's.
"""
import re
import time
import datetime as dt
import requests

_USER_AGENT = "stock-screener-research contact:development@myoctopus.ai"
_HEADERS = {"User-Agent": _USER_AGENT}
_REQUEST_DELAY_SECONDS = 0.12  # ~8 req/sec, under SEC's ~10/sec guidance
_MAX_RETRIES = 4

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
_ADDL_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{filename}"
_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/index.json"
_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{filename}"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _get(url, session, parse_json=True):
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            time.sleep(_REQUEST_DELAY_SECONDS)
            resp = session.get(url, headers=_HEADERS, timeout=20)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json() if parse_json else resp.text
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_exc


def load_ticker_cik_map(session=None):
    """Returns {ticker: zero-padded 10-digit CIK string}. Tickers already
    use our canonical dash form (BRK-B, BF-B) - verified directly against
    this exact file, no aliasing needed."""
    sess = session or requests.Session()
    data = _get(_TICKERS_URL, sess)
    return {row["ticker"]: str(row["cik_str"]).zfill(10) for row in data.values()}


def fetch_8k_filings(cik10, earliest_date, session=None):
    """Returns a list of dicts (accession_number, filing_date, primary_document,
    items) for every 8-K filed on/after `earliest_date`. Follows EDGAR's
    older-filings pagination automatically if the 'recent' block doesn't
    reach back far enough."""
    sess = session or requests.Session()
    out = []

    def scan_block(form, filing_date, accession, primary_doc, items):
        for i, f in enumerate(form):
            if f != "8-K":
                continue
            if filing_date[i] < earliest_date:
                continue
            out.append({
                "accession_number": accession[i],
                "filing_date": filing_date[i],
                "primary_document": primary_doc[i],
                "items": items[i] if items else "",
            })

    data = _get(_SUBMISSIONS_URL.format(cik10=cik10), sess)
    if data is None:
        return out
    recent = data["filings"]["recent"]
    scan_block(recent["form"], recent["filingDate"], recent["accessionNumber"],
               recent["primaryDocument"], recent.get("items"))

    earliest_seen = min(recent["filingDate"]) if recent["filingDate"] else "9999-99-99"
    for older_file in data["filings"].get("files", []):
        if older_file.get("filingTo", "0000-00-00") < earliest_date:
            continue  # this paginated file is entirely before our window
        older = _get(_ADDL_SUBMISSIONS_URL.format(filename=older_file["name"]), sess)
        if older is None:
            continue
        scan_block(older["form"], older["filingDate"], older["accessionNumber"],
                   older["primaryDocument"], older.get("items"))

    return out


def fetch_filing_text(cik10, accession_number, primary_document, session=None):
    """Fetches the primary 8-K document plus its press-release exhibit
    (filename containing 'ex99', the standard EX-99.1 slot for earnings
    releases), strips HTML, and returns combined plain text. Returns ''
    if nothing could be fetched."""
    sess = session or requests.Session()
    cik_nolead = str(int(cik10))
    accn_nodash = accession_number.replace("-", "")

    texts = []
    primary_html = _get(_DOC_URL.format(cik=cik_nolead, accn_nodash=accn_nodash,
                                         filename=primary_document), sess, parse_json=False)
    if primary_html:
        texts.append(primary_html)

    index_data = _get(_INDEX_URL.format(cik=cik_nolead, accn_nodash=accn_nodash), sess)
    if index_data:
        for item in index_data.get("directory", {}).get("item", []):
            name = item.get("name", "")
            # exhibit-99 naming varies by filing agent: ex991.htm, ex-99.1.htm,
            # exhibit991.htm, exhibit99-1.htm, etc. - match the family, not one form.
            if re.search(r"(?i)ex(?:hibit)?[-_]?99", name):
                if name == primary_document:
                    continue
                exhibit_html = _get(_DOC_URL.format(cik=cik_nolead, accn_nodash=accn_nodash,
                                                     filename=name), sess, parse_json=False)
                if exhibit_html:
                    texts.append(exhibit_html)

    combined = " ".join(texts)
    plain = _TAG_RE.sub(" ", combined)
    return _WS_RE.sub(" ", plain).strip()
