#!/usr/bin/env python3
"""
Fixed South Park Commons Portfolio Scraper
Addresses filename issues and updated HTML structure
"""

import json
import time
import pathlib
import re
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE             = "https://www.southparkcommons.com"
LIST_URL         = f"{BASE}/companies"
PAGE_PARAM       = "08159abd_page"
HEADERS          = {"User-Agent": "Mozilla/5.0 (SPC-Scraper/3.0)"}
SLEEP_SEC        = 0.5           # increased politeness delay
OUT_DIR          = pathlib.Path(".")

def sanitize_filename(name):
    """Convert industry names to valid filenames"""
    # Replace problematic characters with hyphens
    sanitized = re.sub(r'[/\\:*?"<>|]', '-', name)
    # Remove multiple consecutive hyphens
    sanitized = re.sub(r'-+', '-', sanitized)
    # Convert to lowercase and strip hyphens from ends
    return sanitized.lower().strip('-')

def get(url):
    """Fetch URL with error handling"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_card_data(card):
    """Extract basic data from company card"""
    try:
        name_elem = card.select_one("a")
        if not name_elem:
            return None

        name = name_elem.get_text(strip=True)

        # Extract description
        desc_elem = card.select_one("div.text-block-7")
        desc = desc_elem.get_text(strip=True) if desc_elem else "N/A"

        # Extract tags (industry, funding, year, location)
        tags = card.select("div.company-tag")
        tag_texts = [t.get_text(strip=True) for t in tags]

        # Ensure we have at least 4 tags, pad with N/A if needed
        while len(tag_texts) < 4:
            tag_texts.append("N/A")

        sector, stage, year, location = tag_texts[:4]

        # Get company slug
        slug = name_elem.get("href", "")

        return {
            "name": name,
            "description": desc,
            "industry": sector,
            "funding": stage,
            "year": year,
            "location": location,
            "slug": slug,
        }
    except Exception as e:
        print(f"Error extracting card data: {e}")
        return None

def enrich_with_detail(data):
    """Enrich company data with details from individual page"""
    if not data["slug"]:
        data["website"] = None
        data["founders"] = None
        return data

    html = get(BASE + data["slug"])
    if not html:
        data["website"] = None
        data["founders"] = None
        return data

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Try multiple selectors for website
        website = None
        website_selectors = [
            "a.button.is-link.external",
            "a[href*='http']:not([href*='southparkcommons.com'])",
            "a.external-link",
            "a[target='_blank']",
            "a.button[href*='http']",
            ".company-links a[href*='http']"
        ]

        for selector in website_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get("href", "").strip()
                if href.startswith("http") and "southparkcommons.com" not in href:
                    website = href
                    break
            if website:
                break

        # Try multiple approaches for founders
        founders = []

        # Method 1: Look for "Founded By" text
        founded_by_text = soup.find(string=re.compile(r"Founded By", re.I))
        if founded_by_text:
            parent = founded_by_text.parent
            if parent:
                # Try to find the next text node or element
                next_text = parent.find_next(string=True)
                if next_text:
                    founders = [next_text.strip()]

        # Method 2: Look for founder section headers
        if not founders:
            founder_headers = soup.find_all(string=re.compile(r"FOUNDERS?", re.I))
            for header in founder_headers:
                parent = header.parent
                if parent:
                    # Look for list items or paragraphs after the header
                    next_elem = parent.find_next_sibling()
                    if next_elem:
                        founder_items = next_elem.find_all(["li", "p"])
                        if founder_items:
                            for item in founder_items:
                                text = item.get_text(strip=True)
                                # Extract name (usually first word or first few words)
                                name_match = re.match(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
                                if name_match:
                                    founders.append(name_match.group(1))

        # Method 3: Look for common founder name patterns
        if not founders:
            page_text = soup.get_text()
            # Pattern like "Founded By John Smith"
            founded_by_match = re.search(r"Founded By\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", page_text)
            if founded_by_match:
                founders = [founded_by_match.group(1).strip()]

        # Method 4: Look for CEO or founder mentions
        if not founders:
            ceo_match = re.search(r"CEO of [^,]+,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", soup.get_text())
            if ceo_match:
                founders = [ceo_match.group(1).strip()]

        data["website"] = website
        data["founders"] = founders if founders else None

    except Exception as e:
        print(f"Error enriching {data['name']}: {e}")
        data["website"] = None
        data["founders"] = None

    return data

def paginate_cards():
    """Generator that yields company cards from all pages"""
    page = 1
    while True:
        url = f"{LIST_URL}?{PAGE_PARAM}={page}"
        html = get(url)
        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")
        cards = soup.select("div.div-block-15")

        if not cards:
            break

        for card in cards:
            card_data = extract_card_data(card)
            if card_data:
                yield card_data

        page += 1
        time.sleep(SLEEP_SEC)

def main():
    """Main scraping function"""
    print("Starting South Park Commons portfolio scraper...")

    records = []
    by_sector = defaultdict(list)

    # Collect all cards first
    cards = list(paginate_cards())
    print(f"Found {len(cards)} companies to process")

    # Process each company
    for card_data in tqdm(cards, desc="Enriching companies"):
        try:
            enriched = enrich_with_detail(card_data)
            records.append(enriched)

            # Add to sector grouping with sanitized filename
            sector_key = sanitize_filename(enriched["industry"])
            by_sector[sector_key].append(enriched)

            # Small delay to be respectful
            time.sleep(0.3)

        except Exception as e:
            print(f"Error processing {card_data.get('name', 'unknown')}: {e}")
            continue

    # Write master file
    try:
        with open("all.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"✓ Wrote {len(records)} companies to all.json")
    except Exception as e:
        print(f"Error writing all.json: {e}")

    # Write industry-specific files
    for sector, items in by_sector.items():
        try:
            filename = f"{sector}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"✓ Wrote {len(items)} companies to {filename}")
        except Exception as e:
            print(f"Error writing {filename}: {e}")

    print(f"\nComplete! Processed {len(records)} companies into {len(by_sector)} sector files")

if __name__ == "__main__":
    main()