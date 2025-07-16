"""
fi_scraper_no_selenium.py
Scrapes Founder Institute Graduates list and saves to JSON without using Selenium.
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import re

def scrape_fi_graduates():
    """
    Scrapes Founder Institute graduates without using Selenium
    """

    url = "https://fi.co/graduates"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all graduate cards using the correct selector
        graduate_cards = soup.find_all('div', class_='js-graduate-each')

        companies = []

        for card in graduate_cards:
            try:
                # Get all text from the card
                all_text = card.get_text(separator='\n', strip=True)
                lines = [line.strip() for line in all_text.split('\n') if line.strip()]

                # Also get the website URL, if any
                website = None
                # If there is an <a ...> inside the card, and it looks like an external website
                for a in card.find_all("a", href=True):
                    href = a['href'].strip()
                    # usually external websites start with http and are not fi.co links
                    if href.startswith('http') and not href.startswith('https://fi.co'):
                        website = href
                        break

                # Initialize variables
                name = "N/A"
                location = "N/A"
                description = "N/A"
                founder = "N/A"
                industry = "N/A"

                # Parse the lines based on the pattern we discovered
                if len(lines) >= 3:
                    # Handle different patterns based on first line
                    if lines[0] in ['IPO\'d', 'Acquired', 'Unicorn']:
                        # Pattern: [Status] [Name] [Location] [Description] [Website] [Founder(s)] [Names]
                        if len(lines) >= 6:
                            name = lines[1]
                            location = lines[2]
                            description = lines[3]
                            if len(lines) >= 7 and lines[5] == "Founder(s)":
                                founder = lines[6]
                            elif len(lines) >= 6 and lines[4] == "Founder(s)":
                                founder = lines[5]
                    else:
                        # Pattern: [Name] [Location] [Description] [Website] [Founder(s)] [Names]
                        name = lines[0]
                        if len(lines) >= 5:
                            location = lines[1]
                            description = lines[2]
                            if len(lines) >= 6 and lines[4] == "Founder(s)":
                                founder = lines[5]
                            elif len(lines) >= 5 and lines[3] == "Founder(s)":
                                founder = lines[4]

                # Try to extract industry from description (basic attempt)
                industry_keywords = {
                    'AI': 'Artificial Intelligence',
                    'artificial intelligence': 'Artificial Intelligence',
                    'machine learning': 'Machine Learning',
                    'fintech': 'Financial Technology',
                    'edtech': 'Education Technology',
                    'health': 'Healthcare',
                    'medical': 'Healthcare',
                    'ecommerce': 'E-commerce',
                    'e-commerce': 'E-commerce',
                    'SaaS': 'Software as a Service',
                    'software': 'Software',
                    'mobile': 'Mobile Technology',
                    'app': 'Mobile Technology',
                    'security': 'Cybersecurity',
                    'crypto': 'Cryptocurrency',
                    'blockchain': 'Blockchain',
                    'marketplace': 'Marketplace',
                    'platform': 'Platform',
                    'analytics': 'Analytics',
                    'data': 'Data Analytics'
                }

                description_lower = description.lower()
                for keyword, industry_name in industry_keywords.items():
                    if keyword in description_lower:
                        industry = industry_name
                        break

                company = {
                    "name": name,
                    "description": description,
                    "industry": industry,
                    "location": location,
                    "founder(s)": founder,
                    "website": website if website else "N/A"
                }

                companies.append(company)

            except Exception as e:
                print(f"Error processing card: {e}")
                continue

        return companies

    except requests.RequestException as e:
        print(f"Request error: {e}")
        return []
    except Exception as e:
        print(f"General error: {e}")
        return []

def save_to_json(companies, filename="fi_graduates.json"):
    """Save companies data to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(companies, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(companies)} companies to {filename}")
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False

def save_industry_jsons(companies):
    """Save companies industry-wise"""
    industry_map = {}
    for company in companies:
        industry = company.get("industry", "unknown") or "unknown"
        # Use a safe and simple filename
        safe_industry = re.sub(r'\W+', '_', industry.strip().lower())
        if safe_industry == "":
            safe_industry = "unknown"
        industry_map.setdefault(safe_industry, []).append(company)

    for safe_industry, group in industry_map.items():
        fname = f"{safe_industry}.json"
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(group, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(group)} companies to {fname}")

def main():
    """Main function to run the scraper"""
    print("Scraping Founder Institute graduates...")
    companies = scrape_fi_graduates()

    if companies:
        print(f"Successfully extracted {len(companies)} companies")

        # Save all companies to all.json
        if save_to_json(companies, "all.json"):
            print("✓ Data successfully saved to all.json")
        else:
            print("✗ Failed to save all.json")

        # Save industry-wise JSONs
        save_industry_jsons(companies)

        # Show first few companies as examples
        print("\nFirst 3 companies:")
        for i, company in enumerate(companies[:3]):
            print(f"\n--- Company {i+1} ---")
            for key, value in company.items():
                print(f"{key}: {value}")

        print(f"\nTotal companies scraped: {len(companies)}")
    else:
        print("No companies found. Please check the scraper.")

if __name__ == "__main__":
    main()
