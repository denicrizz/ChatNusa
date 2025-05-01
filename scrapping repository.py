from bs4 import BeautifulSoup as bs
import requests as req
import json
import re
from datetime import datetime

def scrape_repository():
    # Fetch data from website
    reqWeb = req.get("https://repository.unpkediri.ac.id/view/subjects/462.html")
    results = []
    datas = re.findall(r'<p>(.*?)<\/p>', reqWeb.text, re.DOTALL)
    
    # Process each data entry
    for data in datas:
        # Extract authors
        author_matches = re.findall(r'<span class="person_name">(.*?)<\/span>', data)
        authors = [author.strip() for author in author_matches]
        
        # Extract year
        year_match = re.search(r'\((\d{4})\)', data)
        year = year_match.group(1) if year_match else ""
        
        # Extract title
        title_match = re.search(r'<em>(.*?)<\/em>', data)
        title = title_match.group(1) if title_match else ""
        
        # Extract link
        link_match = re.search(r'<a href="(.*?)">', data)
        link = link_match.group(1) if link_match else ""
        
        # Extract source
        source_match = re.search(r'<\/em><\/a>(.*)', data, re.DOTALL)
        source = source_match.group(1).strip().replace("\n", "") if source_match else ""
        
        # Add to results if entry is valid
        if title and authors:
            results.append({
                "authors": authors,
                "year": year,
                "link": link,
                "title": title,
                "source": source,
            })
    
    # Create JSON output with metadata
    output = {
        "metadata": {
            "source": "UNP Kediri Repository",
            "subject_code": "462",
            "total_entries": len(results),
            "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "data": results
    }
    
    # Save to JSON file
    with open('repository_data.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return output

# Run scraper
repository_data = scrape_repository()
print("Data saved to repository_data.json")
print(f"Total entries: {repository_data['metadata']['total_entries']}")