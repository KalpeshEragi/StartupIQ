"""Sectors and the Wikipedia articles whose daily page views make up each sector's market interest index.

Each sector mixes topic articles (what people look up when interested in the category) with the main Indian brands
in it. Titles are exact Wikipedia article titles; any that do not exist are skipped by the downloader and reported.
Projects: "en.wikipedia" (English) and "hi.wikipedia" (Hindi).
"""
from __future__ import annotations

START = "2023-01-01"  # data window starts here, by design (post-COVID, UPI and AI-era market only)

SECTORS: dict[str, dict] = {
    "food_beverage": {
        "label": "Food and beverage brands (D2C)",
        "en.wikipedia": ["Snack", "Namkeen", "Potato chips", "Energy drink", "Breakfast cereal", "Millet", "Protein bar",
                         "Haldiram's", "Paper Boat", "Amul"],
        "hi.wikipedia": ["हल्दीराम"],
    },
    "fashion_apparel": {
        "label": "Fashion and apparel",
        "en.wikipedia": ["Sari", "Kurta", "Sneakers", "Fast fashion", "Streetwear", "Myntra", "Meesho",
                         "Fabindia", "Raymond Group"],
        "hi.wikipedia": ["साड़ी", "कुर्ता"],
    },
    "beauty_personal_care": {
        "label": "Beauty and personal care",
        "en.wikipedia": ["Cosmetics", "Skin care", "Sunscreen", "Perfume", "Lipstick", "Hair care", "Nykaa",
                         "Lakmé Cosmetics"],
        "hi.wikipedia": ["सौन्दर्य प्रसाधन"],
    },
    "quick_commerce_grocery": {
        "label": "Grocery and quick commerce",
        "en.wikipedia": ["Q-commerce", "Online grocer", "Blinkit", "Zepto (company)", "BigBasket",
                         "JioMart", "DMart"],
        "hi.wikipedia": ["ब्लिंकिट"],
    },
    "food_delivery_restaurants": {
        "label": "Food delivery and restaurants",
        "en.wikipedia": ["Online food ordering", "Virtual restaurant", "Swiggy", "Zomato", "Biryani", "Rebel Foods",
                         "Domino's"],
        "hi.wikipedia": ["बिरयानी"],
    },
    "edtech": {
        "label": "Education technology",
        "en.wikipedia": ["Educational technology", "Online learning", "Joint Entrance Examination",
                         "National Eligibility cum Entrance Test (Undergraduate)", "Union Public Service Commission",
                         "Byju's", "Physics Wallah", "Unacademy", "Vedantu"],
        "hi.wikipedia": ["संघ लोक सेवा आयोग"],
    },
    "fintech_payments": {
        "label": "Fintech and payments",
        "en.wikipedia": ["Unified Payments Interface", "E-commerce payment system", "Buy now, pay later",
                         "Mobile payment", "Credit card", "PhonePe", "Paytm", "Google Pay", "BharatPe"],
        "hi.wikipedia": ["एकीकृत भुगतान अन्तरापृष्ठ", "पेटीऍम"],
    },
    "investing_wealth": {
        "label": "Investing and wealth",
        "en.wikipedia": ["Mutual fund", "Systematic investment plan", "Initial public offering", "NIFTY 50",
                         "BSE SENSEX", "Stock market", "Cryptocurrency", "Zerodha"],
        "hi.wikipedia": ["म्यूचुअल फंड", "बीएसई सेंसेक्स"],
    },
    "saas_software": {
        "label": "SaaS and business software",
        "en.wikipedia": ["Software as a service", "Customer relationship management", "Enterprise resource planning",
                         "Accounting software", "Zoho Corporation", "Freshworks", "Tally Solutions", "Salesforce"],
        "hi.wikipedia": [],
    },
    "ai_tools": {
        "label": "AI tools and products",
        "en.wikipedia": ["Generative AI", "Large language model", "ChatGPT", "Google Gemini", "Claude (AI)",
                         "Perplexity AI", "Microsoft Copilot", "AI agent"],
        "hi.wikipedia": ["कृत्रिम बुद्धि", "चैटजीपीटी"],
    },
    "health_wellness": {
        "label": "Health and wellness",
        "en.wikipedia": ["Telehealth", "Online pharmacy", "Fitness tracker", "Dietary supplement", "Ayurveda", "Yoga",
                         "PharmEasy", "Tata 1mg", "Cult.fit", "HealthifyMe"],
        "hi.wikipedia": ["आयुर्वेद", "योग"],
    },
    "travel_hospitality": {
        "label": "Travel and hospitality",
        "en.wikipedia": ["Tourism in India", "Homestay", "Travel agency", "MakeMyTrip", "Oyo Rooms", "IndiGo",
                         "Indian Railway Catering and Tourism Corporation", "Ixigo"],
        "hi.wikipedia": ["भारत में पर्यटन"],
    },
    "mobility_ev": {
        "label": "Mobility and electric vehicles",
        "en.wikipedia": ["Electric vehicle", "Electric scooter", "Ridesharing company", "Ola Electric", "Ather Energy",
                         "Tata Nexon", "Rapido (company)", "Uber"],
        "hi.wikipedia": ["विद्युत वाहन"],
    },
    "gaming_esports": {
        "label": "Gaming and esports",
        "en.wikipedia": ["Esports", "Mobile game", "Battlegrounds Mobile India", "Free Fire (video game)",
                         "Dream11", "Fantasy sport", "Ludo King", "Nazara Technologies"],
        "hi.wikipedia": ["ड्रीम11"],
    },
    "media_entertainment": {
        "label": "Media, OTT and creators",
        "en.wikipedia": ["Over-the-top media service", "Web series", "Podcast", "Influencer marketing",
                         "Disney+ Hotstar", "Netflix", "Amazon Prime Video", "Pocket FM"],
        "hi.wikipedia": [],
    },
    "real_estate_home": {
        "label": "Real estate and home",
        "en.wikipedia": ["Housing in India", "Coworking", "Interior design", "Mortgage", "NoBroker", "Livspace",
                         "Urban Company", "Pepperfry"],
        "hi.wikipedia": [],
    },
    "jobs_hr": {
        "label": "Jobs, HR and gig work",
        "en.wikipedia": ["Gig worker", "Remote work", "Layoff", "Recruitment", "Naukri.com", "LinkedIn",
                         "Internshala"],
        "hi.wikipedia": ["आजीविका"],
    },
    "logistics_supply": {
        "label": "Logistics and B2B supply",
        "en.wikipedia": ["Last mile (transportation)", "Third-party logistics", "Business-to-business", "Warehouse",
                         "Supply chain", "Delhivery", "IndiaMART", "Open Network for Digital Commerce"],
        "hi.wikipedia": [],
    },
    "agritech": {
        "label": "Agriculture and agritech",
        "en.wikipedia": ["Agriculture in India", "Organic farming", "Minimum support price (India)", "Kharif crop",
                         "Rabi crop", "Agricultural technology", "Drip irrigation", "Fertilizer"],
        "hi.wikipedia": ["भारत में कृषि", "जैविक खेती"],
    },
}


def articles() -> list[tuple[str, str, str]]:
    """Every (sector, project, title) to download."""
    return [(sector, project, title) for sector, spec in SECTORS.items()
            for project in ("en.wikipedia", "hi.wikipedia") for title in spec.get(project, [])]
