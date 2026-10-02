"""
core/sample_queries.py
----------------------
Six natural-language demo queries drawn from real listing content in the
benchmark collection.  Each covers a distinct intent and property profile
so the Model Comparison and Quantization pages have varied, meaningful
examples to demo against.
"""

SAMPLE_QUERIES: list[dict] = [
    {
        "label": "Beachside retreat",
        "query": (
            "A cosy, well-equipped place right by the beach with easy access to "
            "cafes and shops, ideally with good public transport nearby for day trips"
        ),
        "why": "Matches coastal listings in Sydney/Rio/Hawaii — tests neighbourhood + transit fields",
    },
    {
        "label": "City-centre remote work",
        "query": (
            "Quiet apartment in the heart of the city, great wifi, a proper desk, "
            "and close to metro or subway so I can get around without a car"
        ),
        "why": "Targets urban apartments in NYC/Barcelona/Montreal/Istanbul — tests transit + space fields",
    },
    {
        "label": "Family beach holiday",
        "query": (
            "Spacious family home sleeping six or more, private pool or garden, "
            "near the beach, kid-friendly with parking and a full kitchen"
        ),
        "why": "Pulls larger villas/houses in Rio/Hawaii/Sydney — tests bedrooms + amenities in listing_text",
    },
    {
        "label": "Boutique harbour views",
        "query": (
            "Luxurious property with stunning water or harbour views, ideally a "
            "unique space like a villa or boat, perfect for a special occasion"
        ),
        "why": "Surfaces premium/unusual listings — Villa on Bosphorus, Sydney Harbour house, Porto boat",
    },
    {
        "label": "Budget solo traveller",
        "query": (
            "Affordable private room in a lively neighbourhood, friendly host, "
            "walking distance to bars and restaurants, good for solo travel"
        ),
        "why": "Targets private-room listings in Barcelona/Porto/Hong Kong — tests notes + interaction fields",
    },
    {
        "label": "Peaceful rural escape",
        "query": (
            "Charming countryside cottage or rural retreat, surrounded by nature, "
            "very quiet and peaceful, somewhere to completely disconnect and relax"
        ),
        "why": "Draws out farm stays, chalets, garden flats — tests neighborhood_overview + description",
    },
]
