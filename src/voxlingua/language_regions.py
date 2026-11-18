"""Curated illustrative language-region labels, never speaker geolocation."""

REGIONS={
    'English':('English-speaking world',51.5,-.1,'English'),
    'Hindi':('Northern India',26.8,80.9,'हिन्दी'),
    'Tamil':('Tamil Nadu',11.1,78.7,'தமிழ்'),
    'Bengali':('Bengal region',23.5,89.1,'বাংলা'),
    'Telugu':('Andhra Pradesh & Telangana',16.5,80.6,'తెలుగు'),
    'Kannada':('Karnataka',14.5,76.0,'ಕನ್ನಡ'),
    'Malayalam':('Kerala',10.8,76.3,'മലയാളം'),
    'Marathi':('Maharashtra',19.7,75.7,'मराठी'),
    'Gujarati':('Gujarat',22.3,71.2,'ગુજરાતી'),
    'Punjabi':('Punjab region',31.1,74.7,'ਪੰਜਾਬੀ'),
    'Urdu':('South Asia',30.4,73.0,'اردو'),
    'Odia':('Odisha',20.9,85.1,'ଓଡ଼ିଆ'),
    'Assamese':('Assam',26.2,92.9,'অসমীয়া'),
    'Nepali':('Nepal & Himalayan region',28.0,84.1,'नेपाली'),
    'Bhojpuri':('Eastern Uttar Pradesh & Bihar',25.5,84.0,'भोजपुरी'),
    'Maithili':('Mithila region',26.2,86.0,'मैथिली'),
}


def language_region(label: str) -> dict | None:
    """Return a representative association only for a curated known model label."""
    value=REGIONS.get(label)
    if value is None: return None
    name,lat,lon,native=value
    return {'name':name,'lat':lat,'lon':lon,'native':native,'meaning':'Representative language region, not the speaker location or origin'}
