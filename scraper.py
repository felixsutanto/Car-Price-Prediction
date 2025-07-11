import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import json
from urllib.parse import urljoin, urlparse
import logging
from typing import Dict, List, Optional, Tuple
import random
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Mobil123Scraper:
    def __init__(self, debug=False):
        self.base_url = "https://www.mobil123.com"
        self.debug = debug
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })
        
        # Create data directory if it doesn't exist
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Combined strategy: Dictionary for normalization + take all approach
        self.brand_model_dict = {
            'toyota': {
                'models': ['avanza', 'xenia', 'rush', 'calya', 'agya', 'yaris', 'vios', 'camry', 
                          'corolla', 'innova', 'fortuner', 'alphard', 'vellfire', 'hiace', 'hilux',
                          'sienta', 'veloz', 'raize'],
                'variants': ['g', 'e', 's', 'trd', 'gr', 'q', 'x', 'v', 'venturer', 'vrz']
            },
            'honda': {
                'models': ['brio', 'jazz', 'city', 'civic', 'accord', 'crv', 'hrv', 'brv', 
                          'mobilio', 'freed', 'odyssey', 'pilot', 'ridgeline'],
                'variants': ['rs', 'e', 's', 'prestige', 'turbo', 'vtec', 'sensing']
            },
            'daihatsu': {
                'models': ['ayla', 'sigra', 'sirion', 'terios', 'gran max', 'luxio', 'taft',
                          'rocky', 'xenia'],
                'variants': ['r', 'x', 'm', 'adventure', 'custom']
            },
            'mitsubishi': {
                'models': ['xpander', 'pajero', 'outlander', 'triton', 'eclipse', 'lancer',
                          'mirage', 'fuso', 'colt'],
                'variants': ['ultimate', 'exceed', 'sport', 'gls', 'glx']
            },
            'nissan': {
                'models': ['march', 'livina', 'juke', 'x-trail', 'serena', 'evalia', 'navara',
                          'terra', 'kicks', 'magnite'],
                'variants': ['sv', 'vl', 'autech', 'nismo']
            },
            'suzuki': {
                'models': ['ertiga', 'baleno', 'swift', 'ignis', 'sx4', 'jimny', 'carry',
                          'apv', 'karimun', 'wagon r', 'xl7'],
                'variants': ['gx', 'gl', 'ga', 'dreza', 'sport']
            },
            'mazda': {
                'models': ['mazda2', 'mazda3', 'mazda6', 'cx-3', 'cx-5', 'cx-9', 'biante',
                          'bt-50', 'mx-5'],
                'variants': ['gt', 'r', 'skyactiv', 'touring']
            },
            'hyundai': {
                'models': ['i10', 'i20', 'avega', 'getz', 'accent', 'elantra', 'sonata',
                          'tucson', 'santa fe', 'h1', 'creta', 'palisade', 'ioniq'],
                'variants': ['gl', 'gls', 'active', 'style', 'prime']
            },
            'kia': {
                'models': ['picanto', 'rio', 'cerato', 'optima', 'sportage', 'sorento',
                          'carnival', 'seltos', 'sonet'],
                'variants': ['lx', 'ex', 'gt']
            },
            'wuling': {
                'models': ['confero', 'cortez', 'almaz', 'air ev', 'bingo', 'hongguang'],
                'variants': ['l', 's', 'c', 'lux', 'exclusive']
            },
            'dfsk': {
                'models': ['supercab', 'gelora', 'glory', 'seres'],
                'variants': ['std', 'lux']
            },
            'chery': {
                'models': ['qq', 'tiggo', 'omoda'],
                'variants': ['pro', 'luxury']
            },
            'mg': {
                'models': ['zs', 'hs', '5', '6'],
                'variants': ['excite', 'inspire', 'trophy']
            },
            'bmw': {
                'models': ['x1', 'x3', 'x5', 'x7', '1 series', '2 series', '3 series', 
                          '5 series', '7 series', 'z4'],
                'variants': ['m', 'xdrive', 'sdrive', 'luxury', 'sport', 'executive']
            },
            'mercedes': {
                'models': ['a-class', 'c-class', 'e-class', 's-class', 'gla', 'glb', 'glc',
                          'gle', 'gls', 'v-class'],
                'variants': ['amg', 'avantgarde', 'elegance', 'exclusive']
            },
            'audi': {
                'models': ['a3', 'a4', 'a6', 'a8', 'q3', 'q5', 'q7', 'q8', 'tt'],
                'variants': ['s-line', 'quattro', 'tfsi']
            }
        }
        
    def normalize_brand_model_variant(self, title: str) -> Tuple[str, str, str, float]:
        """Normalize brand, model, variant, and engine capacity from title"""
        title_lower = title.lower()
        
        # Strategy 1: Use dictionary for known brands/models
        for brand, data in self.brand_model_dict.items():
            if brand in title_lower:
                # Found brand, now look for model
                for model in data['models']:
                    if model in title_lower:
                        # Found model, now look for variant and engine
                        variant = self.extract_variant(title_lower, data['variants'])
                        engine_cc = self.extract_engine_capacity(title_lower)
                        return brand.title(), model.title(), variant, engine_cc
                
                # Brand found but no specific model, try to extract variant and engine
                variant = self.extract_variant(title_lower, data['variants'])
                engine_cc = self.extract_engine_capacity(title_lower)
                return brand.title(), "", variant, engine_cc
        
        # Strategy 2: Fallback - extract any potential brand/model from title
        extracted_brand = self.extract_brand_fallback(title_lower)
        extracted_model = self.extract_model_fallback(title_lower)
        extracted_variant = self.extract_variant_fallback(title_lower)
        extracted_engine = self.extract_engine_capacity(title_lower)
        
        return extracted_brand, extracted_model, extracted_variant, extracted_engine
    
    def extract_engine_capacity(self, title_lower: str) -> float:
        """Extract engine capacity in liters (separate from variant)"""
        # Look for engine displacement patterns
        engine_patterns = [
            r'\b(\d+\.\d+)\s*l\b',  # 1.5L, 2.0L
            r'\b(\d+\.\d+)\s*liter\b',  # 1.5 liter
            r'\b(\d+\.\d+)(?=\s|$|\s[a-z])',  # 1.5, 2.0 (followed by space or end)
            r'\b(\d{4})\s*cc\b',  # 1500cc, 2000cc
        ]
        
        for pattern in engine_patterns:
            match = re.search(pattern, title_lower)
            if match:
                value = float(match.group(1))
                # Convert cc to liters if needed
                if 'cc' in pattern and value > 10:
                    value = value / 1000
                # Validate reasonable engine size (0.5L to 6.0L)
                if 0.5 <= value <= 6.0:
                    return value
        
        return 0.0
    
    def extract_variant(self, title_lower: str, known_variants: List[str]) -> str:
        """Extract variant from title using known variants (trim levels only)"""
        # First try known variants for this brand
        for variant in known_variants:
            if variant in title_lower:
                return variant.upper()
        
        # Look for common variant patterns (trim levels, not engine specs)
        variant_patterns = [
            r'\b(manual|mt|otomatis|at|matic|cvt)\b',  # Transmission
            r'\b(ultimate|exceed|sport|luxury|premium|executive|elegance|comfort)\b',  # Trim levels
            r'\b(rs|trd|gr|gls|glx|lx|ex|sx|dx|sv|vl)\b',  # Specific trims
            r'\b(turbo|vtec|dohc|sohc|tsi|tfsi|hybrid)\b',  # Engine tech
            r'\b[a-z]\s*(mt|at)\b',  # E MT, S AT, etc.
        ]
        
        variants = []
        for pattern in variant_patterns:
            matches = re.findall(pattern, title_lower)
            for match in matches:
                # Skip engine capacity numbers
                if not re.match(r'^\d+\.\d+$', match):
                    variants.append(match)
        
        return ' '.join(variants).upper() if variants else ""
    
    def extract_variant_fallback(self, title_lower: str) -> str:
        """Extract variant using more general patterns (no engine capacity)"""
        # Common variant patterns in Indonesian car market (excluding engine specs)
        patterns = [
            r'\b(manual|mt|otomatis|at|matic|cvt)\b',
            r'\b(rs|gls|glx|lx|ex|sport|luxury|premium|executive|elegance|comfort)\b',
            r'\b(turbo|vtec|dohc|sohc|tsi|tfsi|hybrid)\b',
            r'\b(ultimate|exceed|prestige|adventure|custom|exclusive)\b',
            r'\b[a-z]\s*(mt|at)\b',  # E MT, S AT, etc.
        ]
        
        variants = []
        for pattern in patterns:
            matches = re.findall(pattern, title_lower)
            variants.extend(matches)
        
        return ' '.join(variants).upper() if variants else ""
    
    def extract_brand_fallback(self, title_lower: str) -> str:
        """Extract brand using broader patterns when not in dictionary"""
        # Extended brand list including new/less common brands
        extended_brands = [
            'toyota', 'honda', 'daihatsu', 'mitsubishi', 'nissan', 'suzuki', 'mazda',
            'hyundai', 'kia', 'wuling', 'dfsk', 'chery', 'mg', 'bmw', 'mercedes',
            'audi', 'volkswagen', 'ford', 'chevrolet', 'isuzu', 'peugeot', 'renault',
            'lexus', 'infiniti', 'volvo', 'jaguar', 'land rover', 'porsche', 'ferrari',
            'lamborghini', 'bentley', 'rolls royce', 'maserati', 'alfa romeo',
            'byd', 'changan', 'geely', 'great wall', 'haval', 'dongfeng',
            'tesla', 'lucid', 'rivian', 'nio', 'xpeng', 'li auto'
        ]
        
        for brand in extended_brands:
            if brand in title_lower:
                return brand.title()
        
        # Try to extract from start of title (often brand comes first)
        words = title_lower.split()
        if words and len(words[0]) > 2:
            return words[0].title()
        
        return ""
    
    def extract_model_fallback(self, title_lower: str) -> str:
        """Improved model extraction with blacklist to avoid generic terms"""
        # Avoid these generic or irrelevant tokens
        blacklist = ['dijual', 'mobil', 'bekas', 'jual', 'second', 'baru', 'type', 
                    'tahun', 'km', 'rp', 'manual', 'otomatis', 'matic', 'at', 'mt']
        
        words = title_lower.split()
        
        # Skip first word (likely brand) and look for model in next positions
        for i in range(1, len(words)):
            word = words[i]
            if (
                len(word) >= 2 and 
                not word.isdigit() and
                word not in blacklist and
                not re.match(r'19\d{2}|20\d{2}', word) and  # Not a year
                not re.match(r'^\d+([.,]\d+)?$', word) and  # Not a decimal like 1.5
                not re.match(r'^\d+cc$', word) and  # Not engine capacity
                not word.startswith('rp') and  # Not price
                len(word) <= 15  # Reasonable model name length
            ):
                return word.title()
        
        return ""

    def extract_kilometer_safe(self, element_text: str) -> int:
        """Enhanced kilometer extraction with better patterns and validation"""
        text_lower = element_text.lower()
        
        # Enhanced regex patterns that avoid years and better handle spacing
        km_patterns = [
            r'(\d{1,6}(?:[.,]\d{3})*)\s*(kilo|km|kilometer)(?!\s*tahun|\s*\d{4})',
            r'(odometer|jarak\s*tempuh)\s*[:\-]?\s*(\d{1,6}(?:[.,]\d{3})*)',
            r'(\d{1,6}(?:[.,]\d{3})*)\s*(k?m)\b(?!\s*tahun)',
            r'(?:km|kilometer)[\s:]*(\d{1,6}(?:[.,]\d{3})*)',
            r'(\d{1,6}(?:[.,]\d{3})*)\s*km(?!\s*tahun|\s*\d{4})',
        ]
        
        for pattern in km_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Handle tuple results from group captures
                if isinstance(match, tuple):
                    # Look for the numeric part in the tuple
                    km_str = None
                    for part in match:
                        if re.match(r'\d+', part):
                            km_str = part
                            break
                else:
                    km_str = match
                
                if km_str:
                    # Clean the match
                    km_value = km_str.replace(',', '').replace('.', '')
                    if km_value.isdigit():
                        km_int = int(km_value)
                        
                        # Validate reasonable kilometer values
                        if (1000 <= km_int <= 999999 and  # Reasonable range
                            not (1990 <= km_int <= 2025)):  # Not a year
                            return km_int
        
        return 0

    def debug_page_structure(self, url: str):
        """Debug function to analyze page structure"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            print(f"\n=== DEBUG: Page Structure Analysis ===")
            print(f"URL: {url}")
            print(f"Response Status: {response.status_code}")
            print(f"Content Length: {len(response.content)}")
            
            # Look for various possible selectors
            selectors_to_try = [
                'div[data-testid*="listing"]',
                'div[class*="listing"]',
                'div[class*="card"]',
                'div[class*="item"]',
                'div[class*="vehicle"]',
                'div[class*="car"]',
                'article',
                'li[class*="item"]',
                'div[class*="result"]',
                'div[class*="product"]'
            ]
            
            for selector in selectors_to_try:
                elements = soup.select(selector)
                if elements:
                    print(f"Found {len(elements)} elements with selector: {selector}")
                    if len(elements) > 0:
                        print(f"First element classes: {elements[0].get('class', [])}")
                        print(f"First element text preview: {elements[0].get_text()[:100]}...")
                        
            # Look for price patterns
            price_patterns = [
                r'rp[\s\d.,]+',
                r'price[\s\d.,]+',
                r'\d+[.,]\d+[.,]\d+',
                r'harga'
            ]
            
            text_content = soup.get_text().lower()
            for pattern in price_patterns:
                matches = re.findall(pattern, text_content)
                if matches:
                    print(f"Found price pattern '{pattern}': {matches[:3]}...")
                    
            # Save HTML for manual inspection in data folder
            debug_file_path = os.path.join(self.data_dir, 'debug_page.html')
            with open(debug_file_path, 'w', encoding='utf-8') as f:
                f.write(str(soup.prettify()))
            print(f"Page HTML saved to {debug_file_path} for manual inspection")
            
        except Exception as e:
            print(f"Debug error: {e}")
    
    def get_search_urls(self, max_pages: int = 3) -> List[str]:
        """Generate search URLs for different car categories"""
        urls = []
        
        # Try different URL patterns
        base_patterns = [
            f"{self.base_url}/mobil-dijual",
            f"{self.base_url}/mobil-bekas",
            f"{self.base_url}/jual-mobil",
            f"{self.base_url}/cars-for-sale"
        ]
        
        # Add pagination to the main pattern
        for page in range(1, max_pages + 1):
            urls.append(f"{self.base_url}/mobil-dijual?page={page}")
            
        return urls
    
    def clean_price(self, price_text: str) -> int:
        """Clean and convert price text to integer"""
        if not price_text:
            return 0
            
        # Remove common price prefixes/suffixes
        price_clean = re.sub(r'(rp|rupiah|ribu|juta|miliar)', '', price_text.lower())
        # Keep only digits
        price_clean = re.sub(r'[^\d]', '', price_clean)
        
        if not price_clean:
            return 0
            
        price_int = int(price_clean)
        
        # Handle common abbreviations (juta, ribu)
        if 'juta' in price_text.lower() and price_int < 1000000:
            price_int *= 1000000
        elif 'ribu' in price_text.lower() and price_int < 1000:
            price_int *= 1000
            
        return price_int
    
    def extract_car_data(self, car_element) -> Dict:
        """Extract car data from individual car listing element"""
        data = {
            'judul_iklan': '',
            'merek': '',
            'model': '',
            'varian': '',
            'mesin_cc': 0.0,  # Renamed from engine_cc for local context
            'tahun': 0,
            'harga': 0,
            'harga_text': '',
            'kilometer': 0,
            'lokasi': '',
            'transmisi': '',
            'bahan_bakar': '',
            'warna': '',
            'kondisi': 'Bekas',
            'deskripsi': '',
            'detail_url': ''
        }
        
        try:
            # Extract all text content for analysis
            element_text = car_element.get_text()
            
            # Try multiple approaches to find title
            title_selectors = [
                'h1', 'h2', 'h3', 'h4',
                '[class*="title"]',
                '[class*="name"]',
                '[class*="model"]',
                'a[href*="mobil"]',
                'a[href*="car"]'
            ]
            
            for selector in title_selectors:
                title_elem = car_element.select_one(selector)
                if title_elem and title_elem.get_text(strip=True):
                    data['judul_iklan'] = title_elem.get_text(strip=True)
                    break
            
            # If no title found, try getting from link text
            if not data['judul_iklan']:
                link_elem = car_element.find('a', href=True)
                if link_elem:
                    data['judul_iklan'] = link_elem.get_text(strip=True)
            
            # Use the improved brand/model/variant/engine extraction
            if data['judul_iklan']:
                brand, model, variant, engine_cc = self.normalize_brand_model_variant(data['judul_iklan'])
                data['merek'] = brand
                data['model'] = model
                data['varian'] = variant
                data['mesin_cc'] = engine_cc
            
            # Extract price with multiple patterns
            price_selectors = [
                '[class*="price"]',
                '[class*="harga"]',
                '[data-testid*="price"]'
            ]
            
            for selector in price_selectors:
                price_elem = car_element.select_one(selector)
                if price_elem:
                    price_text = price_elem.get_text(strip=True)
                    data['harga_text'] = price_text
                    data['harga'] = self.clean_price(price_text)
                    break
            
            # If no price found, search in text content
            if not data['harga']:
                price_matches = re.findall(r'rp[\s\d.,]+', element_text.lower())
                if price_matches:
                    data['harga_text'] = price_matches[0]
                    data['harga'] = self.clean_price(price_matches[0])
            
            # Extract year from title or text
            year_match = re.search(r'\b(19|20)\d{2}\b', element_text)
            if year_match:
                data['tahun'] = int(year_match.group())
            
            # Extract kilometer using enhanced safe method
            data['kilometer'] = self.extract_kilometer_safe(element_text)
            
            # Extract location
            location_patterns = [
                r'jakarta|bandung|surabaya|medan|bekasi|tangerang|depok|semarang|palembang|makassar|batam|bogor|pekanbaru|bandar lampung|malang|padang|denpasar|samarinda|tasikmalaya|banjarmasin|pontianak|balikpapan|jambi|cimahi|surakarta|manado|yogyakarta|ambon|serang|mataram|kendari|sorong|ternate|bengkulu|palu|jayapura|kupang|banda aceh|gorontalo|mamuju'
            ]
            
            text_lower = element_text.lower()
            for pattern in location_patterns:
                location_match = re.search(pattern, text_lower)
                if location_match:
                    data['lokasi'] = location_match.group().title()
                    break
            
            # Extract transmission
            if re.search(r'\b(manual|mt)\b', text_lower):
                data['transmisi'] = 'Manual'
            elif re.search(r'\b(otomatis|automatic|at|matic|cvt)\b', text_lower):
                data['transmisi'] = 'Otomatis'
            
            # Extract fuel type
            if re.search(r'\bbensin\b', text_lower):
                data['bahan_bakar'] = 'Bensin'
            elif re.search(r'\bdiesel\b', text_lower):
                data['bahan_bakar'] = 'Diesel'
            elif re.search(r'\bhybrid\b', text_lower):
                data['bahan_bakar'] = 'Hybrid'
            elif re.search(r'\blistrik\b', text_lower):
                data['bahan_bakar'] = 'Listrik'
            
            # Extract color
            colors = ['putih', 'hitam', 'silver', 'merah', 'biru', 'abu', 'kuning', 'hijau', 'coklat', 'gold', 'emas']
            for color in colors:
                if color in text_lower:
                    data['warna'] = color.title()
                    break
            
            # Extract link
            link_elem = car_element.find('a', href=True)
            if link_elem:
                href = link_elem.get('href')
                if href:
                    data['detail_url'] = urljoin(self.base_url, href)
            
            # Extract description (limit to reasonable length)
            data['deskripsi'] = element_text[:300] if element_text else ""
            
        except Exception as e:
            if self.debug:
                logger.error(f"Error extracting car data: {e}")
        
        return data
    
    def scrape_listings_page(self, url: str) -> List[Dict]:
        """Scrape car listings from a single page"""
        try:
            logger.info(f"Scraping: {url}")
            
            if self.debug:
                self.debug_page_structure(url)
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try multiple selector strategies
            car_elements = []
            
            # Strategy 1: Look for common car listing patterns
            selectors_to_try = [
                'div[data-testid*="listing"]',
                'div[class*="listing"]',
                'div[class*="card"]',
                'div[class*="item"]',
                'article',
                'li[class*="item"]',
                'div[class*="result"]',
                'div[class*="vehicle"]',
                'div[class*="car"]',
                'div[class*="product"]'
            ]
            
            for selector in selectors_to_try:
                elements = soup.select(selector)
                if elements:
                    # Filter elements that likely contain car data
                    for elem in elements:
                        elem_text = elem.get_text().lower()
                        if any(keyword in elem_text for keyword in ['rp', 'price', 'harga', 'juta', 'ribu']) and \
                           any(keyword in elem_text for keyword in ['toyota', 'honda', 'mobil', 'car', 'tahun']) and \
                           len(elem_text) > 50:  # Ensure it's not just a small snippet
                            car_elements.append(elem)
                    
                    if car_elements:
                        logger.info(f"Found {len(car_elements)} car elements using selector: {selector}")
                        break
            
            # Strategy 2: If no elements found, try broader approach
            if not car_elements:
                all_divs = soup.find_all('div')
                for div in all_divs:
                    div_text = div.get_text().lower()
                    if (len(div_text) > 100 and 
                        any(price_word in div_text for price_word in ['rp', 'harga', 'price', 'juta']) and
                        any(car_word in div_text for car_word in ['toyota', 'honda', 'mobil', 'tahun', 'km'])):
                        car_elements.append(div)
                
                if car_elements:
                    logger.info(f"Found {len(car_elements)} potential car elements using broad search")
            
            # Extract data from found elements
            cars_data = []
            for car_element in car_elements:
                car_data = self.extract_car_data(car_element)
                
                # Only add if we have minimum required data
                if (car_data['judul_iklan'] and 
                    (car_data['harga'] > 0 or car_data['harga_text']) and
                    (car_data['merek'] or car_data['tahun'] > 0)):
                    cars_data.append(car_data)
            
            logger.info(f"Successfully extracted {len(cars_data)} car listings")
            return cars_data
            
        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return []
    
    def scrape_all_pages(self, max_pages: int = 3, delay: float = 2.0) -> List[Dict]:
        """Scrape multiple pages of car listings"""
        all_cars = []
        urls = self.get_search_urls(max_pages)
        
        for i, url in enumerate(urls):
            try:
                cars_data = self.scrape_listings_page(url)
                all_cars.extend(cars_data)
                
                # Add delay between requests
                if i < len(urls) - 1:
                    time.sleep(delay + random.uniform(0.5, 1.5))
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                continue
        
        logger.info(f"Total cars scraped: {len(all_cars)}")
        return all_cars
    
    def save_to_csv(self, data: List[Dict], filename: str = "mobil123_data.csv"):
        """Save scraped data to CSV file in data folder"""
        if not data:
            logger.warning("No data to save")
            return None
        
        # Create full path with data directory
        file_path = os.path.join(self.data_dir, filename)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding='utf-8')
        logger.info(f"Data saved to {file_path}")
        
        # Display basic statistics
        print(f"\n=== SCRAPING RESULTS ===")
        print(f"Total records: {len(df)}")
        print(f"Records with prices: {len(df[df['harga'] > 0])}")
        print(f"Unique brands: {df['merek'].nunique()}")
        print(f"Records with variants: {len(df[df['varian'] != ''])}")
        print(f"Records with engine capacity: {len(df[df['mesin_cc'] > 0])}")
        if len(df[df['harga'] > 0]) > 0:
            print(f"Price range: Rp {df[df['harga'] > 0]['harga'].min():,} - Rp {df[df['harga'] > 0]['harga'].max():,}")
        if len(df[df['tahun'] > 0]) > 0:
            print(f"Year range: {df[df['tahun'] > 0]['tahun'].min()} - {df[df['tahun'] > 0]['tahun'].max()}")
        if len(df[df['kilometer'] > 0]) > 0:
            print(f"Kilometer range: {df[df['kilometer'] > 0]['kilometer'].min():,} - {df[df['kilometer'] > 0]['kilometer'].max():,} km")
        
        return df

def main():
    """Main function to run the scraper"""
    # Enable debug mode for first run
    scraper = Mobil123Scraper(debug=True)
    
    print("Starting Mobil123 scraper with improved model and kilometer extraction...")
    cars_data = scraper.scrape_all_pages(max_pages=500, delay=3.0)
    
    if cars_data:
        # Save to CSV
        df = scraper.save_to_csv(cars_data)
        
        if df is not None:
            # Display sample data
            print(f"\n=== SAMPLE DATA ===")
            print(df.head())
            
            print(f"\n=== DATA QUALITY CHECK ===")
            print(f"Records with title: {len(df[df['judul_iklan'] != ''])}")
            print(f"Records with price: {len(df[df['harga'] > 0])}")
            print(f"Records with brand: {len(df[df['merek'] != ''])}")
            print(f"Records with model: {len(df[df['model'] != ''])}")
            print(f"Records with variant: {len(df[df['varian'] != ''])}")
            print(f"Records with engine capacity: {len(df[df['mesin_cc'] > 0])}")
            print(f"Records with year: {len(df[df['tahun'] > 0])}")
            print(f"Records with kilometer: {len(df[df['kilometer'] > 0])}")
            
            # Show brand distribution
            print(f"\n=== BRAND DISTRIBUTION ===")
            print(df['merek'].value_counts().head(10))
            
            # Show model distribution 
            print(f"\n=== MODEL DISTRIBUTION ===")
            model_df = df[df['model'] != '']
            if len(model_df) > 0:
                print(model_df['model'].value_counts().head(10))
            else:
                print("No models detected")
            
            # Show variant distribution
            print(f"\n=== VARIANT DISTRIBUTION ===")
            variant_df = df[df['varian'] != '']
            if len(variant_df) > 0:
                print(variant_df['varian'].value_counts().head(10))
            else:
                print("No variants detected")
            
            # Show engine capacity distribution
            print(f"\n=== ENGINE CAPACITY DISTRIBUTION ===")
            engine_df = df[df['mesin_cc'] > 0]
            if len(engine_df) > 0:
                print(engine_df['mesin_cc'].value_counts().head(10))
            else:
                print("No engine capacities detected")
        
    else:
        print("No data was scraped. Please check the debug_page.html file in the data folder for manual analysis.")

if __name__ == "__main__":
    main()