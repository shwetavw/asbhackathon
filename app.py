import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from supabase import create_client, Client
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import os
import re
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure Gemini Pro
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize Supabase with error handling
try:
    supabase: Client = create_client(
        supabase_url=os.getenv('SUPABASE_URL'),
        supabase_key=os.getenv('SUPABASE_KEY')
    )
except Exception as e:
    print(f"Supabase initialization error: {str(e)}")
    raise


# Constants
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
HEADERS = {'User-Agent': USER_AGENT}

# Add error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# Rate limiting storage
rate_limits = defaultdict(lambda: {'count': 0, 'reset_time': datetime.now()})

def check_rate_limit(url):
    """Check if domain has exceeded rate limit"""
    domain = urlparse(url).netloc
    now = datetime.now()
    
    # Reset counter if time window has passed
    if now - rate_limits[domain]['reset_time'] > timedelta(minutes=1):
        rate_limits[domain] = {'count': 0, 'reset_time': now}
    
    # Check and increment counter
    rate_limits[domain]['count'] += 1
    return rate_limits[domain]['count'] <= 60  # 60 requests per minute per domain

def check_site_permissions(url):
    """Check all site permissions and requirements before scraping"""
    try:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # 1. Check robots.txt
        robots_allowed = check_robots_permission(url)
        if not robots_allowed:
            return False, "Blocked by robots.txt"

        # 2. Check rate limiting headers
        try:
            head_response = requests.head(url, headers=HEADERS, timeout=5)
            
            # Check for rate limit headers
            if 'X-RateLimit-Remaining' in head_response.headers:
                if int(head_response.headers['X-RateLimit-Remaining']) <= 0:
                    return False, "Rate limit exceeded"

            # 3. Check response status
            if head_response.status_code == 403:
                return False, "Access forbidden"
            elif head_response.status_code == 429:
                return False, "Too many requests"
            elif head_response.status_code != 200:
                return False, f"HTTP error: {head_response.status_code}"

            # 4. Check content type
            content_type = head_response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                return False, f"Unsupported content type: {content_type}"

            # 5. Check terms of service URL
            tos_paths = ['/terms', '/tos', '/terms-of-service', '/terms-and-conditions']
            for path in tos_paths:
                tos_url = base_url + path
                try:
                    tos_response = requests.head(tos_url, headers=HEADERS, timeout=3)
                    if tos_response.status_code == 200:
                        return True, "Warning: Please review Terms of Service at " + tos_url
                except:
                    continue

            return True, "All permission checks passed"

        except requests.exceptions.RequestException as e:
            return False, f"Network error during permission check: {str(e)}"

    except Exception as e:
        return False, f"Permission check error: {str(e)}"

def check_robots_permission(url):
    """Check if scraping is allowed by robots.txt"""
    try:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = f"{base_url}/robots.txt"
        
        # Fetch robots.txt
        rp = RobotFileParser()
        rp.set_url(robots_url)
        
        # Try to read robots.txt with timeout
        try:
            response = requests.get(robots_url, headers=HEADERS, timeout=5)
            if response.status_code == 200:
                rp.parse(response.text.splitlines())
            else:
                return True  # No robots.txt, assume allowed
        except:
            return True  # If robots.txt can't be fetched, assume allowed
        
        # Check permission for our user agent
        return rp.can_fetch(USER_AGENT, url)
    
    except Exception as e:
        print(f"Robots.txt check error: {str(e)}")
        return True  # Fail-safe: assume allowed if check fails

def extract_text_from_url(url):
    """Fetch website content and extract clean text"""
    try:
        # Check robots.txt permission
        if not check_robots_permission(url):
            return None, "Scraping disallowed by robots.txt"
        
        # Fetch page content
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            return None, f"Unsupported content type: {content_type}"
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unnecessary elements but keep main content
        for element in soup.find_all(['script', 'style', 'iframe', 'noscript']):
            element.decompose()
        
        # Try different methods to get content
        main_content = []
        
        # Method 1: Try to find content by common content IDs/classes
        content_selectors = [
            '#content', '#main-content', '.content', '.main-content',
            'article', 'main', '[role="main"]', '.post-content',
            '.entry-content', '.article-content'
        ]
        
        for selector in content_selectors:
            content = soup.select(selector)
            if content:
                for element in content:
                    text = element.get_text(separator=' ', strip=True)
                    if text:
                        main_content.append(text)
        
        # Method 2: If no content found, try semantic elements
        if not main_content:
            semantic_tags = soup.find_all(['article', 'main', 'section'])
            for tag in semantic_tags:
                text = tag.get_text(separator=' ', strip=True)
                if text and len(text) > 100:  # Only include substantial content
                    main_content.append(text)
        
        # Method 3: Look for paragraphs with substantial content
        if not main_content:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 50:  # Filter out short paragraphs
                    main_content.append(text)
        
        # Method 4: Last resort - get all divs with substantial text
        if not main_content:
            divs = soup.find_all('div')
            for div in divs:
                text = div.get_text(strip=True)
                if text and len(text) > 100:
                    main_content.append(text)
        
        # If still no content, get all text as last resort
        if not main_content:
            text = soup.get_text(separator=' ', strip=True)
            if text:
                main_content.append(text)
        
        # Combine and clean the text
        if main_content:
            text = ' '.join(main_content)
            # Clean the text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
            text = re.sub(r'\n+', ' ', text)  # Replace newlines
            text = text.strip()
            
            if len(text) > 100:  # Ensure we have substantial content
                return text[:12000], None
        
        return None, "No substantial content found on page"
    
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Processing error: {str(e)}"

def extract_fields_with_gemini(website_text, url):
    """Use Gemini to extract structured data from website text"""
    prompt = f"""
    Extract the following details from the website content below. Return ONLY valid JSON format:
    
    Required Fields:
    - name: (string) Official company name
    - slug: (string) URL-friendly version of the company name
    - entity_type: (string) 'social_enterprise', 'investor', 'ecosystem_builder'
    - website: (string) Main website URL
    - description: (string) Brief description of the company
    - hq_location: (string) location
    - contact_email: (string) Contact email address
    - industry-sector: (string) Primary industry or sector, required if entity_type is 'social_enterprise'
    - social_status: (string) 'Yes', 'No', or 'Unknown', required if entity_type is 'social_enterprise'
    - funding_stage: (string) 'Growth', 'Pre-seed', 'Seed', 'Series A', etc., required if entity_type is 'social_enterprise'
    - cheque_size_range: (string) Range of investment amounts, required if enerty_type is 'investor'
    - investment_thesis: (string) Brief description of investment focus, required if entity_type is 'investor'
    - program_type: (string) 'Accelerator', 'Incubator', 'Grant', etc., required if entity_type is 'ecosystem_builder'
    - next_intake_date: (string) Next application deadline or intake date, required if entity_type is 'ecosystem_builder'
    - impact: (string) Brief description of social/environmental impact, required if entity_type is 'social_enterprise'
    - problem_solved: (string) Description of the problem being addressed, required if entity_type is 'social_enterprise'
    - target_beneficiaries: (string) Who benefits from the company's work, required if entity_type is 'social_enterprise'
    - revenue_model: (string) How the company generates revenue, required if entity_type is 'social_enterprise'
    - year_founded: (string) Year the company was founded, required if entity_type is 'social_enterprise'
    - awards: (string) Any awards or recognitions received, required if entity_type is 'social_enterprise'
    - grants: (string) Any grants received, required if entity_type is 'social_enterprise'
    - institutional_support: (string) Any institutional support received, required if entity_type is 'social_enterprise'
    
    Important:
    - Use "Unknown" for missing information
    - Keep description concise (1-2 sentences)
    - For website, use "{url}" if not found in content
    
    Example Output:
    {{
        "name": "Tech Innovations Inc.",
        "slug": "tech-innovations-inc",
        "entity_type": "social_enterprise",
        "website": "https://www.techinnovations.example.com",
        "description": "Leading provider of AI solutions for businesses",
        "hq_location": "kuala lumpur, malaysia",
        "contact_email": "hello@pichaeats.example.com",
        "industry_sector": "Waste Management, Environmental Services",
        "social_status": "Yes",
        "funding_stage": "Growth",
        "cheque_size_range": "Pre-seed",
        "investment_thesis": "Investing in AI startups",
        "program_type": "Accelerator",
        "next_intake_date": "2024-06-01",
        "impact": "Improving business efficiency",
        "problem_solved": "Businesses struggle with data analysis",
        "target_beneficiaries": "Small to medium enterprises",
        "revenue_model": "Subscription-based",
        "year_founded": "2015",
        "awards": "Best AI Startup 2023",
        "grants": "Received $500,000 grant from TechFund",
        "institutional_support": "Supported by AI Research Institute"
    }}
    
    Website Content:
    {website_text}
    """
    
    try:
        response = model.generate_content(prompt)
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            return json_match.group(0), None
        return None, "No valid JSON found in AI response"
    except Exception as e:
        return None, f"Gemini processing error: {str(e)}"

def clean_contact_info(contact_info):
    """Clean contact info while preserving email addresses"""
    if not contact_info:
        return "Unknown"
    
    try:
        # Check for email addresses
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, contact_info)
        
        if not emails:
            # No emails found, just clean basic formatting
            contact_info = contact_info.replace('\\n', ' ')
            contact_info = re.sub(r'\s+', ' ', contact_info)
            return contact_info.strip()
        
        # If emails exist, preserve them while cleaning
        # Clean the text
        contact_info = contact_info.replace('\\n', ' ')
        contact_info = re.sub(r'\s+', ' ', contact_info)
        
        # Replace any sanitized email addresses with original ones
        for email in emails:
            sanitized_email = email.replace('@', '[at]')
            contact_info = contact_info.replace(sanitized_email, email)
            # Also try replacing HTML encoded version
            html_encoded = email.replace('@', '&#64;')
            contact_info = contact_info.replace(html_encoded, email)
        
        return contact_info.strip()
    except Exception as e:
        print(f"Error cleaning contact info: {str(e)}")
        return contact_info  # Return original if cleaning fails

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/scrape', methods=['POST'])
def scrape_website():
    """API endpoint for website scraping and data extraction"""
    start_time = time.time()
    
    try:
        # Get URL from request
        data = request.get_json()
        if not data or 'url' not in data:
            raise ValueError("Missing URL parameter")
            
        url = data['url'].strip()
        if not url:
            raise ValueError("Empty URL provided")
            
        # Rate limiting check
        if not check_rate_limit(url):
            raise Exception("Rate limit exceeded for this domain")            
    
        # Check if URL already exists in database
        existing_record = supabase.table('entities').select('id').eq('website', url).execute()
        
        # Step 1: Extract website text
        website_text, error = extract_text_from_url(url)
        
        if error:
            raise ValueError(str(error))
            # return jsonify({
            #     "status": "error",
            #     "message": error,
            #     "url": url,
            #     "execution_time": round(time.time() - start_time, 2)
            # }), 403 if "robots.txt" in error else 500
        
        # Step 2: Extract fields with Gemini
        gemini_response, error = extract_fields_with_gemini(website_text, url)
        
        if error:
            raise Exception(str(error))  
            # return jsonify({
            #     "status": "error",
            #     "message": error,
            #     "url": url,
            #     "execution_time": round(time.time() - start_time, 2)
            # }), 500
        
        # Parse and validate Gemini response
        extracted_data = json.loads(gemini_response)
        
        # Clean contact info
        extracted_data['contact_email'] = clean_contact_info(extracted_data.get('contact_email'))
        
        # Add metadata
        extracted_data['updated_at'] = 'now()'
        extracted_data['website'] = url
        
        if existing_record.data and len(existing_record.data) > 0:
            # Update existing record
            record_id = existing_record.data[0]['id']
            result = supabase.table('entities').update(extracted_data).eq('id', record_id).execute()
            operation = "updated"
        else:
            # Insert new record
            extracted_data['created_at'] = 'now()'
            result = supabase.table('entities').insert(extracted_data).execute()
            operation = "created"
            
        if len(result.data) > 0:
            # Prepare API response
            response_data = extracted_data.copy()
            response_data['id'] = result.data[0]['id']
            
            return jsonify({
                "status": "success",
                "message": f"Data {operation} successfully",
                "data": response_data,
                "execution_time": round(time.time() - start_time, 2)
            })
        else:
            raise Exception(f"Failed to {operation} data in Supabase")
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "execution_time": round(time.time() - start_time, 2)
        }), 400
    except Exception as e:
        app.logger.error(f"Scraping error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Data processing failed: {str(e)}",
            #"url": url,
            #"ai_response": gemini_response[:500] + "..." if gemini_response else None,
            "execution_time": round(time.time() - start_time, 2)
        }), 500


@app.route('/check-permission', methods=['POST'])
def check_permission():
    """Check if scraping is allowed for a URL"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing URL parameter"}), 400
    
    url = data['url'].strip()
    allowed, message = check_site_permissions(url)
    
    return jsonify({
        "url": url,
        "scraping_allowed": allowed,
        "message": message
    })

if __name__ == '__main__':
    # Production configurations
    if os.getenv('FLASK_ENV') == 'production':
        from waitress import serve
        app.logger.info('Starting production server...')
        serve(app, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
    else:
        # Development configurations
        app.logger.info('Starting development server...')
        app.run(
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            debug=True
        )