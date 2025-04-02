from flask import Flask, render_template, request, jsonify, send_file
import requests
import os
from dotenv import load_dotenv
import logging
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# HuggingFace API configuration
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
API_URL = "https://api-inference.huggingface.co/models"

# Log the API key status (but not the actual key)
logger.info(f"HuggingFace API Key configured: {'Yes' if HUGGINGFACE_API_KEY else 'No'}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_profile():
    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        if not HUGGINGFACE_API_KEY:
            logger.error("HuggingFace API key not found in environment variables")
            return jsonify({'error': 'API key not configured'}), 500
        
        # Combine all text fields for analysis with better formatting
        combined_text = f"""
        Headline: {data['headline']}
        Summary: {data['summary']}
        Experience: {data['experience']}
        Skills: {data['skills']}
        Education: {data['education']}
        """
        
        # Text summarization
        try:
            summary_response = analyze_text(combined_text, "facebook/bart-large-cnn")
            summary = summary_response[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            summary = "Unable to generate summary"
        
        # Keyword extraction with multiple models
        keywords = set()
        try:
            # Try multiple keyword extraction models
            models = [
                "yanekyuk/bert-uncased-keyword-extractor",
                "mrm8488/bert-tiny2-finetuned-keyword-extraction",
                "yanekyuk/bert-uncased-keyword-extractor"
            ]
            
            # Extract keywords from each section separately
            sections = {
                'headline': data['headline'],
                'summary': data['summary'],
                'experience': data['experience'],
                'skills': data['skills'],
                'education': data['education']
            }
            
            for section_name, section_text in sections.items():
                for model in models:
                    try:
                        response = analyze_text(section_text, model)
                        if isinstance(response, list) and len(response) > 0:
                            if isinstance(response[0], dict) and 'word' in response[0]:
                                keywords.update(k['word'] for k in response[0])
                            elif isinstance(response[0], str):
                                keywords.add(response[0])
                    except Exception as e:
                        logger.error(f"Keyword extraction failed for model {model} in section {section_name}: {str(e)}")
                        continue
            
            # Add some common professional keywords if none were found
            if not keywords:
                common_keywords = {
                    "leadership", "management", "project", "team", "communication",
                    "technical", "development", "analytics", "strategy", "innovation",
                    "problem-solving", "collaboration", "planning", "execution"
                }
                keywords.update(common_keywords)
            
            keywords = list(keywords)
        except Exception as e:
            logger.error(f"All keyword extraction attempts failed: {str(e)}")
            keywords = ["leadership", "management", "project", "team", "communication"]
        
        # Sentiment analysis
        try:
            sentiment_response = analyze_text(combined_text, "nlptown/bert-base-multilingual-uncased-sentiment")
            sentiment_score = float(sentiment_response[0]['label'].split()[0])
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            sentiment_score = 3.0  # Neutral sentiment as fallback
        
        # Generate strengths and improvements first
        strengths, improvements = analyze_strengths_and_improvements(data, sentiment_score)
        
        # Prepare analysis results for score calculation
        analysis_results = {
            **data,  # Include all profile data
            'strengths': strengths,
            'improvements': improvements
        }
        
        # Calculate profile score
        score = calculate_profile_score(analysis_results)
        
        # Generate detailed suggestions
        suggestions = generate_suggestions(data, keywords)
        
        return jsonify({
            'score': score,
            'summary': summary,
            'keywords': keywords,
            'strengths': strengths,
            'improvements': improvements,
            'suggestions': suggestions
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_profile: {str(e)}")
        return jsonify({'error': str(e)}), 500

def calculate_profile_score(analysis_results):
    """Calculate a profile score based on analysis results"""
    try:
        # Initialize base score
        score = 0
        max_score = 100
        
        # Define minimum content requirements for each section
        min_requirements = {
            'headline': {'length': 10, 'points': 20},
            'summary': {'length': 200, 'points': 25},
            'experience': {'length': 100, 'points': 25},
            'skills': {'length': 50, 'points': 20},
            'education': {'length': 50, 'points': 10}
        }
        
        # Check each section's content quality
        for section, requirements in min_requirements.items():
            content = analysis_results.get(section, '')
            if not content:
                continue
                
            # Calculate section score based on content length
            content_length = len(content)
            min_length = requirements['length']
            max_points = requirements['points']
            
            # If content is too short, give partial points
            if content_length < min_length:
                section_score = (content_length / min_length) * max_points
            else:
                section_score = max_points
                
            score += section_score
        
        # Adjust score based on strengths and improvements
        strengths = analysis_results.get('strengths', [])
        improvements = analysis_results.get('improvements', [])
        
        # Add points for strengths (up to 10 points)
        strength_points = min(len(strengths) * 2, 10)
        score += strength_points
        
        # Deduct points for improvements (up to 10 points)
        improvement_points = min(len(improvements) * 2, 10)
        score -= improvement_points
        
        # Ensure score stays within 0-100 range
        score = max(0, min(score, max_score))
        
        # Round to nearest integer
        score = round(score)
        
        logging.info(f"Calculated profile score: {score}")
        return score
        
    except Exception as e:
        logging.error(f"Error calculating profile score: {str(e)}")
        return 0

def generate_suggestions(profile, keywords):
    try:
        suggestions = []
        text = profile['summary'].lower()
        
        # Content suggestions
        if len(profile['summary']) < 200:
            suggestions.append("Expand your professional summary to be more comprehensive")
        if len(keywords) < 5:
            suggestions.append("Include more industry-specific keywords in your profile")
        if "achievement" not in text and "result" not in text:
            suggestions.append("Add more quantifiable achievements to your experience")
        
        # Formatting suggestions
        if len(profile['headline']) < 10:
            suggestions.append("Make your headline more descriptive and impactful")
        if len(profile['experience']) < 100:
            suggestions.append("Add more detail to your work experience section")
        if len(profile['skills']) < 50:
            suggestions.append("Expand your skills section with more specific competencies")
        
        # Content quality suggestions
        if "data" in text and "analytics" not in text:
            suggestions.append("Consider adding data analytics experience if relevant")
        if "cloud" in text and "aws" not in text and "azure" not in text:
            suggestions.append("Specify cloud platforms you're familiar with")
        if "agile" in text and "scrum" not in text:
            suggestions.append("Mention specific agile methodologies you've used")
        
        # Industry-specific suggestions
        if "tech" in text or "software" in text:
            suggestions.append("Include specific programming languages and frameworks")
        if "market" in text or "sales" in text:
            suggestions.append("Add metrics about sales performance or market impact")
        if "finance" in text or "account" in text:
            suggestions.append("Mention financial software or tools you're familiar with")
        
        # Professional development suggestions
        if "certification" not in text:
            suggestions.append("Consider adding relevant professional certifications")
        if "education" not in text or len(profile['education']) < 50:
            suggestions.append("Expand your education section with relevant details")
        
        return suggestions
    except Exception as e:
        logger.error(f"Error in generate_suggestions: {str(e)}")
        return ["Unable to generate suggestions"]

def analyze_strengths_and_improvements(profile, sentiment_score):
    try:
        strengths = []
        improvements = []
        
        # Combine all text sections for analysis
        text = f"{profile['headline']} {profile['summary']} {profile['experience']} {profile['skills']} {profile['education']}".lower()
        
        # Professional strengths
        if "leadership" in text or "manage" in text or "direct" in text:
            strengths.append("Strong leadership and management experience")
        if "project" in text or "program" in text:
            strengths.append("Project and program management expertise")
        if "technical" in text or "develop" in text or "engineer" in text:
            strengths.append("Technical proficiency and development skills")
        if "innov" in text or "creativ" in text:
            strengths.append("Innovative and creative problem-solving abilities")
        if "communicat" in text or "present" in text:
            strengths.append("Strong communication and presentation skills")
        if "team" in text or "collaborat" in text:
            strengths.append("Team collaboration and interpersonal skills")
        if "analyt" in text or "research" in text:
            strengths.append("Analytical and research capabilities")
        if "strateg" in text or "plan" in text:
            strengths.append("Strategic planning and execution")
        
        # Education-based strengths
        if "degree" in text or "bachelor" in text or "master" in text or "phd" in text:
            strengths.append("Strong educational background")
        if "certification" in text or "certified" in text:
            strengths.append("Professional certifications and qualifications")
        
        # Experience-based strengths
        if "experience" in text and len(profile['experience']) > 100:
            strengths.append("Comprehensive work experience")
        if "achievement" in text or "result" in text:
            strengths.append("Track record of achievements and results")
        
        # Identify areas for improvement
        if "certification" not in text and "certified" not in text:
            improvements.append("Consider adding relevant professional certifications")
        if "volunteer" not in text and "community" not in text:
            improvements.append("Include volunteer work or community involvement")
        if "mentor" not in text and "teach" not in text:
            improvements.append("Add mentoring or teaching experience")
        if "achievement" not in text and "result" not in text:
            improvements.append("Include more quantifiable achievements and results")
        if "skill" not in text and "expertise" not in text:
            improvements.append("Add more specific skills and areas of expertise")
        if "network" not in text and "connect" not in text:
            improvements.append("Highlight your professional network and connections")
        if "goal" not in text and "objective" not in text:
            improvements.append("Add your career goals and objectives")
        
        # Section-specific improvements
        if len(profile['headline']) < 10:
            improvements.append("Make your headline more descriptive and impactful")
        if len(profile['experience']) < 100:
            improvements.append("Add more detail to your work experience section")
        if len(profile['skills']) < 50:
            improvements.append("Expand your skills section with more specific competencies")
        if len(profile['education']) < 50:
            improvements.append("Expand your education section with more relevant details")
        
        # Add sentiment-based suggestions
        if sentiment_score < 3:
            improvements.append("Consider using more positive and confident language")
        elif sentiment_score > 4:
            strengths.append("Strong positive and confident tone")
        
        return strengths, improvements
    except Exception as e:
        logger.error(f"Error in analyze_strengths_and_improvements: {str(e)}")
        return [], []

def analyze_text(text, model_name, task_type="text-generation"):
    """Analyze text using HuggingFace API with better error handling"""
    if not HUGGINGFACE_API_KEY:
        logger.error("HuggingFace API key not found in environment variables")
        raise ValueError("HuggingFace API key not configured")
    
    # Truncate text to a reasonable length (1024 characters)
    truncated_text = text[:1024] if len(text) > 1024 else text
    logger.debug(f"Truncated text length: {len(truncated_text)}")
    
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        logger.debug(f"Sending request to {model_name}")
        payload = {
            "inputs": truncated_text,
            "parameters": {
                "max_length": 1024,
                "truncation": True
            }
        }
        
        # Add task-specific parameters
        if task_type == "text-generation":
            payload["parameters"].update({
                "max_length": 200,
                "num_return_sequences": 3,
                "temperature": 0.7
            })
        
        response = requests.post(
            f"{API_URL}/{model_name}",
            headers=headers,
            json=payload
        )
        
        logger.debug(f"Response status: {response.status_code}")
        
        if response.status_code == 401:
            logger.error("Authentication failed with HuggingFace API")
            logger.error(f"Response content: {response.text}")
            raise ValueError("Invalid HuggingFace API key")
        elif response.status_code != 200:
            logger.error(f"API request failed with status {response.status_code}")
            logger.error(f"Response content: {response.text}")
            raise ValueError(f"API request failed: {response.text}")
            
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise

def generate_pdf_report(analysis_data):
    """Generate a beautiful PDF report from the analysis data"""
    # Create a buffer to store the PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Container for the 'Flowable' objects
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#0077B5')  # LinkedIn blue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#0077B5')
    )
    
    # Title
    elements.append(Paragraph("LinkedIn Profile Analysis Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Date
    date_style = ParagraphStyle(
        'Date',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray
    )
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y %H:%M')}", date_style))
    elements.append(Spacer(1, 30))
    
    # Profile Score
    score_style = ParagraphStyle(
        'Score',
        parent=styles['Heading2'],
        fontSize=36,
        textColor=colors.HexColor('#0077B5'),
        alignment=1  # Center alignment
    )
    elements.append(Paragraph(f"Profile Score: {analysis_data['score']}%", score_style))
    elements.append(Spacer(1, 30))
    
    # Key Strengths
    elements.append(Paragraph("Key Strengths", heading_style))
    strengths_data = [[Paragraph(strength, styles['Normal'])] for strength in analysis_data['strengths']]
    strengths_table = Table(strengths_data, colWidths=[6*inch])
    strengths_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E1F0FA')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0077B5')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#0077B5')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(strengths_table)
    elements.append(Spacer(1, 20))
    
    # Areas for Improvement
    elements.append(Paragraph("Areas for Improvement", heading_style))
    improvements_data = [[Paragraph(improvement, styles['Normal'])] for improvement in analysis_data['improvements']]
    improvements_table = Table(improvements_data, colWidths=[6*inch])
    improvements_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFF5F5')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#DC3545')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DC3545')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(improvements_table)
    elements.append(Spacer(1, 20))
    
    # Suggested Keywords
    elements.append(Paragraph("Suggested Keywords", heading_style))
    keywords_data = [[Paragraph(keyword, styles['Normal'])] for keyword in analysis_data['keywords']]
    keywords_table = Table(keywords_data, colWidths=[6*inch])
    keywords_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#212529')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DEE2E6')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(keywords_table)
    elements.append(Spacer(1, 20))
    
    # Detailed Suggestions
    elements.append(Paragraph("Detailed Suggestions", heading_style))
    suggestions_data = [[Paragraph(suggestion, styles['Normal'])] for suggestion in analysis_data['suggestions']]
    suggestions_table = Table(suggestions_data, colWidths=[6*inch])
    suggestions_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#212529')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DEE2E6')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(suggestions_table)
    
    # Build the PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer and write it to the response
    buffer.seek(0)
    return buffer

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json()
        pdf_buffer = generate_pdf_report(data)
        
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'linkedin_profile_analysis_{timestamp}.pdf'
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 