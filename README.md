# LinkedIn Profile Optimizer

A powerful web application that uses AI to analyze and optimize your LinkedIn profile. Get detailed insights, improvement suggestions, and professional recommendations to make your profile stand out.

## 🌟 Features

- **AI-Powered Analysis**: Leverages HuggingFace's advanced AI models to analyze your profile content
- **Profile Scoring**: Get a comprehensive score of your profile's effectiveness
- **Smart Insights**: Receive detailed feedback on:
  - Key strengths
  - Areas for improvement
  - Suggested keywords
  - Professional recommendations
- **PDF Report Generation**: Download a professional PDF report of your analysis
- **Modern UI**: Clean, responsive design with intuitive user interface
- **Real-time Analysis**: Instant feedback on your profile content

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A HuggingFace API key
- python3-venv (for virtual environment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/linkedin-optimizer.git
cd linkedin-optimizer
```

2. Create and activate a virtual environment:
```bash
# Install python3-venv if not already installed
sudo apt install python3-venv

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your HuggingFace API key:
```
HUGGINGFACE_API_KEY=your_api_key_here
```

### Running the Application

1. Make sure your virtual environment is activated:
```bash
source venv/bin/activate
```

2. Start the Flask server:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

### Deactivating the Virtual Environment

When you're done using the application, you can deactivate the virtual environment:
```bash
deactivate
```

## 💡 Usage

1. Enter your LinkedIn profile information in the form:
   - Professional Headline
   - Professional Summary
   - Work Experience
   - Skills
   - Education

2. Click "Analyze Profile" to get your profile analysis

3. Review the results:
   - Overall profile score
   - Key strengths
   - Areas for improvement
   - Suggested keywords
   - Detailed suggestions

4. Generate a PDF report of your analysis by clicking the "Generate PDF Report" button

## 🛠️ Technical Details

### Backend
- Flask web framework
- HuggingFace API integration for:
  - Text summarization
  - Keyword extraction
  - Sentiment analysis
- PDF generation using ReportLab

### Frontend
- HTML5 and CSS3
- Modern, responsive design
- Real-time form validation
- Interactive UI elements
- PDF download functionality

## 📝 Dependencies

- Flask==3.0.2
- requests==2.31.0
- python-dotenv==1.0.1
- reportlab==4.0.9

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for providing the AI models
- Flask team for the web framework
- ReportLab for PDF generation capabilities

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

Made with ❤️ by [Your Name] 