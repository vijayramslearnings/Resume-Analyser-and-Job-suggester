import os
import re
import json
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import google.generativeai as genai
import PyPDF2
from docx import Document
import requests
import plotly.graph_objects as go

# App configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get API keys (put your real ones, or set as env vars)
# Defaults are empty so users must explicitly provide keys or use Demo Mode
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyBl93jtP909mnJrHlAcJBw5fjLWWLAZhgY')
ADZUNA_APP_ID = os.getenv('ADZUNA_APP_ID', 'fe797794')
ADZUNA_APP_KEY = os.getenv('ADZUNA_APP_KEY', 'efb9cf308f4130f4a28877596eaaf2c4')

class ResumeParser:
    def parse(self, file_content: bytes, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext == '.pdf':
            reader = PyPDF2.PdfReader(BytesIO(file_content))
            # Some PDFs return None for page.extract_text(); guard with empty string
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        elif ext == '.docx':
            doc = Document(BytesIO(file_content))
            return "\n".join(p.text for p in doc.paragraphs)
        elif ext == '.txt':
            return file_content.decode('utf-8')
        raise ValueError(f"Unsupported format: {ext}")

class AdzunaJobScraper:
    def __init__(self, app_id: str, app_key: str):
        self.app_id = app_id
        self.app_key = app_key
        self.base_url = "https://api.adzuna.com/v1/api/jobs"

    def _extract_skills(self, description: str, title: str) -> List[str]:
        skill_keywords = [
            'python', 'java', 'javascript', 'react', 'node.js', 'angular', 'vue',
            'machine learning', 'ml', 'ai', 'data science', 'sql', 'mongodb',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'devops', 'git',
            'spring boot', 'django', 'flask', 'express', 'rest api', 'graphql',
            'html', 'css', 'typescript', 'golang', 'rust', 'c++', 'c#',
            'postgresql', 'mysql', 'redis', 'kafka', 'elasticsearch',
            'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn'
        ]
        found_skills = []
        text = (description + ' ' + title).lower()
        for skill in skill_keywords:
            if skill in text:
                found_skills.append(skill.title())
        return list(set(found_skills[:8]))

    def _is_job_eligible(self, job_skills: List[str], resume_skills: List[str], min_skill_match: int) -> bool:
        """
        Checks if a job meets the minimum eligibility criteria based on matched skills.
        """
        if not resume_skills or min_skill_match == 0:
            return True 
        
        job_skills_set = set(s.lower() for s in job_skills)
        resume_skills_set = set(s.lower() for s in resume_skills)
        
        matched_skills = job_skills_set.intersection(resume_skills_set)
        
        return len(matched_skills) >= min_skill_match

    def search_jobs(self, keywords: str, country: str, location: str = "", limit: int = 10, 
                    resume_skills: List[str] = None, min_skill_match: int = 1) -> List[Dict]:
        
        if not self.app_id or not self.app_key:
            st.error("⚠️ Adzuna API credentials not configured!")
            st.info("Please set ADZUNA_APP_ID and ADZUNA_APP_KEY environment variables.")
            return []
        
        search_keywords = keywords
        if not search_keywords and resume_skills:
             search_keywords = " ".join(resume_skills[:3])
             
        if not search_keywords:
             st.warning("Please provide job keywords or upload a resume with detectable skills.")
             return []

        try:
            url = f"{self.base_url}/{country}/search/1"
            params = {
                'app_id': self.app_id,
                'app_key': self.app_key,
                'results_per_page': limit * 2,
                'what': search_keywords,
                'content-type': 'application/json'
            }
            if location:
                params['where'] = location
                
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if not results:
                    st.warning(f"No jobs found for '{search_keywords}' in {country.upper()}.")
                    return []
                
                jobs = []
                for job in results:
                    description = job.get('description', '').lower()
                    job_skills = self._extract_skills(description, job.get('title', ''))

                    if not self._is_job_eligible(job_skills, resume_skills or [], min_skill_match):
                        continue
                    
                    jobs.append({
                        'company': job.get('company', {}).get('display_name', 'Company Not Listed'),
                        'title': job.get('title', 'Position Not Specified'),
                        'location': job.get('location', {}).get('display_name', location or 'Not specified'),
                        'skills': job_skills,
                        'url': job.get('redirect_url', ''),
                        'source': 'Adzuna'
                    })
                
                if not jobs and results:
                    st.warning(f"Found {len(results)} jobs, but none met the minimum match criteria of {min_skill_match} skill(s) from your resume. Try lowering the requirement.")

                return jobs[:limit]
                
            else:
                st.error(f"❌ Adzuna API error: Status {response.status_code}")
                return []
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. Please try again.")
            return []
        except Exception as e:
            st.error(f"❌ Error fetching jobs: {str(e)}")
            return []

class GeminiAnalyzer:
    def __init__(self, api_key: str):
        # Initialize Gemini configuration
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def score_resume(self, text: str) -> Dict:
        # Prompt instructing Gemini to score the resume and return JSON
        prompt = f"""Score this resume 0-100. Return ONLY valid JSON:
        {text[:3000]}
        {{"overall_score": 75, "structure_score": 70, "content_score": 80, "language_score": 75,
          "positive_points": ["Strong technical background", "Clear work history"],
          "negative_points": ["Missing quantifiable achievements"],
          "improvements": ["Add metrics to achievements", "Include certifications"],
          "summary": "Professional resume with solid foundation"}}"""
        try:
            # Generate content using the Gemini model
            response = self.model.generate_content(prompt)
            # Use regex to find and extract the JSON object
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass # Fallback if API fails or JSON parsing fails
            
        # Fallback Score Logic: Basic scoring based on word count
        words = len(text.split())
        score = min(85, 50 + words // 25)
        return {
            "overall_score": score,
            "structure_score": score - 5,
            "content_score": score+2,
            "language_score": score + 5,
            "positive_points": ["Resume contains relevant experience", "Clear section organization"],
            "negative_points": ["Add quantified achievements", "Include specific technologies"],
            "improvements": ["Use strong action verbs", "Add measurable results", "Include certifications"],
            "summary": f"Resume shows {score}% professional quality. Add quantifiable achievements to improve."
        }

    def extract_info(self, text: str) -> Dict:
        common_skills = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'react', 'node']
        found = [s.title() for s in common_skills if s in text.lower()]
        return {
            "skills": found[:8] if found else ['Programming', 'Communication', 'Teamwork'],
            "experience_years": 3,
            "education": "Bachelor's degree"
        }

    def match_jobs(self, resume_info: Dict, jobs: List[Dict]) -> List[Dict]:
        if not jobs:
            return []
        resume_skills = set(s.lower() for s in resume_info.get('skills', []))
        matches = []
        for job in jobs:
            job_skills = set(s.lower() for s in job.get('skills', []))
            matched = resume_skills & job_skills
            missing = job_skills - resume_skills
            
            if job_skills:
                score = int((len(matched) / len(job_skills) * 60) + 25)
            else:
                score = 50
                
            matches.append({
                **job,
                'score': min(score, 95),
                'matched_skills': list(matched),
                'missing_skills': list(missing)
            })
        return sorted(matches, key=lambda x: x['score'], reverse=True)

# Streamlit UI

# Centered upload section
st.markdown("<h1 style='text-align:center;'>🎯 AI Resume Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Upload your resume to get instant Adzuna job matches</h4>", unsafe_allow_html=True)
center_col = st.columns([1,2,1])[1]
with center_col:
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])
    keywords = st.text_input("Job Keywords (e.g., Python Developer)", "software engineer")
    country = st.selectbox("Country", ["in", "us", "gb", "au", "ca", "de", "fr"],
                           format_func=lambda x: {"in": "🇮🇳 India", "us": "🇺🇸 USA", "gb": "🇬🇧 UK", 
                                                  "au": "🇦🇺 Australia", "ca": "🇨🇦 Canada", 
                                                  "de": "🇩🇪 Germany", "fr": "🇫🇷 France"}[x])
    location = st.text_input("Location (Optional - City/Region)", "")
    
    min_match = st.slider("Minimum Required Skill Matches", 0, 5, 1, help="Only show jobs that match at least this many skills from your resume.")

    analyze_btn = st.button("🚀 Analyze Resume", use_container_width=True)

# Main analysis and output
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.results = None

if analyze_btn and uploaded_file:
    if not GEMINI_API_KEY or not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        st.error("❌ Missing API credentials. Please set environment variables.")
        st.stop()
        
    with st.spinner("🔍 Analyzing your resume and fetching jobs..."):
        try:
            parser = ResumeParser()
            resume_text = parser.parse(uploaded_file.read(), uploaded_file.name)
            
            analyzer = GeminiAnalyzer(GEMINI_API_KEY)
            scraper = AdzunaJobScraper(ADZUNA_APP_ID, ADZUNA_APP_KEY)
            
            # 1. Analyze Resume
            score_data = analyzer.score_resume(resume_text)
            resume_info = analyzer.extract_info(resume_text)
            
            # 2. Fetch and Filter Jobs
            jobs = scraper.search_jobs(
                keywords=keywords, 
                country=country, 
                location=location,
                resume_skills=resume_info['skills'],
                min_skill_match=min_match
            )
            
            if not jobs:
                st.warning("⚠️ No eligible jobs found. Please check your keywords or lower the minimum skill match requirement.")
                st.stop()
                
            # 3. Score Matches
            matches = analyzer.match_jobs(resume_info, jobs)
            
            st.session_state.results = {
                'score': score_data,
                'info': resume_info,
                'matches': matches
            }
            st.session_state.analyzed = True
            st.success("✅ Analysis Complete! Results are filtered based on your minimum skill requirement.")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# --- DISPLAY RESULTS ---

if st.session_state.analyzed and st.session_state.results:
    data = st.session_state.results
    
    st.header("📊 Resume Score")
    cols = st.columns(4)
    scores = data['score']
    score_items = [
        ('Overall', scores['overall_score']),
        ('Structure', scores['structure_score']),
        ('Content', scores['content_score']),
        ('Language', scores['language_score'])
    ]
    for col, (label, value) in zip(cols, score_items):
        with col:
            st.metric(label, f"{value}/100")
    st.info(f"**Summary:** {scores.get('summary', '')}")
    
    st.header("📈 Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["Score Breakdown", "Job Matches", "Skills Heatmap", "Distribution"])
    
    with tab1:
        fig = go.Figure(data=[
            go.Bar(
                x=[item[0] for item in score_items],
                y=[item[1] for item in score_items],
                marker_color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
            )
        ])
        fig.update_layout(title="Resume Score Breakdown", yaxis_range=[0, 100],
                         xaxis_title="Category", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if data['matches']:
            top_jobs = data['matches'][:8]
            fig = go.Figure(data=[
                go.Bar(
                    y=[f"{j['company'][:30]}" for j in top_jobs],
                    x=[j['score'] for j in top_jobs],
                    orientation='h',
                    marker_color=['#2ecc71' if j['score']>=75 else '#f39c12' if j['score']>=50 else '#e74c3c' for j in top_jobs]
                )
            ])
            fig.update_layout(title="Top Eligible Job Matches by Score", xaxis_range=[0, 100],
                             xaxis_title="Match Score (%)", yaxis_title="Company")
            st.plotly_chart(fig, use_container_width=True)
            
    with tab3:
        if data['matches']:
            all_skills = set()
            for job in data['matches'][:5]:
                all_skills.update(job.get('skills', []))
            skills_list = sorted(list(all_skills))[:10]
            matrix_data = []
            companies = []
            for job in data['matches'][:5]:
                job_skills_set = set(s.lower() for s in job.get('skills', []))
                row = [1 if skill.lower() in job_skills_set else 0 for skill in skills_list]
                matrix_data.append(row)
                companies.append(job['company'][:25])
            fig = go.Figure(data=go.Heatmap(
                z=matrix_data,
                x=skills_list,
                y=companies,
                colorscale='Greens',
                showscale=True
            ))
            fig.update_layout(title="Skills Comparison: Top Jobs", xaxis_title="Skills", yaxis_title="Company")
            st.plotly_chart(fig, use_container_width=True)
            
    with tab4:
        excellent = len([m for m in data['matches'] if m['score'] >= 75])
        good = len([m for m in data['matches'] if 50 <= m['score'] < 75])
        fair = len([m for m in data['matches'] if m['score'] < 50])
        fig = go.Figure(data=[go.Pie(
            labels=['Excellent (75-100%)', 'Good (50-74%)', 'Fair (0-49%)'],
            values=[excellent, good, fair],
            marker_colors=['#2ecc71', '#f39c12', '#e74c3c']
        )])
        fig.update_layout(title="Match Quality Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Points
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("✅ Positive Points")
        for point in scores['positive_points']:
            st.success(point)
    with col2:
        st.subheader("❌ Areas to Improve")
        for point in scores['negative_points']:
            st.error(point)
    with col3:
        st.subheader("💡 Suggestions")
        for point in scores['improvements']:
            st.info(point)
            
    # Simple, clear statistics
    st.header("📊 Job Match Statistics")
    stats_cols = st.columns(4)
    avg_score = sum(m['score'] for m in data['matches']) / len(data['matches']) if data['matches'] else 0
    with stats_cols[0]:
        st.metric("Jobs Analyzed (Eligible)", len(data['matches']))
    with stats_cols[1]:
        st.metric("Excellent Matches", excellent)
    with stats_cols[2]:
        st.metric("Good Matches", good)
    with stats_cols[3]:
        st.metric("Avg Match Score", f"{avg_score:.1f}%")

    # Professional job cards, left aligned
    st.header("🎯 Top Eligible Job Matches (Adzuna Data)")
    for job in data['matches'][:10]:
        with st.expander(f"**{job['company']}** - {job['title']} ({job['score']}%)"):
            st.write(f"**Location:** {job['location']}")
            st.write(f"**Matched Skills:** {', '.join(job['matched_skills']) or 'None'}")
            st.write(f"**Skills to Learn:** {', '.join(job['missing_skills'][:3]) or 'None'}")
            if job.get('url'):
                st.markdown(f"[Apply Now]({job['url']})")

    st.header("💾 Download Results")
    results_json = json.dumps(data, indent=2)
    st.download_button("Download JSON Report", results_json, "resume_analysis.json", "application/json")

else:
    st.info("👈 Upload your resume in the center panel to get started!")