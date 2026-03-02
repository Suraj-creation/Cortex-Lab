#!/usr/bin/env python3
"""
Add missing structured data to Cortex Lab database.
Uses the running API (port 8000) to avoid DuckDB locking issues.
Does NOT delete existing data - only adds what's missing.
"""

import requests
import time
import sys

API = "http://localhost:8000"

# Check API health
try:
    r = requests.get(f"{API}/api/health", timeout=5)
    if r.status_code != 200:
        print("❌ Backend not healthy"); sys.exit(1)
    print("✅ Backend is healthy")
except:
    print("❌ Cannot connect to backend on port 8000"); sys.exit(1)

# Get existing memories to avoid duplicates
existing = requests.get(f"{API}/api/memories").json().get("memories", [])
existing_sources = set()
for m in existing:
    src = m.get("source", "")
    if src:
        existing_sources.add(src)
print(f"📊 Existing memories: {len(existing)} with sources: {existing_sources}")


MISSING_CHUNKS = [
    # ─── CONTACT & PROFILE ───
    {
        "source": "contact-links",
        "content": (
            "Suraj Kumar's online profiles and contact links: "
            "Email: surajcreationinfinity1@gmail.com. "
            "Phone: +91 6204153972. "
            "LinkedIn: http://www.linkedin.com/in/surajkumarvu. "
            "GitHub: https://github.com/Suraj-creation. "
            "Portfolio website: https://portfolio-n93j.vercel.app/. "
            "He is active on LinkedIn and GitHub where he showcases his projects and professional profile."
        )
    },
    {
        "source": "professional-summary",
        "content": (
            "Suraj Kumar's professional summary: "
            "AI enthusiast with a strong foundation in ML/DL/RL, probabilistic modelling, and optimization. "
            "Specialized in building end-to-end ML systems—from data analysis, feature engineering, model development, "
            "and evaluation to production deployment using MLOps and DevOps practices for scalable, reliable AI solutions. "
            "He has strong CS fundamentals in DSA, OS, CN, DBMS, and computer architecture. "
            "Driven to create disruptive, high-impact AI products across industries by combining deep technical execution "
            "with strong product and systems thinking. He is an AI-focused B.Tech CSE (Data Science) undergrad "
            "with a Minor in Finance at Vidyashilp University, Bangalore."
        )
    },

    # ─── MISSING PROJECTS ───
    {
        "source": "projects-EdgeMemory",
        "content": (
            "📌 Project Name: EdgeMemory - On-Device Lifelong Personal Memory AI. "
            "Status: Ongoing project (2026). Domain: Deep Learning, On-Device AI, Privacy-First AI. "
            "Description: Building a privacy-first on-device lifelong memory system that converts voice reflections "
            "into timestamped memory chunks, enabling time-aware recall and semantic search using ASR, embeddings, "
            "and vector retrieval. Developing a deep learning pipeline with MLP classifiers for topic/importance/emotion "
            "tagging, CNNs on spectrograms for audio quality and emotion cues, and GRU/LSTM with Attention for temporal "
            "memory modelling across weeks/months/years. Implementing memory compression and edge optimisation using "
            "Autoencoders/VAEs, distillation, pruning, quantisation, and ONNX deployment, with a future roadmap to "
            "apply RL-based memory retention policies for selecting the most valuable memories under storage/compute constraints."
        )
    },
    {
        "source": "projects-ResumeEnhancer",
        "content": (
            "📌 Project Name: AI-Powered Resume Enhancer & Job Matching Application. "
            "Timeline: March 2025 - April 2025. Domain: AI, DevOps, Full Stack. "
            "Description: Built an end-to-end AI-driven resume enhancement and job matching platform, enabling users "
            "to upload resumes, receive ATS optimization suggestions, and compute resume-job similarity scores. "
            "Developed a production-grade full-stack application with secure authentication, role-based access, "
            "multi-format resume ingestion (PDF/DOCX), real-time editing, and live resume preview. "
            "Integrated Supabase (PostgreSQL + Auth + Storage) for scalable backend infrastructure. "
            "Deployed using Dockerized services with CI/CD automation (GitHub Actions) and cloud-ready delivery practices. "
            "Designed a modular AI recommendation engine to generate actionable improvements (missing skills, weak bullets, "
            "impact rewrites) and automatically generate ATS-friendly PDF exports."
        )
    },
    {
        "source": "projects-MahindraFinance",
        "content": (
            "📌 Project Name: Mahindra & Mahindra Financial Analysis Dashboard. "
            "Timeline: 2025. Domain: Financial Analytics, Data Visualization, Corporate Finance. "
            "Description: Designed and developed an interactive financial analytics dashboard to evaluate the multi-year "
            "performance of Mahindra & Mahindra Ltd, benchmarked against TATA Motors Ltd, covering fiscal years 2022-2025. "
            "Performed comprehensive ratio analysis across liquidity, solvency, profitability, and efficiency. "
            "Implemented advanced financial diagnostics, including DuPont ROE decomposition (3-point & 5-point models), "
            "working-capital efficiency analysis, and capital structure evaluation. "
            "Built executive-grade visualizations using Plotly.js—KPI cards with sparklines, radar charts, waterfall charts, "
            "gauge-based financial health scores, and side-by-side peer comparison tables. "
            "Engineered a modular, SPA-style frontend architecture (HTML, CSS, JavaScript, Tailwind). "
            "Integrated predictive trend projections for revenue and profitability."
        )
    },
    {
        "source": "projects-SelfStabilizingSpoon",
        "content": (
            "📌 Project Name: Self-Stabilizing Spoon for Parkinson's Disease. "
            "Timeline: February 2024 - March 2024. Role: Project Lead. "
            "Domain: Hardware, IoT, Healthcare, Arduino. "
            "Description: Designed and developed an Arduino-powered adaptive spoon using IMU sensors and motion "
            "stabilization techniques to counteract hand tremors in Parkinson's patients. "
            "Enhanced grip stability and usability, enabling independent eating for Parkinson's patients "
            "with an 80% success rate in testing. This project demonstrates Suraj's skills in hardware prototyping, "
            "sensor integration, and assistive technology for healthcare applications."
        )
    },
    {
        "source": "projects-CrimeEDA",
        "content": (
            "📌 Project Name: Exploratory Data Analysis on Crimes Against Women in India. "
            "Timeline: November 2023 - December 2023. Domain: Data Analytics, EDA, Social Impact. "
            "Description: Performed EDA on large-scale public crime datasets using Python libraries "
            "(Pandas, Matplotlib/Seaborn) to identify patterns and trends in crimes against women in India. "
            "Developed visual reports and dashboards using Power BI, presenting key findings and predictive insights "
            "to potentially aid prevention strategies. Improved data processing efficiency by 35% using optimized workflows."
        )
    },
    {
        "source": "projects-PredictiveCrime",
        "content": (
            "📌 Project Name: Predictive Crime Detection System Using CCTV. "
            "Timeline: January 2025 - April 2025. Domain: AI, Computer Vision, Security, Research Project. "
            "Description: By integrating facial recognition, gaze estimation, emotion analysis, sequential behavior analysis, "
            "and semantic segmentation, the system detects suspicious activities, predicts potential crimes, and alerts "
            "authorities in real time. The goal is to revolutionize security by reducing crime rates, improving response times, "
            "and minimizing reliance on manual monitoring while ensuring ethical implementation and privacy compliance. "
            "By leveraging advanced AI models and probabilistic networks, it not only predicts but also prevents criminal "
            "activities, creating a safer, smarter, and more adaptive surveillance ecosystem."
        )
    },
    {
        "source": "projects-DevPsychology",
        "content": (
            "📌 Project Name: Developmental Psychology Research. "
            "Timeline: January 2024 - February 2024. Domain: Psychology, Research, Child Development. "
            "Description: Conducted observational studies on child development across different environments, "
            "analyzing cognitive, emotional, and social growth. Researched the impact of parenting styles on "
            "behavioural and psychological development, providing insights into early childhood education and mental health. "
            "This demonstrates Suraj's interdisciplinary interests beyond technology."
        )
    },
    {
        "source": "projects-WaterScarcity",
        "content": (
            "📌 Project Name: Data Visualization & Case Study on Water Scarcity in India. "
            "Timeline: October 2023 - November 2023. Domain: Analytics, Data Storytelling, Social Impact. "
            "Description: Performed in-depth data-driven storytelling using Power BI and Tableau to highlight "
            "water scarcity challenges across Indian states. Utilized statistics with R to uncover trends and "
            "correlations of regional disparities, root causes, and policy gaps, proposing data-driven sustainable solutions."
        )
    },
    {
        "source": "projects-PortfolioOptimization",
        "content": (
            "📌 Project Name: Portfolio Optimization & Risk Analytics Platform. "
            "Timeline: October 2025 - November 2025. Domain: Fintech, Risk Analytics, Full Stack. "
            "Description: Designed and implemented a full-stack portfolio optimization system applying Modern Portfolio Theory "
            "(Markowitz) to construct an optimal 8-asset NSE equity portfolio (2020-2024) under real-world constraints "
            "(no short-selling, transaction costs, Indian risk-free rate). "
            "Engineered risk diagnostics and diversification analysis, including rolling volatility, skewness/kurtosis, "
            "correlation heatmaps, efficient frontier construction (30 portfolios), and Monte Carlo simulations. "
            "Achieved a 15.6% improvement in Sharpe ratio over an equal-weighted baseline (1.377 vs 1.192). "
            "Developed an interactive, Excel-inspired analytics dashboard with Plotly.js and Tailwind CSS. "
            "Integrated AI-powered portfolio insights using Google Gemini API for interpretable investment rationale."
        )
    },

    # ─── MISSING LEADERSHIP PROGRAMS ───
    {
        "source": "leadership-NSDP",
        "content": (
            "Suraj Kumar completed the National Scholar Development Program by Dexterity Global (Patna, India) "
            "from August 2023 to October 2023. This was a three-month intensive program focused on preparing young "
            "scholars for top-tier academic and professional environments. He acquired advanced skills in research "
            "methodologies, effective communication, storytelling, and strategic planning, critical for college "
            "applications and leadership roles. He received personalized mentoring, resulting in a graduation "
            "certificate and strong recommendation letters for future endeavors."
        )
    },
    {
        "source": "leadership-AspireLeaders",
        "content": (
            "Suraj Kumar graduated from the Aspire Leaders Program by Aspire Institute (initiated by Harvard Business "
            "School faculty) from August 2024 to February 2025. This is a globally recognized leadership development "
            "program designed for first-generation college students and recent graduates. He enhanced leadership capabilities, "
            "professional branding, and global networking skills through interactive online modules. He collaborated with "
            "a diverse international cohort from 190+ countries, fostering cross-cultural understanding and innovative "
            "idea exchange. He acquired strategies to drive positive community impact."
        )
    },

    # ─── MISSING AWARDS & DETAILS ───
    {
        "source": "awards-detailed",
        "content": (
            "Suraj Kumar's detailed awards and recognitions: "
            "1) Top 50 Brightest Minds of Bihar (2023) - Recognized as a Times Bihar Scholar by the Ministry of Finance, "
            "Government of Bihar. Awarded a Certificate of Achievement and a Xiaomi Pad 5. "
            "2) Vibe Coding Hackathon (2025) - Secured a position among the Top 7 winning teams, recognized for "
            "delivering a high-impact solution with strong technical execution. "
            "3) Science Model Competition Finalist (2021) - Organized by Agastya International Foundation. Secured "
            "Top 10 out of 80 participants with an innovative project on electromagnetic induction-based walking-powered "
            "battery charging. "
            "4) 100% Scholarship at Vidyashilp University, Bangalore for the undergraduate program. "
            "5) Karta Initiative India Foundation Scholar - Selected in recognition of academic excellence, leadership "
            "potential, and commitment to social impact. Received one-on-one mentoring support and participated in "
            "leadership and workshop events."
        )
    },

    # ─── SPORTS & LANGUAGES ───
    {
        "source": "personal-sports-languages",
        "content": (
            "Suraj Kumar is a sports enthusiast who has participated in Hockey, Martial Arts, Volleyball, Badminton, "
            "Cricket, and Table Tennis. He also plays Carrom. He speaks two languages: English and Hindi."
        )
    },

    # ─── EDUCATION DETAILS (CLASS SCORES) ───
    {
        "source": "education-scores",
        "content": (
            "Suraj Kumar's academic scores: Class 10th (2021) - 92% from Shoshit Samadhan Kendra, Patna, Bihar. "
            "Class 12th (2023) - 79% from Shoshit Samadhan Kendra, Patna, Bihar. "
            "Currently pursuing B.Tech (Hons) in Computer Science Engineering with specialization in Data Science "
            "and a Minor in Finance at Vidyashilp University, Bangalore, Karnataka (2023-2027). "
            "He received a 100% scholarship for the undergraduate program."
        )
    },

    # ─── TECHNICAL SKILLS DETAIL ───
    {
        "source": "skills-math-cs-fundamentals",
        "content": (
            "Suraj Kumar's CS fundamentals and mathematics skills: "
            "CS Fundamentals: Data Structures & Algorithms (DSA), Operating Systems (OS), Computer Networks (CN), "
            "Database Management Systems (DBMS), Computer Architecture, Object-Oriented Programming (OOPs). "
            "Mathematics & Statistics: Probability, Linear Algebra, Calculus, Optimization Techniques, Statistics. "
            "Hardware & IoT: Arduino, IMU Sensors, 3D Design & Modeling, Robot Operating System (ROS), Research and Analytics. "
            "Data Analytics & Visualization: Power BI, Tableau, EDA, Statistics, Data Storytelling. "
            "Tools: Arduino, Git, Canva, Cursor AI, Trello, Notion, Replit, n8n."
        )
    },
    {
        "source": "skills-soft",
        "content": (
            "Suraj Kumar's soft skills: Effective Communication, Presentation Skills, Leadership, Teamwork, "
            "Critical Thinking, Problem-Solving, Adaptability. He has demonstrated these through his leadership "
            "programs (DexSchool, Aspire Leaders), volunteering at Tan90 SPARC, and coordinating events for 90+ schools. "
            "He is also skilled in strategic thinking, entrepreneurship, storytelling, research methodologies, "
            "and professional branding."
        )
    },

    # ─── GoCrackIT INTERNSHIP DETAILS ───
    {
        "source": "internship-GoCrackIT-detailed",
        "content": (
            "Suraj Kumar's internship at GoCrackIT (AI Intern, Full Stack): Duration June 1 to July 31, 2025. "
            "He designed and developed an AI-powered Autofill and Resume Generation platform using Flask and Python, "
            "enabling automatic extraction of structured data from text, PDFs, DOCX, images, and audio inputs. "
            "He implemented a dual-pipeline architecture combining Gemini AI and open-source tools (PyPDF2, pdfplumber, "
            "Tesseract OCR, spaCy), achieving 98% accuracy for personal information and 96% accuracy for structured fields. "
            "He performed advanced prompt engineering, document chunking, and validation logic to mitigate LLM context "
            "limitations and ensure consistent JSON-formatted outputs."
        )
    },

    # ─── GRAM VIKAS INTERNSHIP DETAILS ───
    {
        "source": "internship-GramVikas-detailed",
        "content": (
            "Suraj Kumar's volunteering internship at Gram Vikas, Jharsuguda, Orissa: Duration June 2024 - July 2024. "
            "Domain: Rural Development, Data Analysis, EDA. "
            "He collected and meticulously analyzed quantitative and qualitative data on nutrition, agriculture practices, "
            "and water resource availability across 20+ villages. He contributed to impact reports aimed at improving "
            "the livelihood and resource accessibility for marginalized rural communities served by Gram Vikas. "
            "His work supported evidence-based policy recommendations for improving rural livelihoods."
        )
    },

    # ─── STARTUP IDEAS / VISION ───
    {
        "source": "vision-startup-ideas",
        "content": (
            "Suraj Kumar's startup and product vision ideas: "
            "1) Advanced Agentic Code Editor - Like Cursor AI but with split-screen: one half writes code, "
            "the other half explains the logic in conversational English line by line as code proceeds. "
            "Also teaches the language from scratch and provides best practices for production-grade development. "
            "2) AI-Powered Comprehensive Personal Platform - A website serving as portfolio, IDE for projects, "
            "frequent thoughts updates, and storehouse of vision/ideas/thoughts/actions with public/private toggle. "
            "3) Replace the whole foundation of Technology - Replacing even hardware architecture and binary computation. "
            "4) Own Operating System + conversational coding language. "
            "5) AI that thinks and makes you think. "
            "6) Context-aware Prompt Generation application. "
            "7) Advance Agent Assistant that knows everything about you, defining per-minute goals."
        )
    },

    # ─── VOLUNTEERING DETAILS ───
    {
        "source": "volunteering-detailed",
        "content": (
            "Suraj Kumar's volunteering at Tan90 SPARC (Student-Powered Action for Research & Challenge) at "
            "Bal Bhavan, Bangalore (2025): He spearheaded the planning and execution of interactive STEM activities "
            "including Telescope Tales, F1 Race, and Big Buzzer Game, engaging students from 90+ schools across Bangalore. "
            "He cultivated organizational, leadership, and communication skills by mentoring student teams, driving "
            "enthusiasm for sustainability and scientific innovation. He directed a team of volunteers to streamline "
            "event logistics, ensuring a seamless experience for hundreds of participants including children, competitors, "
            "and visitors."
        )
    },

    # ─── PERSONAL PHILOSOPHY ───
    {
        "source": "personal-philosophy",
        "content": (
            "Suraj Kumar's personal motto: 'Be a warrior like Life. My life owes to all the suffering, cause, "
            "and future of this planet. I myself would transform every sector on this planet and initiate a most "
            "sustainable, revolutionized world of innovations for every sector.' "
            "He is deeply passionate about transforming education, technology, and society through AI and innovation. "
            "He believes in building disruptive products that solve high-impact problems across industries."
        )
    },

    # ─── SYSMIND-CLI DETAILS ───
    {
        "source": "projects-SysmindCLI-detailed",
        "content": (
            "📌 Project Name: Sysmind-CLI - System Intelligence & Automation CLI. "
            "GitHub: https://github.com/Suraj-creation/Sysmind-CLI. "
            "Domain: Systems Programming, DevOps, System Administration, Command-Line Tools. "
            "A unified command-line utility that provides intelligent system monitoring, process management, "
            "disk analytics, network diagnostics, and automated maintenance through a single cohesive interface. "
            "Key features: real-time system monitoring with historical baseline comparison, multi-phase duplicate "
            "file detection (size → quick hash → full hash), SQLite-based data persistence for trend analysis, "
            "cross-component intelligence correlation, health scoring algorithm (0-100), and platform abstraction layer "
            "supporting Windows, Linux, and macOS. Tech Stack: Python 3.8+, SQLite, Python Standard Library only."
        )
    },
]


def ingest_chunk(source: str, content: str) -> bool:
    """Ingest a single chunk via the chat API."""
    try:
        # Use the ingest endpoint directly
        r = requests.post(
            f"{API}/api/memories/ingest",
            json={"content": content, "source": source, "session_id": "structured-data"},
            timeout=60
        )
        if r.status_code == 200:
            return True
        else:
            print(f"    HTTP {r.status_code}: {r.text[:100]}")
            return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    success = 0
    fail = 0
    skipped = 0

    for i, chunk in enumerate(MISSING_CHUNKS):
        src = chunk["source"]
        content = chunk["content"]

        # Skip if source already exists
        if src in existing_sources:
            print(f"  [{i+1}/{len(MISSING_CHUNKS)}] ⏭  SKIP (exists): {src}")
            skipped += 1
            continue

        print(f"  [{i+1}/{len(MISSING_CHUNKS)}] 📝 Ingesting: {src}...")
        ok = ingest_chunk(src, content)
        if ok:
            print(f"    ✅ SUCCESS")
            success += 1
        else:
            print(f"    ❌ FAILED")
            fail += 1

        time.sleep(0.5)  # Small delay to avoid overwhelming the server

    print(f"\n{'='*60}")
    print(f"RESULTS: {success} added, {skipped} skipped, {fail} failed")
    print(f"Total memories now: {len(existing) + success}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
