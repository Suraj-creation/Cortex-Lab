#!/usr/bin/env python3
"""
Clean Data Ingestion Script for Cortex Lab
============================================
Reads raw_data files and ingests ONLY clean, structured, non-duplicated 
personal data chunks into DuckDB + FAISS + Knowledge Graph.

Each chunk is:
- Well-structured with clear context
- Non-duplicated (one authoritative version)
- Tagged with proper source
- Sized appropriately for embedding quality (100-500 words ideal)
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.embeddings import EmbeddingModel
from src.storage.vector_store import VectorStore
from src.storage.metadata_store import MetadataStore
from src.storage.knowledge_graph import KnowledgeGraph
from src.ingestion import MemoryIngestionPipeline
from src.llm import LocalLLM


# ═══════════════════════════════════════════════════════════════════
# CLEAN, STRUCTURED DATA CHUNKS
# Each chunk is a well-formed, self-contained piece of personal data
# ═══════════════════════════════════════════════════════════════════

PERSONAL_INFO_CHUNKS = [
    # ── Identity & Contact ───────────────────────────────────────
    {
        "content": "My name is Suraj Kumar. I am from Patna, Bihar, India. My email address is surajcreationinfinity1@gmail.com. My phone number is +91 6204153972. My LinkedIn profile is linkedin.com/in/suraj-kumar1806. My GitHub is github.com/Suraj-creation. My portfolio website is https://portfolio-n93j.vercel.app/.",
        "source": "personal-identity"
    },

    # ── Education ────────────────────────────────────────────────
    {
        "content": "I am currently pursuing B.Tech in Computer Science and Engineering with specialization in Data Science at Vidyashilp University, Bangalore, Karnataka, India. My enrollment period is 2023-2027. I received a 100% scholarship at Vidyashilp University. I was also selected as a Karta Initiative India Foundation Scholar.",
        "source": "education"
    },
    {
        "content": "I completed my schooling at Shoshit Samadhan Kendra, Bihar from 2010 to 2023. I scored 92% in Class 10th (CBSE Board). I scored 79% in Class 12th (CBSE Board).",
        "source": "education"
    },

    # ── Skills ───────────────────────────────────────────────────
    {
        "content": "My programming languages include Python, Java, C, and R. I have expertise in AI/ML/Deep Learning/Reinforcement Learning frameworks including PyTorch, TensorFlow, Keras, Scikit-learn, and XGBoost. I am proficient in web development using FastAPI, Flask, Node.js, Express.js, React.js, Next.js, HTML, CSS, JavaScript, and TypeScript. I also know MongoDB, SQL, DuckDB for databases. For DevOps and tools, I use Docker, Git, GitHub, Vercel, and Linux.",
        "source": "skills"
    },
    {
        "content": "My specialized skills include Natural Language Processing (NLP), Computer Vision (CV), Retrieval-Augmented Generation (RAG), Transformers, LLM Fine-tuning, Agentic AI, Prompt Engineering, EEG Signal Processing, and Modern Portfolio Theory. I also have experience with Google Gemini API integration, FAISS vector databases, and Streamlit for building ML applications.",
        "source": "skills"
    },

    # ── Internships ──────────────────────────────────────────────
    {
        "content": "I worked as an AI and Data Science Intern at GoCrackIT from January 2025 to April 2025. During this internship, I built an AI Resume Enhancer that analyzed resumes with NLP and provided optimization recommendations. I also developed an interactive Mahindra analytics dashboard using Python, Streamlit, and data visualization.",
        "source": "internships"
    },
    {
        "content": "I completed a Volunteering Internship at Gram Vikas, a rural development organization. This experience involved community engagement and understanding grassroots-level social development work.",
        "source": "internships"
    },

    # ── Volunteering ─────────────────────────────────────────────
    {
        "content": "I am a dedicated volunteer at Tan90 SPARC (Student-Powered Action for Research & Community). My volunteer work includes organizing community events, mentoring peers, and contributing to social impact initiatives. I have been actively involved in volunteering throughout my university life at Vidyashilp University.",
        "source": "volunteering"
    },

    # ── Scholarships & Awards ────────────────────────────────────
    {
        "content": "I have received several scholarships and awards: 1) 100% Scholarship at Vidyashilp University for B.Tech CSE. 2) Karta Initiative India Foundation Scholar. 3) Selected in Top 50 Students of Bihar by Bihar government. 4) Top 7 in Vibe Coding Hackathon at Vidyashilp University. 5) Top 10 in Agastya Foundation Innovation Challenge.",
        "source": "awards"
    },

    # ── Leadership ───────────────────────────────────────────────
    {
        "content": "I am involved in several leadership programs: 1) DexSchool Leadership Program focusing on building leadership and entrepreneurial skills. 2) Aspire Leaders Program for developing global leadership capabilities. These programs have helped me develop strategic thinking, team management, and visionary leadership qualities.",
        "source": "leadership"
    },

    # ── Extracurriculars ─────────────────────────────────────────
    {
        "content": "My extracurricular activities include playing Badminton, Cricket, and Table Tennis. I enjoy outdoor sports and physical activities. I also actively participate in hackathons, coding competitions, and tech community events.",
        "source": "extracurriculars"
    },
]


PROJECT_CHUNKS = [
    # ── Project: Cortex Lab (this project) ───────────────────────
    {
        "content": "📌 Project Name: Cortex Lab - Personal AI Memory Engine. This is an advanced RAG-based personal AI assistant that stores, retrieves, and reasons over personal memories. It uses a fine-tuned DeepSeek-R1-7B model, FAISS vector search with BGE-large-en-v1.5 embeddings, DuckDB for metadata, and a NetworkX knowledge graph. Built with FastAPI backend and Next.js frontend. Features include multi-agent orchestration, hybrid retrieval, streaming responses, and observability.",
        "source": "projects-repository"
    },

    # ── Project: Sysmind CLI ─────────────────────────────────────
    {
        "content": "📌 Project Name: Sysmind-CLI - System Intelligence & Automation CLI. A unified command-line utility providing intelligent system monitoring, process management, disk analytics, network diagnostics, and automated maintenance. Built with Python 3.8+ using only standard library (no external dependencies). Features include real-time CPU/memory monitoring, disk intelligence with duplicate detection, safe cleanup with quarantine and undo support, health scoring (0-100), and cross-platform support (Windows, Linux, macOS). GitHub: https://github.com/Suraj-creation/Sysmind-CLI",
        "source": "projects-repository"
    },

    # ── Project: Jarurat Care ────────────────────────────────────
    {
        "content": "📌 Project Name: Jarurat Care Foundation - Cancer Support Platform. A digital platform serving as the technological backbone for Jarurat Care Foundation, an NGO dedicated to supporting cancer patients and their families. Founded in memory of Rekha Joshi (1963-2023). Features include Hope AI Chatbot (24/7 cancer support assistant powered by Google Gemini 2.0 Flash), patient support request forms, volunteer coordination, and analytics dashboard. Impact: 150+ patients assisted, 54+ mentors, 28+ doctors, 2000+ community reach. Live at: https://jarurat-care-cyan.vercel.app. GitHub: https://github.com/Suraj-creation/Jarurat-Care",
        "source": "projects-repository"
    },

    # ── Project: EEG Alzheimer's Detection ───────────────────────
    {
        "content": "📌 Project Name: EEG-Based Alzheimer's Disease Classifier. A machine learning platform for automated classification of Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and Cognitively Normal (CN) individuals using resting-state EEG biomarkers. Uses OpenNeuro ds004504 dataset (88 subjects). Implements 438 biomarkers, Power Spectral Density analysis, non-linear dynamics metrics. Achieves 72% accuracy for dementia screening and 77.8% recall for AD detection. Built with Python, MNE-Python, LightGBM, XGBoost, Streamlit. Live demo: https://machine-learning-suraj-creation.streamlit.app/. GitHub: https://github.com/Suraj-creation/Machine_learning",
        "source": "projects-repository"
    },

    # ── Project: Healthcare AI Assistant ─────────────────────────
    {
        "content": "📌 Project Name: Healthcare AI Assistant - Intelligent Disease Prediction System. An AI-powered healthcare diagnostic assistant combining machine learning with Google Gemini 2.5 Flash AI. Features 41 disease models, 132 symptoms database with severity weighting (1-7 scale), smart symptom checker with multi-modal input (text, voice, interactive body map), personalized health recommendations, and privacy-first design with all data stored locally. Built with HTML5, JavaScript ES6+, Tailwind CSS, ECharts.js, Anime.js. Live at: https://healthcare-prediction.vercel.app/. GitHub: https://github.com/Suraj-creation/Healthcare_Prediction",
        "source": "projects-repository"
    },

    # ── Project: DL Course Platform ──────────────────────────────
    {
        "content": "📌 Project Name: DL Course Platform - Educational Website with Admin Panel. A full-stack educational platform (LMS) for course instructors to manage courses, lectures, assignments, and resources. Built with Node.js, Express.js, React.js, MongoDB, JWT authentication. Features include admin panel with 8 manager modules, file upload system, real-time updates, and responsive design. 58 Vercel deployments. GitHub: https://github.com/Suraj-creation/DL_course-Shabbeer.Basha",
        "source": "projects-repository"
    },

    # ── Project: Gemini Chat UI (ChatGPT Clone) ─────────────────
    {
        "content": "📌 Project Name: Gemini Chat UI (ChatGPT Clone). A production-quality web application replicating the ChatGPT interface using Google's Gemini API. Built with Next.js 14, TypeScript, React 18, Tailwind CSS. Features real-time token-by-token streaming, conversation management, model selection (Gemini 1.5 Flash/Pro), custom system instructions, markdown rendering with syntax highlighting, and dark theme. Live at: https://chatgpt-clone-taupe-one.vercel.app/. GitHub: https://github.com/Suraj-creation/chatgpt_clone",
        "source": "projects-repository"
    },

    # ── Project: Snake and Ladder Game ───────────────────────────
    {
        "content": "📌 Project Name: Snake and Ladder Game - AI-Powered Interactive Board Game. A modern interactive Snake and Ladder board game built with TypeScript, React, Vite, and Google Gemini AI. Dedicated to nephews Reesu and Reetu. Features smooth animations, responsive design, and AI-enhanced gameplay. Live at: https://snake-and-ladder-game-ten.vercel.app/. GitHub: https://github.com/Suraj-creation/Snake-and-Ladder-game",
        "source": "projects-repository"
    },

    # ── Project: Portfolio Finance Optimizer ──────────────────────
    {
        "content": "📌 Project Name: Portfolio Finance Optimizer - MPT Dashboard. An Excel-themed interactive portfolio analysis dashboard implementing Modern Portfolio Theory for optimal asset allocation. Analyzes NSE equity securities using SLSQP optimization. Achieved 15.6% Sharpe ratio improvement (1.192 to 1.377). Features Plotly.js visualizations, efficient frontier with Monte Carlo simulations, AI-powered insights via Google Gemini 1.5 Pro. Live at: https://portfolio-finance-optimal.vercel.app. GitHub: https://github.com/Suraj-creation/Portfolio_finance_Optimal",
        "source": "projects-repository"
    },

    # ── Project: ExplainBoard (Live Classroom) ───────────────────
    {
        "content": "📌 Project Name: ExplainBoard - Live Classroom powered by AI. An AI-powered visual learning whiteboard for interactive classroom experiences. Built with React, TypeScript, Vite, Tailwind CSS, Google Gemini API. Features dual modes (Live and Explain), real-time AI-generated explanations, classroom chalkboard aesthetic. Live at: https://live-classroom-powered-by-ai.vercel.app/. GitHub: https://github.com/Suraj-creation/Live_Classroom-powered_by_AI",
        "source": "projects-repository"
    },

    # ── Project: Image Captioning & Segmentation ─────────────────
    {
        "content": "📌 Project Name: Image Captioning & Segmentation. A production-quality Streamlit web application combining image captioning and image segmentation using COCO 2014 dataset. Implements ResNet50+LSTM, InceptionV3+Transformer for captioning, and Mask R-CNN, DeepLabV3+, U-Net for segmentation. Features batch processing, Docker support (CPU/GPU), BLEU/CIDEr metrics, developer mode. Built with Python, PyTorch, OpenCV, Streamlit. GitHub: https://github.com/Suraj-creation/Image_captioning_-_Segmentation",
        "source": "projects-repository"
    },

    # ── Project: Gemini CBSE Classroom ───────────────────────────
    {
        "content": "📌 Project Name: Gemini CBSE Classroom (Important_files). An AI-powered educational platform for CBSE curriculum learning. Built with FastAPI backend and React/Material-UI frontend. Features PDF upload and rendering with PDF.js, chat-based interaction with educational content, content expansion functionality, and Google Gemini API integration. GitHub: https://github.com/Suraj-creation/Important_files",
        "source": "projects-repository"
    },

    # ── Project: AI Note-Making Mobile App ───────────────────────
    {
        "content": "📌 Project Name: AI-Powered Note-Making Mobile App. An intelligent mobile note-taking application powered by Google Gemini AI. Features dual content model (raw thoughts + AI-polished versions), version history, automatic background sync every 12 hours, AI-suggested tags, automatic task extraction, offline-first architecture. Built with React, TypeScript, Vite, Google Gemini AI. Live at: https://ai-powered-notemaking-mobile-app.vercel.app/. GitHub: https://github.com/Suraj-creation/AI_powered_notemaking_mobile_app",
        "source": "projects-repository"
    },

    # ── Project: NotemakingAI (Thought Canvas) ───────────────────
    {
        "content": "📌 Project Name: NotemakingAI (Thought Canvas). An AI-powered note-taking Android application using Google Gemini AI. Built with Kotlin, Jetpack Compose, Material Design 3, Room Database, Retrofit, WorkManager. Features Clean Architecture with MVVM pattern, StateFlow for reactive UI, dual content model, version history with human vs AI attribution, background sync, and offline-first design. GitHub: https://github.com/Suraj-creation/NotemakingAI",
        "source": "projects-repository"
    },

    # ── Project: Echo Chamber Buster ─────────────────────────────
    {
        "content": "📌 Project Name: Echo Chamber Buster - Challenge Your Reasoning. An AI-powered adversarial debate platform that challenges beliefs through evidence-based philosophical sparring. Covers 40+ controversial topics across 8 domains. Uses Google Gemini AI to systematically dismantle arguments using authoritative sources (Socrates, Nietzsche, MLK, Gandhi, Einstein, Bhagavad Gita, Bible, Quran). Single HTML file with zero dependencies. Live at: https://challenge-your-reasoning.vercel.app/. GitHub: https://github.com/Suraj-creation/Challenge_your_Reasoning",
        "source": "projects-repository"
    },
]


VISION_CHUNKS = [
    # ── Vision: Education Transformation ─────────────────────────
    {
        "content": "My deepest pursuit is to fundamentally transform the architecture of education by harnessing exponential AI, agentic intelligence, AR/VR immersion, game-engine interactivity, and human-centered design. I envision a seamlessly integrated learning ecosystem where humans and intelligent systems co-create knowledge. The core principle is that when building blocks of knowledge are sequenced optimally at the earliest stages, we can cultivate entirely new ways of thinking. The ultimate goal is an accelerated pathway to breakthrough innovation, sustainable progress, and human flourishing.",
        "source": "vision-education"
    },
    {
        "content": "I am building a multi-agentic AI tutoring ecosystem designed to radically restructure how students learn. It begins with a simple gateway where learners select their class (Grades 1-12). Two pathways are offered: Create a Chapter (structured, persistent learning space) and Learn (flexible, on-demand mode). Learners can upload books which are transformed into comprehensive learning blueprints. The core agent generates powerful abstract overviews and hierarchical step-by-step To-Do Lists. Specialized sub-agents support immersive learning, contextual explanations, interdisciplinary connections, note synthesis, and research guidance. This is for the SIH (Smart India Hackathon) 2025 Problem Statement ID 25140.",
        "source": "vision-education"
    },
    {
        "content": "My three core visions for education are: 1) Advancing the Learning Curve - collapse the time-cost of learning through agentic, hyper-efficient knowledge delivery. 2) Restructuring the Purpose and Definition of Education - deliver knowledge in vector-based, synthesized, multidimensional format, integrate epistemology and critical thinking at earliest stages. 3) Revolutionizing Human Thought and Pursuits - align education with grand challenges like sustainability, peace, innovation, and discovery. Build an ecosystem where research, technology, and education co-evolve.",
        "source": "vision-education"
    },

    # ── Vision: Technology Transformation ────────────────────────
    {
        "content": "My second core life vision is to simplify and redefine the core of technology. Today's technology depends heavily on unsustainable extraction of Earth's resources, steep financial costs, and long learning curves that create monopolies. I imagine replacing the very base of computation, currently built on 0s and 1s, with something more sustainable, democratic, and liberating. This radical shift would democratize access to advanced technology and allow for harmonious advancement of technology that aligns with nature and humanity.",
        "source": "vision-technology"
    },

    # ── Vision: Startup Ideas ────────────────────────────────────
    {
        "content": "My startup ideas include: 1) Advanced Agentic Code Editor like Cursor AI but with split-screen teaching mode that explains code logic in conversational English. 2) A new conversational programming language that makes coding as intuitive as human conversation. 3) Building an identical AI agent that knows the user better than anyone - a personal mentor, companion, motivator, and personalized teacher that evolves into a lifelong collaborator. 4) An institute committed to solving all problems of this world through deeply researched foundations and continuous learning.",
        "source": "vision-startups"
    },

    # ── Vision: Life Philosophy ──────────────────────────────────
    {
        "content": "My core life philosophy: I believe the true purpose of education is not repetition but reinvention towards innovation, peace, and breakthroughs. Education should define your life - everything from birth requires understanding and knowledge. I am committed to learning through deep work and generational work. My inspiration comes from leaders like Sir Ratan Tata. I believe India is uniquely positioned to redefine education for humanity through its diverse thought traditions including Upanishads, Vedas, Yoga, and spiritual sciences combined with exponential technologies.",
        "source": "vision-philosophy"
    },

    # ── Manifesto ────────────────────────────────────────────────
    {
        "content": "I authored a Manifesto for the Future of Education titled 'Redefining Learning, Life, and the Purpose of Human Existence'. The manifesto argues that today's education system fails to shape individuals of character, wisdom, and vision despite consuming decades of life. It proposes education as the evolution of life and consciousness, where every act of learning leads to reflection, relation, and realization. The ultimate goal is not to produce graduates but to cultivate creators, thinkers, and visionaries who can rebuild systems based on understanding, equity, and truth.",
        "source": "vision-manifesto"
    },
]


# ── Personal facts for quick retrieval ───────────────────────────
QUICK_FACTS = [
    {
        "content": "Suraj Kumar's salary expectation: I do not have a fixed salary expectation. I am currently a B.Tech student (2023-2027) and have not discussed salary requirements.",
        "source": "personal-facts"
    },
    {
        "content": "Suraj Kumar has not pursued a PhD. He is currently in his B.Tech undergraduate program (2023-2027) at Vidyashilp University.",
        "source": "personal-facts"
    },
    {
        "content": "Suraj Kumar has not published any research papers yet. He is currently an undergraduate student focused on building projects and gaining practical experience.",
        "source": "personal-facts"
    },
    {
        "content": "Suraj Kumar is not married. He is a young undergraduate student currently focused on his education and career development.",
        "source": "personal-facts"
    },
    {
        "content": "The Hope chatbot is a key feature of the Jarurat Care Foundation platform. Hope is a 24/7 cancer support assistant powered by Google Gemini 2.0 Flash with cancer-specific knowledge. It provides emotional support, treatment navigation information, and connects patients with resources. It was built as part of the Jarurat Care project by Suraj Kumar. The Hope chatbot serves cancer patients and their families through the platform at jarurat-care-cyan.vercel.app.",
        "source": "projects-repository"
    },
]


ALL_CHUNKS = PERSONAL_INFO_CHUNKS + PROJECT_CHUNKS + VISION_CHUNKS + QUICK_FACTS


async def main():
    print("=" * 70)
    print("  CLEAN DATA INGESTION FOR CORTEX LAB")
    print("=" * 70)
    print(f"\n  Total chunks to ingest: {len(ALL_CHUNKS)}")
    print(f"    - Personal info: {len(PERSONAL_INFO_CHUNKS)}")
    print(f"    - Projects: {len(PROJECT_CHUNKS)}")
    print(f"    - Vision/Ideas: {len(VISION_CHUNKS)}")
    print(f"    - Quick facts: {len(QUICK_FACTS)}")
    print()

    # Initialize components
    print("  Initializing embedding model...")
    embedding_model = EmbeddingModel()

    print("  Initializing vector store...")
    vector_store = VectorStore(dimension=embedding_model.dimension, data_dir="data/vectors")

    print("  Initializing metadata store...")
    metadata_store = MetadataStore(db_path="data/cortex.duckdb")

    print("  Initializing knowledge graph...")
    knowledge_graph = KnowledgeGraph(data_dir="data/graph")

    # Create a minimal LLM (no model loading needed for ingestion)
    print("  Initializing LLM (minimal, no model load)...")
    llm = LocalLLM.__new__(LocalLLM)
    llm.model = None
    llm.tokenizer = None
    llm.model_path = ""
    llm._generation_lock = None
    llm._loaded = False

    # Create pipeline
    pipeline = MemoryIngestionPipeline(
        llm=llm,
        embedding_model=embedding_model,
        vector_store=vector_store,
        metadata_store=metadata_store,
        knowledge_graph=knowledge_graph,
    )

    # Ingest all chunks
    print("\n" + "─" * 70)
    print("  Starting ingestion...")
    print("─" * 70)

    success = 0
    failed = 0
    for i, chunk in enumerate(ALL_CHUNKS):
        try:
            memory = await pipeline.ingest(
                content=chunk["content"],
                session_id="clean-ingest-v2",
                source=chunk["source"],
            )
            if memory:
                success += 1
                print(f"  ✅ [{i+1}/{len(ALL_CHUNKS)}] {chunk['source']}: {chunk['content'][:60]}...")
            else:
                failed += 1
                print(f"  ❌ [{i+1}/{len(ALL_CHUNKS)}] REJECTED: {chunk['content'][:60]}...")
        except Exception as e:
            failed += 1
            print(f"  ❌ [{i+1}/{len(ALL_CHUNKS)}] ERROR: {e}")

    # Save all stores
    print("\n" + "─" * 70)
    print("  Saving stores...")
    vector_store.save()
    knowledge_graph.save()
    print("─" * 70)

    print(f"\n  ✅ INGESTION COMPLETE")
    print(f"     Succeeded: {success}")
    print(f"     Failed:    {failed}")
    print(f"     Vectors:   {vector_store.count()}")
    print(f"     Graph:     {knowledge_graph.get_stats()}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
