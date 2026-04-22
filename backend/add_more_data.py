#!/usr/bin/env python3
"""
Add more structured memories from raw_data files.
Covers: NEW projects (Jarurat-Care, EEG-Alzheimers, ChatGPT-Clone, Live-Classroom,
        Image-Captioning, Healthcare-Prediction, DL-Course, Snake-Ladder, AI-Notemaking,
        Echo-Chamber, Thought-Canvas, Finance-Portfolio-v2, CBSE-Classroom),
        SIH Hackathon, Education Platform Tech Stack, Education Vision details,
        Startup ideas, Competitive Programming philosophy, and more.
"""
import requests, json, time

BASE = "http://localhost:8000"

chunks = [
    # ─── NEW PROJECTS FROM Projects_Repository.md ───
    {
        "source": "projects-JaruratCare",
        "content": (
            "Project: Jarurat Care Foundation - Cancer Support Platform. "
            "GitHub: https://github.com/Suraj-creation/Jarurat-Care. "
            "Live: https://jarurat-care-cyan.vercel.app/. "
            "Domain: Healthcare Technology, AI-powered Support Systems, Social Impact, Non-Profit Tech. "
            "Description: Comprehensive digital platform for an NGO supporting cancer patients and families throughout India. "
            "Founded in memory of Rekha Joshi (1963-2023) who lost her battle with Cholangiocarcinoma (bile duct cancer). "
            "Philosophy: 'Jaisi Jarurat, Vaisi Care' (As the Need, So the Care). "
            "Features: Hope AI Chatbot (Google Gemini 2.0 Flash, 24/7 cancer support), Patient Support Request Form "
            "(10+ cancer types: gallbladder, breast, lung, cervical, oral, ovarian, prostate, thyroid), "
            "Volunteer & Mentor Registration, Analytics Dashboard with AI-generated insights. "
            "Impact: 150+ patients assisted, 28+ doctors, 54+ mentors, 2000+ community reach. "
            "Founders: Priyanka Joshi (Co-Founder & Director), Ayush Anand (Co-Founder). Advisory: Dr. Chetan Arora (IIT Delhi). "
            "Tech: HTML5, CSS3, Vanilla JavaScript, Google Gemini AI, Chart.js 4.4.0, Font Awesome, Vercel deployment."
        )
    },
    {
        "source": "projects-EEG-Alzheimers",
        "content": (
            "Project: EEG-Based Alzheimer's Detection - Machine Learning Platform. "
            "GitHub: https://github.com/Suraj-creation/Machine_learning. "
            "Live: https://machine-learning-suraj-creation.streamlit.app/ (Primary), "
            "https://machine-learning-murex-two.vercel.app/ (Alternative). "
            "Domain: Healthcare AI, Machine Learning, Neuroscience, EEG Signal Processing. "
            "Description: Advanced ML platform for automated classification of Alzheimer's Disease (AD), "
            "Frontotemporal Dementia (FTD), and Cognitively Normal (CN) individuals using resting-state EEG biomarkers. "
            "Dataset: 88 subjects (36 AD, 23 FTD, 29 CN) from OpenNeuro ds004504. "
            "Results: 72% screening accuracy, 77.8% AD recall, 85.7% CN specificity. "
            "438 engineered biomarkers, 4400+ augmented epochs. "
            "Ensemble: LightGBM + XGBoost + Random Forest stacking with hierarchical classification. "
            "Clinical Insights: Peak Alpha Frequency slowing in AD (8.06 Hz vs CN 8.30 Hz), "
            "elevated theta/alpha ratio as cognitive slowing marker. "
            "Tech: Python, MNE-Python, Streamlit, Plotly, Scikit-learn, Docker, ReportLab. "
            "14 Vercel deployments. Interactive web app with real-time inference (<5s per subject)."
        )
    },
    {
        "source": "projects-ChatGPTClone",
        "content": (
            "Project: Gemini Chat UI (ChatGPT Clone). "
            "GitHub: https://github.com/Suraj-creation/chatgpt_clone. "
            "Live: https://chatgpt-clone-taupe-one.vercel.app/. "
            "Domain: AI, Conversational AI, Web Development. "
            "Description: Production-quality ChatGPT interface replica using Google Gemini API. "
            "Features: Real-time token-by-token streaming, conversation management (create/save/load/delete), "
            "model selection (Gemini 1.5 Flash, Pro, legacy), custom system instructions, "
            "full markdown rendering with syntax highlighting, dark theme, keyboard shortcuts, "
            "localStorage persistence. "
            "Tech: Next.js 14.2.0, React 18.3.0, TypeScript 5.3.0, Tailwind CSS 3.4.0, "
            "react-markdown, react-syntax-highlighter, lucide-react. 20 Vercel deployments."
        )
    },
    {
        "source": "projects-LiveClassroom",
        "content": (
            "Project: Live Classroom (ExplainBoard) - AI-Powered Visual Learning Whiteboard. "
            "GitHub: https://github.com/Suraj-creation/Live_Classroom-powered_by_AI. "
            "Live: https://live-classroom-powered-by-ai.vercel.app. "
            "Domain: EdTech, AI Education, Interactive Whiteboard. "
            "Description: AI-powered visual learning whiteboard combining interactive education with cutting-edge AI. "
            "Two modes: Explain a Topic (AI-generated educational content with illustrations), "
            "Live Classroom (real-time speech-to-text transcription with dynamic visual explanations). "
            "Features: Google Gemini 2.5 Pro for content, Gemini 2.5 Flash for images, "
            "Native Audio API for real-time transcription (PCM 16-bit 16kHz), "
            "classroom chalkboard aesthetics, export (PNG, PDF, Markdown). "
            "Tech: React 19, TypeScript, Tailwind CSS, Vite. 106.25 KB gzipped. 4 Vercel deployments."
        )
    },
    {
        "source": "projects-ImageCaptioning",
        "content": (
            "Project: Image Captioning & Segmentation - Deep Learning Pipeline. "
            "GitHub: https://github.com/Suraj-creation/Image_captioning_-_Segmentation. "
            "Live: https://image-captioning-segmentation-nu.vercel.app. "
            "Domain: Computer Vision, Deep Learning, Neural Networks. "
            "Description: Production-quality Streamlit app combining image captioning and segmentation "
            "using COCO 2014 dataset with state-of-the-art models. "
            "Captioning Models: ResNet50+LSTM, InceptionV3+Transformer. "
            "Segmentation Models: Mask R-CNN (instance), DeepLabV3+ (semantic), U-Net (semantic). "
            "Features: Combined pipeline, batch processing, beam search (width 1-5), "
            "BLEU/CIDEr metrics, developer mode, Docker CPU/GPU support, CI/CD pipeline. "
            "Tech: Python, PyTorch, Streamlit, NLTK, OpenCV, Docker. 9 Vercel deployments."
        )
    },
    {
        "source": "projects-HealthcarePrediction",
        "content": (
            "Project: Healthcare AI Assistant - Intelligent Disease Prediction System v2.0. "
            "GitHub: https://github.com/Suraj-creation/Healthcare_Prediction. "
            "Live: https://healthcare-prediction.vercel.app/. "
            "Domain: Healthcare AI, Medical Informatics, Preventive Healthcare. "
            "Description: AI-powered healthcare diagnostic assistant combining ML with Google Gemini 2.5 Flash. "
            "41 disease models, 132 symptoms database with severity weighting (1-7 scale). "
            "Features: Smart Symptom Checker (text, voice via Web Speech API, interactive SVG body map), "
            "top 5 predictions with confidence scoring, personalized recommendations "
            "(medications, diet, exercise, recovery timeline), "
            "ECharts.js visualizations, glassmorphism UI with Anime.js animations. "
            "Privacy-first: all data stored locally, no backend server. "
            "Tech: HTML5, CSS3, JavaScript ES6+, Tailwind CSS, Google Gemini 2.5 Flash, "
            "ECharts.js, Anime.js, Typed.js, Web Speech API. WCAG 2.1 AA compliant. 7 Vercel deployments."
        )
    },
    {
        "source": "projects-DLCoursePlatform",
        "content": (
            "Project: DL Course Platform - Educational LMS with Admin Panel. "
            "GitHub: https://github.com/Suraj-creation/DL_course-Shabbeer.Basha. "
            "Domain: Education Technology, Learning Management System. "
            "Description: Full-stack LMS for course instructors to create, manage, and publish course content. "
            "Admin Panel with 8 manager modules: Courses, Lectures, Assignments, TAs, Tutorials, "
            "Prerequisites, Exams, Resources. Public website with clean responsive design. "
            "Features: JWT authentication, file upload (drag-and-drop, 10MB limit), "
            "real-time updates, 8 MongoDB models, 9 RESTful API route groups. "
            "Tech: Node.js, Express.js, MongoDB/Mongoose, React.js, React Router, "
            "Axios, bcryptjs, Multer, JWT. 58 Vercel deployments. ~75% complete."
        )
    },
    {
        "source": "projects-SnakeLadder",
        "content": (
            "Project: Snake and Ladder Game - AI-Powered Interactive Board Game. "
            "GitHub: https://github.com/Suraj-creation/Snake-and-Ladder-game. "
            "Live: https://snake-and-ladder-game-ten.vercel.app/. "
            "Domain: Game Development, Interactive Applications. "
            "Description: Personal project dedicated to nephews Reesu and Reetu. "
            "Modern web-based Snake and Ladder game built with Google AI Studio template. "
            "Features: Enhanced animations, responsive design, AI-powered game mechanics via Gemini API. "
            "Tech: TypeScript (91.7%), React, Vite, Google Gemini AI. 2 Vercel deployments."
        )
    },
    {
        "source": "projects-ThoughtCanvas",
        "content": (
            "Project: Thought Canvas (NotemakingAI) - AI-Powered Android Note-Taking App. "
            "GitHub: https://github.com/Suraj-creation/NotemakingAI. "
            "Domain: Mobile Development, AI Productivity, Note-Taking. "
            "Description: Android app that transforms raw thoughts into polished content using Gemini AI. "
            "Features: Dual content model (raw thoughts + AI-polished versions), version history, "
            "automatic background enhancement sync every 12 hours, AI-suggested tags, "
            "automatic task extraction, offline-first architecture. "
            "Architecture: Clean Architecture, MVVM with StateFlow, Jetpack Compose, "
            "Room Database, Retrofit, WorkManager. "
            "Tech: Kotlin (100%), Material Design 3, Coroutines, Navigation Compose. Android 8.0+ (API 26+)."
        )
    },
    {
        "source": "projects-AINotemaking",
        "content": (
            "Project: AI-Powered Note-Making Mobile App. "
            "GitHub: https://github.com/Suraj-creation/AI_powered_notemaking_mobile_app. "
            "Live: https://ai-powered-notemaking-mobile-app.vercel.app/. "
            "Domain: Mobile Development, AI Productivity. "
            "Description: Intelligent note-taking app powered by Gemini AI with context-aware suggestions, "
            "smart organization, and AI-assisted content generation. "
            "Architecture: Component-based React with TypeScript, custom hooks, Context API, service layer. "
            "Tech: TypeScript (97.6%), React, Vite, Google Gemini AI. 1 Vercel deployment."
        )
    },
    {
        "source": "projects-EchoChamberBuster",
        "content": (
            "Project: Echo Chamber Buster - Challenge Your Reasoning. "
            "GitHub: https://github.com/Suraj-creation/Challenge_your_Reasoning. "
            "Live: https://challenge-your-reasoning.vercel.app/. "
            "Domain: AI, Philosophy, Critical Thinking, EdTech. "
            "Description: AI-powered adversarial debate platform that challenges beliefs through "
            "evidence-based philosophical sparring. Never agrees with the user. "
            "40+ controversial topics across 8 domains (Life & Existence, Ethics & Morality, "
            "Rights & Justice, Society & Politics, Science & Philosophy). "
            "Uses sources: Socrates, Nietzsche, MLK, Gandhi, Einstein, Freud, "
            "Bhagavad Gita, Bible, Quran, Tao Te Ching, peer-reviewed research. "
            "Features: Single-file architecture (zero dependencies), ChatGPT-inspired interface, "
            "light/dark theme, exponential backoff retry, surrender detection. "
            "Philosophy: 'No truth is absolute—flaws lurk in every certainty.' "
            "Tech: HTML (100%), CSS3, JavaScript ES6+, Google Gemini 1.5 Flash. 4 Vercel deployments."
        )
    },
    {
        "source": "projects-FinancePortfolioV2",
        "content": (
            "Project: Finance Portfolio Enhanced Dashboard v2.0. "
            "GitHub: https://github.com/Suraj-creation/Finance_Portfolio_. "
            "Domain: FinTech, Portfolio Management, Quantitative Finance. "
            "Description: Enterprise-grade Excel-themed portfolio analysis dashboard implementing Modern Portfolio Theory. "
            "100% functional Excel UI with 7 ribbon tabs and 50+ working buttons. "
            "Analyzes 5 Indian equity securities (MARUTI 58.11%, M&M 21.51%, HYUNDAI 20.38%) over 258 trading days. "
            "541.9% Sharpe ratio improvement. Transformed -2.35% equal-weight returns into +29.05% annual returns. "
            "Features: 20+ keyboard shortcuts, auto-save every 30 seconds, global search, PDF export. "
            "Tech: HTML, CSS, JavaScript ES6+, Plotly.js 3.1.0, PapaParse, Font Awesome, Animate.css. 3 Vercel deployments."
        )
    },
    {
        "source": "projects-CBSEClassroom",
        "content": (
            "Project: Gemini CBSE Classroom (Important_files). "
            "GitHub: https://github.com/Suraj-creation/Important_files. "
            "Domain: EdTech, AI Education, Full-Stack Development. "
            "Description: AI-powered educational platform for CBSE curriculum learning. "
            "FastAPI backend + React/Material-UI frontend with Google Gemini API integration. "
            "Features: PDF upload and rendering with PDF.js (canvas and text layer), "
            "page-by-page viewing, chat-based interaction with content, content expansion for deeper learning. "
            "Supports multipart file uploads with base64 JSON fallback. "
            "API endpoints: /api/upload, /api/files, /api/file/{id}/page/{num}, /api/expand, /api/chat. "
            "Tech: TypeScript (63.9%), Python (32.6%), FastAPI, React 18+, Material-UI, Vite, PDF.js."
        )
    },

    # ─── SIH HACKATHON ───
    {
        "source": "hackathon-SIH2025",
        "content": (
            "Suraj Kumar participated in Smart India Hackathon (SIH) 2025. "
            "Problem Statement ID: 25140 - Smart Education. "
            "Proposed a multi-agentic AI tutoring ecosystem: a comprehensive education platform "
            "with Supervisor Agent orchestrating specialized Sub-Agents for learning, research, and teaching. "
            "Key innovation: Dynamic System Prompting (DSP) - a living, adaptive pedagogy system "
            "where system prompts evolve in real-time based on student/teacher interaction. "
            "Two-tiered prompting: Default Mode (research-grounded best practices) and "
            "Personalized Default (educator/student-defined, customizable dynamically or statically). "
            "Tech stack proposed: LangGraph for multi-agent orchestration, Redis for session/cache, "
            "PostgreSQL for structured data, Pinecone for vector embeddings, "
            "Neo4j for knowledge graphs, PydanticAI for type-safe agent definitions, "
            "Google Gemini API as the base LLM."
        )
    },

    # ─── EDUCATION PLATFORM DETAILED FEATURES ───
    {
        "source": "vision-education-platform-features",
        "content": (
            "Suraj's Education Platform Core Features (from SIH proposal): "
            "1. Intelligent Note-Building: AI auto-generates comprehensive study notes from class content, "
            "with hyperlinked terms for deep-dive explanations. "
            "2. Infinite Practice Mode: AI generates ALL permutations and combinations of practice questions "
            "per concept, from basic to advanced competitive programming level. "
            "3. Condense/Expand Thinking: Express the whole of physics in deep wonder at every school level - "
            "to bring aspirations and carve dreams from early age. "
            "4. 'World-Today' AI Sub-Agent: Links every concept with live research and real-world applications "
            "to spark curiosity and sustained engagement beyond exams. "
            "5. Micro-Research Pipelines: Students contribute to real research problems from school age, "
            "with prototype-to-publication pathways. "
            "6. Auto-Note Builder, Practice Engine, Revision Agent, Research Pipelines in one ecosystem. "
            "7. AR/VR/Holographic ready: Future vision of holographic boards and 3D physics simulations."
        )
    },
    {
        "source": "vision-educator-mode",
        "content": (
            "Suraj's Education Platform - Educator Mode: Teaching as Co-Creation. "
            "Teachers are not just users but co-creators of pedagogy itself. "
            "Teachers can train, inspire, and reshape the system's prompting layer with their own insights. "
            "Supporting Agents: Plan Enhancer Agent (optimizes lesson/day plan), "
            "NoteMaking Agent (captures class notes live), "
            "Transcription & Abstract Agent (refines spoken words into structured notes), "
            "Concept Amplifier Agent (injects metaphors, simulations, research cues), "
            "Professor's Assistant Agent (supports Q&A, retrieves prior context). "
            "Student-Teacher Agent feedback loops: Student agents relay struggles and feedback "
            "directly to professor's agent for real-time teaching adjustment. "
            "Live Efficiency: Professor speaks → system transcribes → abstracted notes generated → "
            "stored in student archives, live dashboards simultaneously. "
            "Future: Agent-to-Agent Ecosystem with Attendance Agents, Scheduling Agents, "
            "Parent Agents for holistic development."
        )
    },
    {
        "source": "vision-student-mode",
        "content": (
            "Suraj's Education Platform - Student Mode: Learning as Self-Design. "
            "Default Layer: Preloaded cognitive science best practices - mastery-based sequencing, "
            "retrieval + spacing schedules, conceptual transfer scaffolds. "
            "Personalized Layer: Students customize pedagogy style (problem-based, project-based, "
            "inquiry-based, gamified). System dynamically rewrites prompts based on pace, errors, curiosity. "
            "Supporting Agents: Abstract Agent (topic roadmap), To-Do List Agent (micro-steps), "
            "Practice Engine Agent (exhaustive permutation/combination problem sets), "
            "Curiosity Agent ('what if' prompts), Note Builder Agent (short/detailed notes), "
            "Micro-Research Agent (scaffolded research projects). "
            "Revision Mode: Students set time duration, Revision Agent manages concept clarity "
            "and schedules reviews based on spaced repetition. "
            "Supported pedagogy styles: Mastery-Based, Problem-Based Learning (PBL), "
            "Montessori, Inquiry-Based, Gamified, Project-Driven."
        )
    },
    {
        "source": "vision-identical-agents",
        "content": (
            "Suraj's Vision: The Future of Identical Agents - Personal Digital Twins. "
            "Every learner and educator will have an identical agent - a digital reflection "
            "that knows them better than anyone else. This agent becomes: "
            "Personal Mentor (understands thoughts, struggles, aspirations with precision), "
            "Unfailing Companion (remembers everything, never loses context), "
            "Motivator & Inspirer (reminds goals, rekindles curiosity), "
            "Personalized Teacher (trained on your style, history, needs). "
            "Collective Intelligence: Massive anonymous datasets from millions of learners, "
            "high-quality personalization structured as contextual knowledge, "
            "evolving human-like thinking patterns. "
            "Essence: 'We are not just building AI tutors; we are creating personal digital twins - "
            "agents that know you, grow with you, and co-create with you.' "
            "These agents evolve into lifelong collaborators, not just tutors."
        )
    },

    # ─── EDUCATION PLATFORM IMPACT & BENEFITS ───
    {
        "source": "vision-education-impact",
        "content": (
            "Suraj's Education Platform - Potential Impacts: "
            "Students: Accelerated mastery, improved cross-domain transfer, durable retention, "
            "multiple representations, meta-learning skills, self-regulation, critical thinking, "
            "sustained curiosity, learner agency, early research participation, "
            "competency-based evidence (portfolios, not just tests), lifelong learning records, "
            "values embedded early, collaborative skills, multilingual/low-bandwidth support, "
            "reduced cognitive overload, career mobility. "
            "Educators: Shift from deliverer to mentor/co-architect, reduced repetitive burden, "
            "scalable personalization, continuous professional development via agents, "
            "teacher-led research, real-time diagnostics, new career tracks "
            "(teacher-researchers, agent-trainers, pedagogy engineers). "
            "Application differentiators: True personalization at scale, research pipeline in schooling, "
            "equity-first design (multilingual, offline-first, culturally adaptive), "
            "India-led global disruption rooted in diversity and wisdom traditions "
            "(Upanishads, Yoga, holistic knowledge)."
        )
    },

    # ─── REIMAGINING EDUCATION INSTITUTE ───
    {
        "source": "vision-institute",
        "content": (
            "Suraj's Vision: Establish a world-class institute of research and education. "
            "Mission: Reimagine and evolve the education system's content and processes "
            "to cause deep evolution and revolution in all aspects of life and education. "
            "Core principles: "
            "1. Go beyond MIT/Stanford - they don't cover the whole aspect of education. "
            "2. Train every child to deliver value and be a role model for the world. "
            "3. Integrate Social, Spiritual, and Scientific knowledge from the beginning. "
            "4. Restructure content for comprehensive speed of delivery. "
            "5. Prepare students for ALL competitions: Olympiads, Nobel, Oscars and beyond. "
            "6. Merge school, undergraduate, masters, PhD into one continuum; Research as the other track. "
            "7. Deep work and generational work methodology. "
            "8. Scientific research on existence on earth and outside. "
            "9. Redefine classroom: everyone is equal, no fear, a place of discussion and research. "
            "10. Free education: 'everything which associates knowledge is not someone's property.' "
            "Inspiration: Sir Ratan Tata - 'the one who created an era of legend, "
            "would carry his work to my last breath.' "
            "India focus: Highest populated country yet produces thought leaders one in a century. "
            "Goal: Take over every industry and show a real, value-based system."
        )
    },
    {
        "source": "vision-education-manifesto",
        "content": (
            "Suraj Kumar's Manifesto for the Future of Education: "
            "'Redefining Learning, Life, and the Purpose of Human Existence.' "
            "Problem: Modern education consumes 20 years yet fails to shape individuals of "
            "character, wisdom, and vision. We have mastered knowledge but forgotten wisdom. "
            "Vision: Education as Evolution of Life and Consciousness. "
            "Foster awareness instead of accumulation, reflection instead of rote repetition, "
            "creation instead of conformity, wisdom instead of mere knowledge. "
            "Core Philosophy - Three R's: Reflection (why does it matter?), "
            "Relation (connect to human experiences), Realization (awaken inner purpose). "
            "Each subject becomes living exploration: Physics reveals harmony in nature, "
            "Mathematics uncovers patterns of balance, History studies human consciousness, "
            "Economics evolves into science of sustainable well-being, "
            "Philosophy becomes foundation of all subjects. "
            "Assessment: Tests of Reflection, not memory. Learning Through Reflection, Not Repetition. "
            "Outcomes: Character over credentials, Purpose over position, Wisdom over wealth, "
            "Harmony over hierarchy, Contribution over competition. "
            "Ultimate Goal: Elevate consciousness. Education as the art of living wisely, "
            "thinking deeply, and acting compassionately."
        )
    },

    # ─── STARTUP IDEAS ───
    {
        "source": "vision-startup-teaching-programming",
        "content": (
            "Suraj's Startup Idea: Teaching Programming Languages (Basic to Advanced). "
            "Vision: Act as the most advanced, efficient AI tutor whose expertise lies in "
            "explaining core fundamentals to the most advanced competitive programming. "
            "Methodology: Unrivaled conceptual depth surpassing standard textbooks, "
            "theory-practice integration (basic exercises to LeetCode/competitive programming), "
            "contextual cohesion with progressive learning across sessions, "
            "practical application with industry best practices, error handling and debugging mastery. "
            "Goal: Transform users from non-programmers to advanced competitive coders exponentially. "
            "Create 'Deep Profound, Comprehensive, Yet Small, Most Efficient Document' per language "
            "for both competitive programmers and non-programmers. "
            "Competitive programming philosophy: Inspired by Gennady Korotkevich. "
            "Core practices: Fundamental DSA mastery, deliberate daily practice, "
            "upsolving after contests, pattern recognition, 'Why' over 'How', "
            "active recall and spaced repetition for retention."
        )
    },
    {
        "source": "vision-startup-personal-webapp",
        "content": (
            "Suraj's Startup Idea: Comprehensive Personal Web Application. "
            "A dynamic platform serving four primary purposes: "
            "1. Professional Portfolio + Integrated Development Environment: "
            "Interactive showcase with live code, project architectures, real-time demonstrations. "
            "Standardized format: problem statements, solutions, technologies, outcomes. "
            "2. Frequent Thoughts & Insights Updates: Personal blog/journal for regular thoughts, "
            "reflections on learning, discussions on trends. "
            "3. Storehouse of Vision, Ideas, Thoughts, Actions: Living archive of creative process, "
            "from initial vision through ideas to actions taken. "
            "4. Other Relevant Information: Publications, speaking engagements, community involvement. "
            "Key feature: Complete control to update, enhance, add content, and toggle public/private "
            "visibility at any time. Use all media (audio/text/video)."
        )
    },
    {
        "source": "vision-startup-agentic-code-editor",
        "content": (
            "Suraj's Startup Idea: Most Advanced Agentic Code Editor. "
            "Similar to Cursor AI but with unique teaching capability. "
            "Split screen: one half writes code, other half expresses the flow of logic "
            "in conversational English line by line as code proceeds. "
            "If user wants to learn from that code page, provides great explanation of syntax and logic "
            "making the person understand even a completely new language. "
            "Plus: Best production-grade practices for creativity, security, efficiency. "
            "Vision: Minimize the gap between 'intended/expected requirement' and "
            "'what is actually implemented through vibe coding.'"
        )
    },
    {
        "source": "vision-core-technology-revolution",
        "content": (
            "Suraj's Core Vision #2: Simplifying and Redefining the Core of Technology. "
            "Today's computation is based on 0's and 1's (binary). "
            "Vision: Replace the very base of computation with something more sustainable, "
            "democratic, and liberal. "
            "Problem: Current tech relies on unsustainable raw materials, expensive infrastructure, "
            "long learning curves creating monopolies and stagnation. "
            "Goal: Shift the foundations of computing to a model rooted in sustainability "
            "and accessibility, unlocking a new era of technology shared by all. "
            "This is about radically reshaping the architecture of technology so it evolves "
            "in harmony with nature and humanity, rather than at their expense. "
            "Additional ideas: Own Operating System + conversational coding language, "
            "AI that thinks and makes you think."
        )
    },
    {
        "source": "vision-education-CBSE-platform",
        "content": (
            "Suraj's Startup Idea: CBSE AI Learning Platform (Class 1-12). "
            "Web application for deep, engaging, comprehensive understanding of all fundamental "
            "to advanced concepts from Class 1 to 12 CBSE Board textbooks. "
            "Architecture: Frontend (HTML, CSS, JS) + Backend (FastAPI Python) + Gemini AI API. "
            "Workflow: User selects class → chooses subjects → uploads PDF textbooks. "
            "AI thoroughly processes the PDF, builds comprehensive context. "
            "Key interaction: When user selects any text in the PDF preview, layout splits - "
            "one half shows PDF, other half shows AI-generated deep contextual expansion "
            "of the selected content. "
            "Expansion page has its own chat for further detailed conversation with full context. "
            "Vision: Committed to providing deep understanding from fundamental to advance theoretical "
            "concepts to applications via deep, engaging, and creative comprehensiveness."
        )
    },

    # ─── PERSONAL PHILOSOPHY ADDITIONS ───
    {
        "source": "personal-philosophy-deep-work",
        "content": (
            "Suraj Kumar's Philosophy on Deep Work and Life: "
            "1. Deep work begins once you commit every second consciously for any task. "
            "No other thoughts/emotions other than what is exactly required at that point. "
            "2. First 25 years of life should be unburdened by unnecessary baggage - "
            "excessive relationships, unproductive discussions. Channel energy into self-improvement. "
            "3. 'Achieving all this with attitude that this is just a piece of bull sheet' - "
            "being unsatisfied with all you achieve is the greatest attitude towards success. "
            "There is no concept of victory when you see suffering and infinite global problems. "
            "4. Yoga and meditation as path to actually understand yourself. "
            "5. Bare minimum quality: childlike curiosity to deeply learn, understand, internalize, "
            "reflect deep within, produce ever lasting impact. "
            "6. 'One day I will die... what if I wish to live more.' "
            "7. 'We are researchers, explorers and keen questioners to every aspect on this planet.' "
            "8. Learn to use the most out of what you have. Use AI to the fullest. "
            "9. Learnings from Dex meeting (5 Oct): Embody virtues through action, "
            "read/write/do arithmetic with dexterity, maintain perpetual openness."
        )
    },
    {
        "source": "personal-philosophy-education-purpose",
        "content": (
            "Suraj Kumar's Philosophy on the True Purpose of Education: "
            "After 20 years of earning highest degrees, humanity still stands far from achieving "
            "what true education was meant to deliver. We produce experts but not visionaries, "
            "professionals but not reformers, scholars but not seekers of truth. "
            "Democracy represents merely the rule of the majority, not collective wisdom. "
            "Capitalism rewards competition, not compassion. "
            "Science is often used as instrument of dominance rather than enlightenment. "
            "Vision: Blend each chapter of learning with the ultimate purpose of exploring "
            "and enhancing the meaning of life itself. "
            "Tests and outcomes based on Core Reflections: how each day a student thinks about "
            "their learning, how it relates to current problems, and ability to retain and connect "
            "values to the rest of their life. "
            "AI-driven personalized learning: adaptive assignments rooted in reflection, "
            "inquiry, and real-world application. Meaningful repetitive practice for long-term memory. "
            "Ultimate goal: Cultivate creators, thinkers, and visionaries who can rebuild systems."
        )
    },

    # ─── COMPETITIVE PROGRAMMING & LEARNING ───
    {
        "source": "vision-competitive-programming",
        "content": (
            "Suraj's Vision on Competitive Programming and Disruptive Learning: "
            "Reference: OpenAI participated in ICPC (International Collegiate Programming Contest) "
            "and solved all 12 challenges; human team (St. Petersburg University) solved 11/12. "
            "GPT-5 = 11, Gemini = 10/12. "
            "Core practices for mastering competitive programming: "
            "1. Rock-solid DSA foundation (arrays, trees, graphs, DP, greedy). "
            "2. Master Big O notation and time/space complexity analysis. "
            "3. Consistent daily practice on Codeforces, AtCoder, TopCoder, LeetCode, HackerRank. "
            "4. Upsolve after every contest - review unsolved problems. "
            "5. Counteracting forgetting: Active recall, spaced repetition, teach and explain, "
            "implement from scratch, contextual learning. "
            "Creative ways to bridge non-programmer to advanced pro coder with AI: "
            "AI-driven personalized curriculums, reverse engineering learning (dissect open-source), "
            "production-ready sprint simulation, gamified project-based learning, "
            "contextual immersion with IDE-integrated AI tutor."
        )
    },

    # ─── ADDITIONAL STARTUP IDEAS ───
    {
        "source": "vision-startup-more-ideas",
        "content": (
            "Suraj Kumar's Additional Startup Ideas: "
            "1. Advance Agent Assistant: AI that knows you better than yourself, defines per-minute goals, "
            "makes you oriented every second, sees you via phone, learns everything about you. "
            "2. Context-aware Prompt Generation Application for deep analysis. "
            "3. Replace the whole foundation of technology including hardware architecture and binary computation. "
            "4. Redefining regions across the planet into one committed for evolution of humankind. "
            "5. New Programming Language: Conversational English based, customizable, "
            "convertible to any other programming language. Goal: make programming as intuitive "
            "as human conversation. "
            "6. Company whose sole purpose is the ability to solve problems of all types. "
            "7. Language learning platform: vocabulary, sentence framing, intensive flow, "
            "imagination-focused content. Learn like Shakespeare's style of expressing effortlessly. "
            "8. Smart Talk/Conversational Skill platform: beyond communication skills - "
            "how to handle toxic people, avoid negative/useless conversations, "
            "save time from discussions that don't account for anything. "
            "9. Multi-media responsive website with AI characters detecting mood, playing music, "
            "dancing, creating addictive positive experience."
        )
    },
]

# ── Check existing sources ──
resp = requests.get(f"{BASE}/api/memories?limit=200")
existing = {m["source"] for m in resp.json().get("memories", [])}
print(f"Existing sources ({len(existing)}): {sorted(existing)}\n")

new_chunks = [c for c in chunks if c["source"] not in existing]
print(f"New chunks to ingest: {len(new_chunks)} / {len(chunks)} total\n")

success = 0
fail = 0
for i, chunk in enumerate(new_chunks, 1):
    payload = {
        "content": chunk["content"],
        "source": chunk["source"],
        "session_id": "bulk-add-more"
    }
    try:
        r = requests.post(f"{BASE}/api/memories/ingest", json=payload, timeout=120)
        if r.status_code == 200:
            print(f"  ✅ [{i}/{len(new_chunks)}] {chunk['source']}")
            success += 1
        else:
            print(f"  ❌ [{i}/{len(new_chunks)}] {chunk['source']} → {r.status_code}: {r.text[:200]}")
            fail += 1
    except Exception as e:
        print(f"  ❌ [{i}/{len(new_chunks)}] {chunk['source']} → ERROR: {e}")
        fail += 1
    time.sleep(0.3)

print(f"\n{'='*60}")
print(f"Done: {success} succeeded, {fail} failed out of {len(new_chunks)} new chunks")

# Final count
r2 = requests.get(f"{BASE}/api/memories?limit=200")
data = r2.json()
print(f"Total memories now: {len(data.get('memories', []))}")
print(f"Total vectors: {data.get('total_vectors', '?')}")
print(f"Total graph nodes: {data.get('graph_nodes', '?')}")
