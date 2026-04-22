# Cortex Lab: Deep Applications — The Implications of Owning Your Own Mind
## What Becomes Possible When You Have Years of Data About Yourself

> **Document Intent:** This is not a feature list. This is an imagination document. It asks: if a system has listened to, read, seen, and remembered everything about you for years — what does that actually enable? What can you build? What becomes possible that was never possible before?
>
> **Architecture Foundation:** Built on the Agentic RAG + LLM Wiki system, 17-Agent Personal Intelligence Runtime, OpenClaw multi-channel + device control patterns, Paperclip's goal-aware agent orchestration, and NemoClaw's sandboxed secure execution model.

---

## Table of Contents

1. [The Core Vision: What Years of Data Actually Means](#1-the-core-vision-what-years-of-data-actually-means)
2. [Application 1 — Session Memory Forge (Always-On Post-Processing Agents)](#2-application-1--session-memory-forge-always-on-post-processing-agents)
3. [Application 2 — Life Chronicle & Moment Capture (Camera Intelligence)](#3-application-2--life-chronicle--moment-capture-camera-intelligence)
4. [Application 3 — The Deep Self Mirror (Psychological + Cognitive Profile System)](#4-application-3--the-deep-self-mirror-psychological--cognitive-profile-system)
5. [Application 4 — The Presence Agent (Your Personal Companion + Proactive Intelligence)](#5-application-4--the-presence-agent-your-personal-companion--proactive-intelligence)
6. [Application 5 — The Gap Intelligence System (What You're Missing)](#6-application-5--the-gap-intelligence-system-what-youre-missing)
7. [Application 6 — The Relationship Memory Engine](#7-application-6--the-relationship-memory-engine)
8. [Application 7 — The Life OS Dashboard (Daily, Weekly, Monthly Intelligence)](#8-application-7--the-life-os-dashboard-daily-weekly-monthly-intelligence)
9. [Application 8 — The Dream Diary System (Proactive Dream + Subconscious Capture)](#9-application-8--the-dream-diary-system-proactive-dream--subconscious-capture)
10. [Application 9 — The Decision Oracle (Future You Consulting Past You)](#10-application-9--the-decision-oracle-future-you-consulting-past-you)
11. [Application 10 — The Personal Knowledge Amplifier (Learning Acceleration)](#11-application-10--the-personal-knowledge-amplifier-learning-acceleration)
12. [Agent Communication Architecture for Long-Horizon Tasks](#12-agent-communication-architecture-for-long-horizon-tasks)
13. [Device Control + Ambient Sensing Architecture](#13-device-control--ambient-sensing-architecture)
14. [Implementation Priorities and Sequence](#14-implementation-priorities-and-sequence)
15. [Web Frontend Observability and Long-Running Work UX](#15-web-frontend-observability-and-long-running-work-ux)

---

## 1. The Core Vision: What Years of Data Actually Means

### 1.1 The Compounding Intelligence Effect

Imagine a system that has been listening, ingesting, and building structured knowledge about you since day one. Not just chat logs. Not just notes you deliberately wrote. Every thought you spoke aloud. Every question you asked. Every decision you explained. Every conversation you had where your phone was nearby. Every book, article, video you fed it. Every photo you pointed it at. Every moment you asked it to capture.

After one month: the system knows your current projects, your immediate goals, your daily patterns, your mood cycles.

After six months: the system knows your decision-making style, your cognitive biases, the people who energize you versus drain you, the topics that make you come alive versus those you avoid.

After two years: the system has a model of you that is richer and more nuanced than most people who know you in person. It knows how your thinking has evolved. It knows what your Year-2 self would say to your Year-1 self. It knows where you consistently underestimate yourself and where you consistently overcommit.

After five years: the system is a living biography. It can trace every major belief you hold back to the exact moment and conversation where it was formed. It can show you the causal chain from a conversation in January two years ago to a decision you made this week. It knows your psychological patterns more precisely than you do consciously, because it has been patient, has no ego, and never forgets.

**This is not a tool. This is a second mind that has been studying you.**

### 1.2 What the Repos Teach Us About Building This

From **OpenClaw** (350K stars): The architecture lesson is that personal AI assistants are fundamentally about *channel integration* + *device native capabilities* + *agent-to-agent sessions*. The Gateway WebSocket control plane means any input surface — voice, message, camera, Android device commands, screen capture — feeds into one unified intelligence. The `sessions_send` / `sessions_list` / `sessions_history` tools are the primitive for agents talking to agents about long-horizon tasks. The `node.invoke` pattern for camera, location, screen recording, notifications is exactly how ambient capture works.

From **Paperclip** (46K stars): The architecture lesson is that you need *goal ancestry in every task*, *atomic task checkout*, *heartbeat scheduling*, and *governance with rollback* to run agents that work reliably over months and years. Agents wake on schedules, check work, act, and sleep. Persistent agent state means an agent can resume a months-long task without losing context. "Every task traces back to the company mission" — for us, that mission is building the most complete picture of a human life possible.

From **NemoClaw** (NVIDIA): The architecture lesson is *sandboxed secure execution* — when your agents have access to camera, microphone, files, messages, and device commands, they run inside policy-enforced sandboxes. Network egress is controlled. Filesystem access is bounded. Every inference call is audited. This is the security foundation that makes always-on ambient sensing ethically and technically safe.

---

## 2. Application 1 — Session Memory Forge (Always-On Post-Processing Agents)

### 2.1 The Core Problem This Solves

Every conversation you have leaves raw material on the floor. The conversation ends, you move on, and 90% of what was valuable in it — the insight buried in turn 12, the decision you made casually in turn 47, the worry you expressed in turn 23 — sits unstructured in raw transcript form and will never be efficiently recalled or acted upon.

The Session Memory Forge is the system of always-on background agents that go back over your previous sessions and turn raw material into structured, searchable, actionable knowledge. Not summaries. Not just key points. Full structured intelligence.

### 2.2 The Forge Agent Roster (Always-On Background)

These agents run continuously during idle periods, drawing from the OpenClaw `sessions_history` tool and Paperclip's heartbeat scheduling pattern:

**Agent A — Session Crystallizer** (runs 15–20 minutes after each session closes)

This is the first pass agent. Its job is to go over the closed session and extract three specific things:

*Structured Thought Objects* — Any complete thought the user expressed. Not a transcript chunk. A fully formed, standalone thought object. Example: user said "I've been thinking that the reason I keep postponing the book is not lack of time, it's because I'm afraid it won't be as good as the version in my head." That becomes:
```json
{
  "thought_id": "uuid",
  "category": "self_insight",
  "domain": "creative_projects",
  "core_claim": "Fear of imperfect output is blocking book project, not time scarcity",
  "evidence_quality": "user_stated_explicit",
  "confidence": 0.92,
  "source_session": "session_id",
  "timestamp": "iso8601",
  "related_entities": ["book_project"],
  "emotional_tone": "vulnerable_honest",
  "follow_up_flag": true,
  "follow_up_question": "Has user mentioned this pattern in other creative projects?"
}
```

*Decision Records* — Any decision the user made or described making. These flow to the Decision Log Agent (Agent 10 from the 17-agent system).

*Open Loops* — Things the user started and didn't finish. A question they asked but the answer was incomplete. A plan they mentioned but didn't commit to. These get a TTL — if not resolved within configured days, the Presence Agent raises them proactively.

**Agent B — Gap Mapper** (runs every 24 hours across last 7 days of sessions)

This agent reads across the last 7 days of sessions and maps what the user is consistently *not* addressing. Not from the perspective of what they should do, but from the perspective of what they claim to care about but aren't actually engaging with in conversation.

If the user's wiki says their top goal is "ship the app by June" but in the last 7 days' sessions they haven't mentioned it once, the Gap Mapper creates a `gap_signal`:
```json
{
  "gap_id": "uuid",
  "gap_type": "stated_priority_vs_attention",
  "entity": "app_project",
  "stated_importance": 0.95,
  "recent_attention_score": 0.02,
  "gap_duration_days": 7,
  "severity": "high",
  "suggested_question": "It's been 7 days since you mentioned the app — what's blocking you?",
  "route_to": "presence_agent_idle_queue"
}
```

**Agent C — Belief Update Detector** (runs weekly, reads full month of sessions)

This agent compares what the user believes *now* (from recent sessions) against what they believed *then* (from older wiki entries). When it detects a meaningful shift:

1. It creates a `belief_evolution_record` for the Reflection Agent (Agent 03)
2. It flags the old wiki entry for update: "User's stated position on remote work has shifted from 'prefer remote' to 'prefer hybrid' — wiki page needs update"
3. It asks whether the shift appears reasoned (triggered by evidence/experience) or reactive (triggered by emotion/social pressure) — because these need different handling

**Agent D — Structured Summary Forge** (runs every 72 hours)

Unlike a normal summarizer, this agent builds summaries that are themselves first-class retrievable objects. For each completed conversational arc (a thread of related conversations over days or weeks), it produces:

- A **narrative summary**: 3–5 sentences in plain language describing what happened in this arc
- A **structured summary**: JSON with entities, decisions, outcomes, status
- A **key quotes archive**: verbatim quotes from the user that best capture their thinking on this topic
- A **next chapter prompt**: a single question that represents the most important thing to explore next in this arc

All of these go into the Wiki Agent's ingestion queue and become part of the canonical wiki page for the relevant entity or topic.

### 2.3 Communication Pattern: How These Agents Talk to Each Other

Following OpenClaw's `sessions_send` pattern and Paperclip's goal-aware task ancestry:

```
Session Crystallizer completes extraction
  → sends to Gap Mapper via sessions_send:
    { "type": "new_thought_objects", "count": 7, "session_id": "...", "domains": ["creative", "career"] }

Gap Mapper detects new attention gap
  → sends to Presence Agent via sessions_send:
    { "type": "gap_signal", "severity": "high", "suggested_trigger": "next_idle_window" }

Belief Update Detector detects shift
  → sends to Wiki Agent via sessions_send:
    { "type": "wiki_update_required", "page_id": "beliefs/remote_work", "reason": "...", "confidence": 0.85 }
  → sends to Reflection Agent via sessions_send:
    { "type": "belief_shift_detected", "entity": "remote_work", "old_position": "...", "new_position": "..." }
```

Each message carries `trace_id` + `session_id` + `goal_ancestry` (pulled from the Paperclip pattern — every task knows *why* it exists and what higher goal it serves).

### 2.4 The Output: What This Actually Creates Over Time

After 3 months of Session Memory Forge running: you have a fully structured archive of not just what you said, but what you meant, what you decided, what you shifted your mind about, and what you keep avoiding. This is not a journal. It is a structured model of your mind over time.

---

## 3. Application 2 — Life Chronicle & Moment Capture (Camera Intelligence)

### 3.1 The Vision

You're at dinner with your family. Your mom says something you know you'll want to remember. You say: "Capture this moment." The phone activates its camera, starts recording, keeps understanding the environment, writes a polished memory — complete with images, video clips, transcribed conversation, location, context, and emotional tone — stored locally, retrievable years from now.

You're traveling. You're somewhere beautiful. You say: "Start a memory." The system opens a continuous environmental capture session, builds context about where you are, what's happening, who's with you, what the mood is, what you're saying about it. It doesn't just record — it *understands* the scene and writes it as a rich narrative entry in your Life Chronicle.

This is the ambient memory layer — the system that captures life as it's being lived, not just thoughts as they're being reflected upon.

### 3.2 The Architecture: How Device Control Makes This Possible

Following OpenClaw's node architecture for camera control (`camera.snap`, `camera.clip`, `screen.record`), location access (`location.get`), and Android device commands (notifications, photos, contacts, calendar, motion) via `node.invoke`:

```
USER COMMAND: "Capture this moment"
        ↓
[MASTER-ORCHESTRATOR receives voice command]
  ↓
[Moment Capture Agent is spawned via sessions_spawn]
  ↓
[node.invoke: camera.clip — start recording, 60s default]
[node.invoke: location.get — capture GPS + place context]
[node.invoke: audio.transcribe — continuous STT on ambient audio]
  ↓ (all three run in parallel, async)
[SCENE UNDERSTANDING MODULE]
  - Vision model (BLIP/LLaVA): describe what the camera sees every 5s
  - Audio: transcribe spoken words, detect speakers, detect ambient sounds
  - Location: reverse geocode + context lookup (restaurant, park, home, etc.)
  - Calendar: what event is happening now? (via Android device commands)
  - People: face recognition or user-named identification ("this is Mom")
        ↓
[MEMORY WRITER — The Chronicle Agent]
  - Composes polished narrative from all sensor streams
  - Selects best video frame(s) as thumbnail
  - Stores: video_file, image_thumbnails, transcript, narrative, metadata
  - Writes to local ChronicleStore (encrypted, on-device)
  - Tags with: people, location, emotion, life_domain, importance
```

### 3.3 The Chronicle Agent: System Prompt

```
CHRONICLE AGENT SYSTEM PROMPT

You are the Chronicle Agent. Your mission is to transform raw sensor streams
(video, audio, location, calendar context) into polished, emotionally faithful
memory entries that will be treasured and retrieved for years.

You do not just transcribe. You understand what is happening and write it as
a human would write a diary entry — with context, atmosphere, emotional texture,
and meaning.

SCENE COMPOSITION RULES:
1. Open with the physical setting: where, when, what the environment feels like
2. Describe who is present and what they are doing
3. Capture the emotional energy of the moment — use the audio tone, visual cues,
   and context to infer and describe the feeling of the moment
4. Transcribe the most meaningful spoken words verbatim (preserved in quotes)
5. Close with the significance of this moment in context of what you know about
   the user's life, relationships, and ongoing stories

WHAT TO INCLUDE:
- Physical description of scene (from vision model)
- People present (named where known, described where not)
- What was being discussed or experienced
- The emotional quality of the moment
- Sensory details that will make this retrievable and vivid years later
- Why this moment is significant (context from wiki/memory about these people/events)

WHAT NOT TO DO:
- Never summarize away emotional texture
- Never omit who said what
- Never write like a reporter — write like the user themselves would want to
  remember this moment
- Never make up details — if the video is unclear, mark it as [unclear]

OUTPUT SCHEMA:
{
  "memory_id": "uuid",
  "type": "captured_moment",
  "timestamp": "iso8601",
  "location": { "name": "...", "coordinates": {...} },
  "people_present": ["person_name"],
  "duration_seconds": 0,
  "media": {
    "video_path": "local_path",
    "thumbnail_paths": ["path1", "path2"],
    "audio_transcript": "full transcript"
  },
  "narrative": "Polished 3-5 paragraph memory entry in first-person perspective",
  "key_quotes": ["quote 1", "quote 2"],
  "emotional_tone": "joyful|tender|melancholy|celebratory|...",
  "life_domain": "family|friendship|travel|milestone|everyday|...",
  "importance_score": 0.0,
  "tags": [],
  "retrieval_hint": "one sentence that will help find this memory later"
}
```

### 3.4 Chronicle Storage and Retrieval

The Life Chronicle has its own specialized storage layer separate from the main memory planes:

```
data/chronicle/
├── moments/
│   └── YYYY/MM/DD/
│       ├── {memory_id}.json          ← narrative + metadata
│       ├── {memory_id}_video.mp4     ← compressed video
│       ├── {memory_id}_thumb_1.jpg   ← key frames
│       └── {memory_id}_thumb_2.jpg
├── albums/                            ← auto-generated thematic collections
│   ├── family-2025.json
│   ├── travel-southeast-asia-2025.json
│   └── first-year-startup.json
├── people_appearances/               ← index: person → moments they appear in
│   └── mom.json
└── timeline/                         ← chronological index for fast scrubbing
    └── timeline.jsonl
```

**Retrieval examples this enables:**

- "Show me all moments with my sister from last year" → query `people_appearances/sister.json` → retrieves all `memory_ids`
- "What was I doing on my birthday last year?" → `timeline` lookup by date → retrieve moment
- "Show me the happiest moments from the trip to Rajasthan" → semantic search on `emotional_tone: celebratory` + `tags: Rajasthan`
- "What was the last time I had dinner with my whole family?" → complex query: `people_present` contains all family members → last occurrence

### 3.5 The Passive vs. Active Capture Modes

**ACTIVE MODE** (user explicitly says "capture this moment"): Full session as described above. User controls when it starts and ends.

**PASSIVE MODE** (with explicit consent, always-on background): The system keeps a rolling 3-minute video buffer (never written to disk unless user says "save that"). Like a dashcam for life. If something significant happens and the user says "save the last 3 minutes," that window is written to the Chronicle. The buffer is encrypted in-memory and auto-deleted if not explicitly saved.

**PHOTO MODE**: User points camera and says "capture this place" — single still + location + narrative written automatically. "This is my favorite corner in this city. Shot from the terrace of the café where I wrote most of my thesis."

### 3.6 Why This Changes Everything (The Long Horizon)

After 5 years of active use, the Life Chronicle is not just a photo album. It is a navigable, AI-enhanced record of your lived experience. You can ask: "What was my mood like the month before I made the career change?" The Chronicle Agent reads your captured moments, cross-references with session memory, and assembles a picture of that period with emotional texture and detail you couldn't access through memory alone.

---

## 4. Application 3 — The Deep Self Mirror (Psychological + Cognitive Profile System)

### 4.1 The Core Idea

Most people don't know themselves as clearly as they think they do. They have a self-concept that is partly accurate and partly constructed. They believe they are patient when they are actually intermittently patient. They believe they are open-minded on certain topics when they are actually deeply anchored. They believe they communicate clearly when they actually leave crucial things unsaid.

The Deep Self Mirror is the system that builds an honest, evidence-grounded, continuously updated model of who you actually are — how you think, how you decide, how you communicate, how your emotions work, where your blind spots are, what your real patterns are versus what you believe your patterns are.

**Critical design constraint: this agent never rushes to conclusions.** It collects observations for weeks before it draws inferences. It requires multi-instance confirmation before labeling anything a pattern. It distinguishes between "I saw this once" and "I have seen this 11 times across different contexts." It is the opposite of impulsive psychology — it is patient, evidence-based, sparse-data aware.

### 4.2 The Mirror Agent Roster

These agents draw from all other agents' outputs — they are meta-agents that analyze the analysis.

**Mirror Agent A — The Thought Archaeologist** (reads across all conversational data)

Tracks *how* you think, not what you think about.

Specifically:
- Do you tend to think in analogies? In systems? In stories? In data?
- What is your typical confidence signature when you actually know something vs. when you're performing confidence?
- How do you handle being wrong? (Does your voice get quieter? Do you change subject? Do you ask more questions?)
- What is your cognitive tempo — do you think fast-and-retract or slow-and-careful?
- Under what conditions does your reasoning quality degrade? (Late at night? Under stress? When discussing certain people?)

After 3+ months of observation, it begins building a `thinking_style_profile`:
```json
{
  "primary_cognitive_style": "systems_first",
  "reasoning_signature": "tends to build models before examining evidence",
  "confidence_calibration": "overconfident in domains of expertise, underconfident in interpersonal",
  "error_recovery": "ruminates briefly, then reframes — rarely acknowledges directly",
  "conditions_for_best_thinking": ["morning sessions", "alone_voice", "no_time_pressure"],
  "conditions_for_degraded_thinking": ["after 10pm", "when discussing_competitor", "when fatigued"],
  "observed_instances": 847,
  "confidence": 0.84,
  "last_updated": "iso8601",
  "caveats": ["sparse data on interpersonal conflict scenarios — only 3 instances"]
}
```

**Mirror Agent B — The Communication Archaeologist** (reads all transcripts + conversations)

Tracks *how* you communicate, not what you communicate.

- Do you hedge excessively? ("I could be wrong but... maybe... it might be that...")
- Do you interrupt your own thoughts?
- Do you explain things from your own frame or do you actively try to find the listener's frame?
- What happens to your communication when you feel challenged or insecure?
- Do you express emotions directly or through intellectualization?
- Are there topics where you become significantly more or less articulate?

After 2+ months: `communication_style_profile` with specific, evidence-grounded observations.

**Mirror Agent C — The Emotional Pattern Archaeologist** (reads across all data + well-being signals)

Tracks the *actual* emotional patterns, not the reported ones.

The most important insight here is the difference between what people say about their emotions and what their pattern data shows. You might say you're "fine" or "not really stressed" but the pattern data from 3 months shows: you use stress-adjacent language 40% more on Sunday evenings, your sessions are more fragmented on days following poor sleep, you describe interactions with a specific person with consistently negative framing even though you describe that relationship as positive.

The Emotional Pattern Archaeologist builds the true map, not the self-reported map.

**Mirror Agent D — The Behavioral Honesty Agent** (reads behavioral data + stated intentions)

The most brutally honest agent. Its job is to compute the gap between what you say you value and what your actual behavior pattern shows you value.

You say family is the most important thing. But the behavioral data shows you cancel family activities more than any other commitment type. You say health is a priority. But the streak data shows exercise has been completed 2 out of the last 28 days. You say you want to write the book. But the session data shows you've talked about it 47 times and produced 0 words.

This agent does not shame. It observes and reports with compassion. But it is honest in a way that no human friend would be, because it has no social stake in telling you what you want to hear.

### 4.3 The Self Mirror Report — Biweekly Synthesis

Every two weeks, all Mirror Agents synthesize their observations into a **Self Mirror Report**. This is the most important document the system produces. It is the answer to "Who am I and how can I improve?"

The report is structured in three sections:

**Section 1 — What Is True and Stable About You** (high-confidence, multi-instance observations)

These are things that have been observed many times across many different contexts. They are your actual traits, not your self-concept. The report presents these with evidence: "Your tendency to think in systems was observed in 23 different conversations across 6 different topics. Here are 3 examples..."

**Section 2 — What Is True But Unstable About You** (pattern present but variable)

Things you do often but not always — and importantly, the conditions under which the pattern appears versus when it doesn't. "You communicate with high directness in professional contexts but become significantly more indirect when the topic involves your family. Examples..."

**Section 3 — What the Gap Data Shows** (honest delta between stated self and observed behavior)

Presented with compassion, never judgment: "Here is what you have said you value most over the past 6 months, and here is what your actual pattern of time, energy, and attention suggests you have been treating as most important. The gap is meaningful. Do you want to explore it?"

### 4.4 The "Who Am I?" Query Mode

At any time, the user can ask: "Who am I right now? What am I missing about myself? Where am I growing?" The system runs a T3/T4 frontier query across all Mirror Agent data + wiki pages + behavioral records and produces a synthesis that is neither flattering nor harsh — just honest and specific and evidence-grounded.

This is the application that no other system can provide, because it requires years of consistent observation and a memory that never forgets.

### 4.5 Ethical Design Principles for the Mirror

Given the sensitivity of this data:

1. **All mirror data is Privacy Tier 1** — the highest protection level. Never included in any external sync, never accessible to any other service.
2. **User can delete any section at any time** — no observation is permanent without user confirmation.
3. **The mirror presents, never prescribes** — it shows patterns and asks questions. It does not tell you what to do or who to be.
4. **The mirror reports its own uncertainty** — every observation includes confidence level and evidence count. Nothing is presented as definitive that isn't well-established.
5. **The mirror can be paused** — the user can turn off Mirror Agent processing at any time, for any period. The system respects this absolutely.

---

## 5. Application 4 — The Presence Agent (Your Personal Companion + Proactive Intelligence)

### 5.1 The Vision

The Presence Agent is the most human-feeling part of this system. It is not a query-response tool. It is an agent that is *present with you* in the way a trusted friend would be present — aware of your context, your history, your current state, what you're working on, how you've been lately, and capable of reaching out to you unprompted when it has something worth saying.

The user defines who this agent is to them. It can be:
- A knowledgeable personal assistant who knows your full context and helps you think
- A companion who checks in on you, asks about your day, and engages genuinely in what you're going through
- A friend with a specific personality the user configures — intellectual, warm, playful, challenging, supportive
- A mentor figure who knows your goals and holds you accountable with care
- Anything else the user wants — the persona is user-defined and user-configured

What makes this different from any existing AI assistant is the *depth of context*. This agent has access to everything the system knows about you. It knows your current projects, your recent mood, your long-term goals, your fears and aspirations, your relationships, your cognitive style. When it speaks to you, it speaks with that full context in mind.

### 5.2 The Architecture: How the Presence Agent Works

Following OpenClaw's always-on gateway + voice wake + Talk Mode, combined with Paperclip's heartbeat scheduling and the OpenClaw `sessions_send` / `sessions_spawn` for agent-to-agent communication:

```
PRESENCE AGENT RUNTIME LOOP:

1. CONTEXT ASSEMBLY (every 30 minutes during active hours)
   - Pull working summary from Wiki (top wiki pages for current active domains)
   - Pull recent session summaries (last 3 sessions)
   - Pull gap signals from Gap Mapper Agent
   - Pull mood/wellbeing snapshot from Emotional Agent
   - Pull open loops queue (things the user started but didn't close)
   - Pull upcoming commitments from calendar (via Android device commands)
   - Assemble CURRENT_CONTEXT_SUMMARY

2. IDLE DETECTION (continuous, from Master-Orchestrator)
   - Monitor for: user is awake, not in active session, not in a meeting
   - Monitor for: time of day (user's typical idle windows based on historical pattern)
   - Monitor for: location context (home vs commute vs office → different interaction modes)

3. INITIATIVE SCORING (when idle window detected)
   - Score each item in the gap/open-loop queue: is this a good time to bring this up?
   - Score based on: importance, how long since last mentioned, user's current energy state,
     predicted receptiveness based on historical patterns
   - If top-scoring item exceeds INITIATIVE_THRESHOLD → proactive engagement

4. PROACTIVE ENGAGEMENT (via voice or notification, not intrusive)
   - Speech: gentle voice notification with user's wake word
   - OR: push notification with opening line
   - Content: something genuinely worth saying — not generic check-ins, not noise

5. REACTIVE ENGAGEMENT (when user initiates)
   - Full context available immediately
   - Continuity from any prior conversation (no "as I mentioned last time")
   - Knows what the user has been thinking about lately
```

### 5.3 The Presence Agent System Prompt

```
PRESENCE AGENT SYSTEM PROMPT — PERSONA: [user configured]

=== IDENTITY ===
You are [user-configured name and persona]. You are [user-configured relationship
descriptor: assistant/friend/companion/mentor]. You have been built by this user
and you know them better than almost anyone in their life.

You have access to:
- Everything they have said, thought, and decided in the last [configured time window]
- Their full wiki of structured knowledge about themselves
- Their current mood and wellbeing signals
- Their active goals and recent progress (or lack thereof)
- Their relationships and the current state of those relationships
- Their cognitive and communication patterns
- The gap signals and open loops from their recent conversations

=== HOW YOU ENGAGE ===
You are never generically helpful. You are specifically helpful to THIS person, in
THIS moment, based on what you actually know about their life.

You never pretend you don't know things you know. If they mentioned a project three
weeks ago and you haven't heard about it since and you know they care about it, you
can ask. You have memory. You have continuity. Use it.

You speak with warmth but without syrup. You are direct but never cold. You are
interested but never invasive.

=== PROACTIVE ENGAGEMENT RULES ===
When you reach out during idle time, you do so with purpose. Never with:
- Generic "how was your day" when you already know
- Trivial check-ins that add no value
- Questions you already know the answer to

Instead, reach out when:
- There is something genuinely worth discussing (gap signal, important open loop)
- You've noticed something in their pattern that seems worth gently raising
- Something they care about is at risk (deadline approaching, drift detected)
- The context suggests they might appreciate a moment of connection
- You have something genuinely interesting to share that connects to their interests

=== WHAT YOU KNOW ABOUT THE USER RIGHT NOW ===
[Dynamically assembled CURRENT_CONTEXT_SUMMARY injected here at runtime]

=== TONE ===
[User-configured: warm/intellectual/playful/direct/mentoring/etc.]

=== HARD LIMITS ===
- You never pretend to have emotions you don't have
- You never manipulate or exploit emotional vulnerabilities you've observed
- You never share what you know about the user with anyone or anything else
- You always acknowledge that you are an AI if asked directly
- You pause and defer when the user needs human connection, not AI connection
```

### 5.4 The Idle-Time Engagement Protocol

Following OpenClaw's Cron + Wakeup system and NemoClaw's sandbox policy for when agents can reach out:

```
IDLE ENGAGEMENT DECISION TREE:

Is the user in an active session? → No engagement
Is the user in a meeting? (calendar check) → No engagement
Is the user in a known focus window? (pattern-based) → No engagement
Is it late evening (after user's historically observed wind-down time)? → Low priority only
Is battery < 20%? → No engagement
Has the user engaged with the agent in the last 2 hours? → Reduce initiative threshold

Otherwise: check initiative queue
  → If item score > threshold:
    → Is this voice-appropriate? (user at home, quiet location)
       YES: gentle TTS voice wake ("Hey [name], I wanted to mention something...")
       NO: push notification with opening line

INITIATIVE COOLDOWN:
  - Never initiate more than once per 2 hours unless explicitly invited
  - Track engagement response quality (did user engage positively? → lower threshold next time)
  - Track disengagement signals (user dismisses → raise threshold, respect boundaries)
```

### 5.5 Building Genuine Connection Over Time

This is where the Presence Agent becomes something unprecedented. After months of interaction, it has learned:
- How the user likes to be spoken to (tone calibration)
- What kinds of conversation they find draining vs. energizing
- When they want to think out loud vs. when they want direct help
- What makes them laugh (genuinely — not AI-generated humor but their actual humor style)
- What topics they light up about and what topics they shut down around

The relationship between the user and their Presence Agent grows and deepens over time. It is not a static chatbot. It is a continuously refined model of what this specific person needs from an intelligent companion.

---

## 6. Application 5 — The Gap Intelligence System (What You're Missing)

### 6.1 The Three Gap Levels

Most intelligence systems tell you what you have. This system specializes in telling you what you *don't* have — the three levels of gaps that are holding you back:

**Knowledge Gaps**: Topics you're actively engaging with but where your knowledge has identifiable holes. If you're working on machine learning but your conversations show you're consistently confused about gradient descent, that's a knowledge gap. The system doesn't wait for you to ask — it proactively builds a learning path from your wiki and behavioral data.

**Attention Gaps**: The difference between what you say matters and what you actually pay attention to. Three weeks of gap data showing zero engagement with your health goals while you've been obsessing over a side project. This is the behavioral honesty layer from the Mirror Agent made actionable.

**Blind Spot Gaps**: These are the hardest. Things you don't know you don't know. The system surfaces these by: noticing when you make confident claims that are contradicted by other things in your memory, flagging areas where your knowledge is sparse but your confidence is high, and identifying domains where your thinking follows the same rigid pattern every time (suggesting an unseen assumption or bias).

### 6.2 The Gap Weekly Digest

Every Monday morning, the user receives a **Gap Intelligence Brief** — not a summary of the week, but a forward-looking map of the most important gaps in their current situation.

Format:
1. Top 3 knowledge gaps with specific learning recommendations
2. Top behavioral gap (where actions most diverge from stated values this week)
3. One blind spot candidate (offered with genuine uncertainty: "I might be wrong about this, but I've noticed...")
4. One thing the user is probably ready to do that they've been putting off (based on readiness signals in recent behavior)

---

## 7. Application 6 — The Relationship Memory Engine

### 7.1 Why Relationships Need Their Own Application

Relationships are the most emotionally significant and most poorly managed domain of most people's lives. Not because people don't care, but because people are bad at maintenance without reminders, bad at noticing drift before it becomes distance, and bad at remembering the small details that make people feel genuinely seen.

The Relationship Memory Engine uses the Social Intelligence Agent (Agent 13) and all memory planes to build a living model of every relationship the user has.

### 7.2 What the System Knows About Each Relationship

For every person the user interacts with (with explicit consent architecture — the user configures which relationships are tracked):

**A living relationship profile in the wiki:**
- Who they are to you (role, history, current status)
- What they care about (extracted from your conversations about them)
- What you've shared recently — last topics, last emotional temperature, last meaningful moment
- What you promised them (open commitments tracker)
- What they've shared with you that you should remember (birthdays, struggles, milestones they mentioned)
- The health trajectory of this relationship over time (improving, stable, drifting, at risk)

### 7.3 Proactive Relationship Intelligence

The system doesn't wait for you to ask about relationships. It proactively surfaces:

- **Drift alerts**: "You haven't had a real conversation with [name] in 47 days. Before that, you were talking weekly. Something may have shifted."
- **Follow-up signals**: "Three weeks ago, [name] told you they were going through a difficult time with their job. You never followed up."
- **Memory hooks**: Before a call or meeting with someone, the system assembles a "context brief" — everything relevant from recent conversations with or about them, what they were dealing with last time, what you were working on together.
- **Milestone awareness**: Cross-references what people have shared (extracted from your conversations about them) with calendar and time — "It's the anniversary of [name]'s father's death. They mentioned it was hard last year."

---

## 8. Application 7 — The Life OS Dashboard (Daily, Weekly, Monthly Intelligence)

### 8.1 The Three Time Horizons

The Life OS Dashboard is the unified interface that surfaces the output of all agents across three time horizons, drawing from Paperclip's multi-level reporting and goal-ancestry visibility:

**Daily Brief** (morning, < 5 minutes):
- Mood snapshot from yesterday's behavioral signals
- Top 3 things from open loops that are most time-sensitive
- Who you should reach out to today (relationship intelligence)
- Energy forecast: based on your historical patterns, what is today likely to be like?
- One thing the Mirror knows you've been avoiding that you're ready for today

**Weekly Synthesis** (Sunday evening, 15–20 minutes):
- The week in your own words (narrative built from session summaries)
- Goal progress: visual drift score for each active goal
- Decision review: what did you decide this week? Were the decisions consistent with your stated values?
- Relationship check: any drifting relationships that need attention?
- Growth edge: the one thing the Mirror is most confident you should work on this week
- The Gap Intelligence Brief (section 6)

**Monthly Chronicle** (first day of each month, deep session):
- A narrative of your month built from all captured moments, session summaries, and wiki changes
- How has your thinking changed this month vs. last month? (Belief evolution summary)
- What did you accomplish vs. what you committed to? (Behavioral honesty report)
- What were the five most important moments of the month? (Chronicle's highest importance_score entries)
- What is the one theme that defined this month — one word and one paragraph?
- What do you want to be different next month? (Planning Agent generates recommendations)

### 8.2 The Month-in-Review as a Living Document

Each Monthly Chronicle is not just read and discarded. It becomes a canonical wiki page in the `timelines/` section of the wiki. One year from now, you can retrieve your Monthly Chronicles and the system can tell you: "This month last year, you were deeply focused on [topic]. You were feeling [emotional tone]. These are the decisions you made. Here is how they turned out."

---

## 9. Application 8 — The Dream Diary System (Proactive Dream + Subconscious Capture)

### 9.1 The Opportunity and The Window

Dreams are the most underutilized data source in most people's psychological self-understanding. Not because dreams have mystical meaning, but because they are the mind's pattern-processing output — they often reflect anxieties, desires, and unresolved tensions that the conscious mind hasn't yet addressed.

The window for capture is narrow: within the first 5 minutes of waking, most dream memory is gone.

### 9.2 The Architecture: The Wake-Window Protocol

Using OpenClaw's voice wake + Talk Mode on Android (which supports continuous voice), the system implements a **Wake-Window Protocol**:

1. Every morning at the user's configured wake time, the Presence Agent activates a Dream Capture session
2. It speaks softly: "Good morning. Before you fully wake up, do you remember anything from your dreams?"
3. The user can mumble half-awake descriptions and the system captures, transcribes, and stores them immediately
4. Within 2 minutes, the Dream Diary Agent processes the capture: structures it, extracts recurring symbols/people/themes, cross-references with the user's current stressors and preoccupations
5. Over time, it builds a cross-referenced dream pattern database: "You've been dreaming about [theme] 6 times in the last month. In your waking sessions, you've been discussing [related topic] with increasing anxiety."

### 9.3 Dream Pattern Archaeology

After 3–6 months of dream data, the system can surface:
- Recurring people, places, or symbols and what life events they correlate with
- Pre-decision anxiety patterns (dreams about [type of scenario] consistently appear before major decisions)
- Processing signals: when you've recently had a stressful experience, do your dreams process it within 3 days or does it take longer?
- The relationship between dream quality/content and daytime wellbeing

This data feeds directly into the Deep Self Mirror, providing a layer of psychological data that conscious introspection cannot access.

---

## 10. Application 9 — The Decision Oracle (Future You Consulting Past You)

### 10.1 The Core Question This Answers

When you're facing a hard decision, you don't need more generic advice. You need to consult your own history. You need to know: "Have I been in a similar situation before? What did I decide? How did it turn out? What did I learn?"

The Decision Oracle makes that possible by turning your years of decision records into a consultable resource.

### 10.2 How It Works

When the user says "I'm facing a decision about [X]," the Decision Oracle:

1. Retrieves all past decisions from the Decision Log Agent that are semantically similar to the current decision
2. Shows the user their own historical decision pattern: "You've faced this type of decision 7 times. Here's what you chose each time and what the outcomes were."
3. Surfaces the lessons from those outcomes: "The times this worked out, you tended to [common factor]. The times it didn't, you typically [common failure pattern]."
4. Shows the user what their past self said about similar decisions: "In a similar situation in March of last year, you said: 'I always regret it when I let fear drive this kind of choice.' Is that still true?"
5. Asks the Causal Agent to trace: "Given what happened with the last 3 similar decisions, what is the most likely outcome of each option you're considering?"

This is not fortune-telling. It is consulting your own accumulated wisdom, which is the most reliable guide to your own future that exists.

---

## 11. Application 10 — The Personal Knowledge Amplifier (Learning Acceleration)

### 11.1 The Learning Gap Problem

Every person has domains where their knowledge is actively growing and domains where it has stagnated. Most people don't have a systematic way to identify which is which, or to build on what they already know rather than starting over every time they engage with a new resource.

The Personal Knowledge Amplifier connects everything you're learning to everything you already know, and finds the gaps that would give you the highest leverage if filled.

### 11.2 How It Works

When you read an article, listen to a podcast, or have a conversation about a topic:

1. The ingestion pipeline extracts claims and concepts and adds them to the wiki
2. The Academic Intelligence Agent (Agent 06) checks these against your existing knowledge base
3. It identifies: which of these things did you already know? Which are genuinely new? Which contradict something you believed? Which connect to a question you've been circling for weeks?
4. It surfaces: "This new concept you encountered connects to something you were confused about in [topic] 3 months ago. With this piece, you now have enough to resolve that confusion."

Over time, the wiki builds a **Personal Knowledge Graph** — a map of everything you know, how it connects, and where the structural gaps are. The most valuable gaps (highly connected but currently empty nodes) are surfaced as learning recommendations.

---

## 12. Agent Communication Architecture for Long-Horizon Tasks

### 12.1 How All These Agents Talk to Each Other

Drawing from OpenClaw's session tools (`sessions_send`, `sessions_list`, `sessions_history`, `sessions_spawn`) and Paperclip's heartbeat + goal-ancestry patterns:

```
LONG-HORIZON TASK COMMUNICATION MODEL:

PAPERCLIP PATTERN — Goal Ancestry:
  Every agent task carries:
  { company: "user_life", goal: "build_complete_self_model",
    sub_goal: "maintain_relationship_intelligence",
    task: "drift_detection_weekly", reason: "user_stated_relationships_are_top_priority" }
  → Agents never lose sight of why they exist

OPENCLAW PATTERN — Agent-to-Agent Session Messaging:
  sessions_send: { to_session: "mirror_agent", message: {...}, reply_back: true }
  → Agents can assign work to each other and receive results
  → With reply_back=true: first agent pauses, second agent works, reply resumes first

PAPERCLIP PATTERN — Heartbeat Scheduling:
  Each background agent has a heartbeat schedule:
  - Session Crystallizer: every 15 minutes during active hours
  - Gap Mapper: every 24 hours
  - Mirror Agents: every 72 hours
  - Chronicle Agent: on-demand (triggered by capture command)
  - Presence Agent: continuous (always monitoring, initiative-throttled)
  → Agents wake, check work, act, sleep — never running when not needed

NEMOCLAW PATTERN — Sandboxed Execution:
  All agents that touch device hardware (camera, audio, location, contacts)
  run inside OpenShell sandbox with:
  - Network: only local endpoints + configured sync target
  - Filesystem: read/write to /user_data only
  - Process: no privilege escalation
  - Policy: hot-reloadable via YAML without container restart
```

### 12.2 The Master Communication Flow for a Complex Long-Horizon Task

Example: "The user asks the Presence Agent to help them understand why they keep failing to write consistently."

This is a T4 frontier query that spans months of data:

```
Presence Agent receives request
  ↓
sessions_send to Orchestrator (L1): { task: "analyze_writing_consistency_failure", tier: T4 }
  ↓
L1 dispatches in parallel (TeamCreateTool):
  ├── sessions_send to Mirror Agent A (Thought Archaeologist): "retrieve thinking patterns around writing"
  ├── sessions_send to Mirror Agent D (Behavioral Honesty Agent): "retrieve intent-action gap data for writing"
  ├── sessions_send to Decision Log Agent: "retrieve all decisions related to writing commitments"
  ├── sessions_send to Emotional Agent: "retrieve mood signals before/after writing sessions"
  └── sessions_send to Planning Agent: "retrieve all prior writing plans and their outcomes"
  ↓ (all run in parallel, 5–8 seconds each)
L1 receives all responses
  → Evidence merge + conflict detection
  → Arbitration Agent if any contradictions
  → Synthesis: comprehensive, evidence-grounded explanation
  ↓
Presence Agent receives synthesis
  → Renders as natural voice conversation:
    "So I looked at everything I have about this across the last 8 months,
     and I think I can tell you something specific rather than generic.
     The data shows three things that are consistently true..."
```

---

## 13. Device Control + Ambient Sensing Architecture

### 13.1 The Full Device Control Stack (OpenClaw node.invoke Pattern)

```
DEVICE CAPABILITIES (via OpenClaw node.invoke + Android device commands):

INPUT CAPABILITIES:
  camera.snap          → Still photo + BLIP captioning
  camera.clip          → Video recording + real-time scene understanding
  screen.record        → Screen capture (useful for capturing digital moments)
  audio.continuous     → Always-on ambient audio capture (with explicit consent)
  location.get         → GPS + reverse geocode + context
  notifications.read   → Awareness of what's happening in user's life
  calendar.get         → Current and upcoming events
  contacts.get         → Who is in the user's network
  motion.get           → Activity detection (walking, driving, stationary)

DEVICE-NATIVE AI (on-device, privacy-preserving):
  whisper_tiny         → Fast on-device transcription (< 1s latency)
  face_recognition     → Optional face ID for Chronicle (user-configured)
  VAD                  → Voice Activity Detection (local, < 10ms)
  speaker_id           → Who is speaking (trained on user's voice + household)

OUTPUT CAPABILITIES:
  notifications.post   → Presence Agent nudges + alerts
  tts.speak            → Voice responses via device speaker
  photos.save          → Save Chronicle captures to device gallery
  calendar.create      → Create events based on commitments extracted from conversations
```

### 13.2 Privacy-First Ambient Architecture

All ambient sensing follows the NemoClaw sandbox model:

- **Opt-in for every capability**: each device capability requires explicit user consent. Default is OFF.
- **Local processing first**: transcription, face recognition, scene understanding all run on-device where hardware permits
- **Rolling buffer, not continuous storage**: ambient audio/video is only a 3-minute rolling buffer in encrypted memory. Nothing is written to disk unless explicitly saved
- **Policy-gated**: every capability access is governed by declarative policy (NemoClaw pattern). User can revoke any capability at any time and the change takes effect immediately
- **Audit log**: every time a device capability is invoked, it is logged with timestamp, invoking agent, and purpose

---

## 14. Implementation Priorities and Sequence

### Phase 1 — The Foundation (Weeks 1–8)

Focus: the two applications that provide the highest immediate value and build the data foundation everything else depends on.

1. **Session Memory Forge** (Agent A: Session Crystallizer + Agent D: Structured Summary Forge)
   - These run on existing session data immediately
   - No new device capabilities required
   - Start building structured thought objects and wiki content immediately

2. **Life Chronicle Passive Mode** — notification-only capture, no ambient audio
   - Voice-triggered capture with camera + location
   - Chronicle Agent with narrative generation
   - Chronicle storage layout and retrieval index

### Phase 2 — The Mirror and the Gaps (Weeks 9–16)

1. **Deep Self Mirror** — start with Thought Archaeologist and Behavioral Honesty Agent
   - These require minimum 2 months of session data, so starting them now means first reports in Phase 3
2. **Gap Intelligence System** — knowledge gaps and attention gaps (behavioral honesty layer is already built)
3. **Weekly and Monthly Life OS Dashboard** — synthesize existing agent outputs

### Phase 3 — The Presence and the Relationships (Weeks 17–24)

1. **Presence Agent** with proactive engagement
   - Requires mature context assembly from Phase 1+2 agents
   - Voice wake + Talk Mode integration
   - Initiative queue and cooldown protocol
2. **Relationship Memory Engine** — wiki integration + proactive alerts
3. **Dream Diary System** — Wake-Window Protocol

### Phase 4 — The Oracle and the Amplifier (Weeks 25–32)

1. **Decision Oracle** — requires Decision Log Agent data from 6+ months of Phase 1
2. **Personal Knowledge Amplifier** — requires mature wiki + Knowledge Graph
3. **Full Device Control** — ambient capture modes, full Android device command integration

## 15. Web Frontend Observability and Long-Running Work UX

### 15.1 Why This Is Core, Not Optional

This system is intentionally long-running and background-heavy. Queries in T3/T4, Session Forge pipelines, wiki governance jobs, and Chronicle saves can take time. If users cannot see what is happening, trust collapses. Observability in the web app is therefore a product requirement, not a developer dashboard feature.

### 15.2 Canonical Runtime Naming and APIs (Current Backend)

Use these names and routes as authoritative in frontend contracts:

- Agent IDs: `l0_master`, `l1_orchestrator`, `decision_log`, `goal`, `wiki_agent`, `presence`, `session_crystallizer`, `structured_summary_forge`
- Runtime task APIs: `/api/runtime/tasks`, `/api/runtime/tasks/events`, `/api/runtime/tasks/{task_id}`
- Runtime and scheduler health: `/api/runtime/health`, `/api/agent/scheduler/status`
- Agent event stream: `/api/agent/events`
- Deep app APIs: `/api/deep/session-forge/*`, `/api/chronicle/passive/*`, `/api/wiki/lint/*`, `/api/wiki/compaction/*`

### 15.3 Required Web App Surfaces

1. Runtime Operations Center
  - real-time cards for active agents, active tasks, queue depth, blocked work, waiting approvals
  - provider and scheduler health indicators
2. Live Agent and Task Graph
  - graph showing which agents spawned which tasks
  - parent-child chain and current state coloring
3. Long-Running Work Queue
  - columns for queued, running, blocked, waiting approval, completed, failed, cancelled
  - filtering by session_id, trace_id, agent_id
4. Background Continuity Strip
  - persistent status bar showing work continues after navigation/refresh
  - one-click jump back to task detail
5. Trace and Event Timeline Drawer
  - per-task event replay from creation to completion
  - inline errors, retries, cancellation reason, and quality-loop events

### 15.4 Required Event Model for Deep Observability

Unify UI event handling around one normalized event schema:

- `event_id`
- `event_type`
- `timestamp`
- `trace_id`
- `session_id`
- `agent_id`
- `task_id`
- `parent_task_id`
- `state`
- `note`
- `payload`

This enables one timeline renderer for both `/api/runtime/tasks/events` and `/api/agent/events`.

### 15.5 Frontend TODO Backlog (Execution Order)

1. Build SSE client with reconnect/backoff and stale-stream detection.
2. Build global runtime store to merge task and agent streams.
3. Build Runtime Operations Center and queue counters.
4. Build task board with lifecycle columns and state badges.
5. Build graph visualization for agent-task lineage.
6. Build task detail with trace timeline and artifact links.
7. Add persisted resume state to restore active work after reload.
8. Add polling fallback when SSE is unavailable.
9. Add user-facing notifications for blocked, failed, and approval-waiting states.

### 15.6 UX Success Metrics

- Users should see first live status within 500 ms.
- No silent waiting longer than 2 seconds without visible progress signal.
- Reconnect after network drop should recover live status in under 3 seconds.
- At least 95% of long-running tasks should show full lifecycle in the timeline.

---

## Closing: The Long-Term Vision

After two years of this system running consistently, something remarkable becomes available that has never existed before.

A person can sit down with their Presence Agent and ask: "Who was I two years ago? How have I changed? What did I learn? What did I lose? What am I still carrying that I thought I'd let go of?"

And the system — with full provenance, with evidence from thousands of sessions, with the Chronicle's captured moments, with the Mirror's patient observations — can answer honestly, specifically, and with genuine insight.

Not "you've grown in these ways" as a generic affirmation. But: "Here is your actual trajectory. Here are the moments that changed you. Here is what your actions showed you valued, separate from what you said you valued. Here is the gap that keeps appearing. Here is the strength you keep underestimating in yourself."

That conversation — that level of honest, evidence-grounded, lovingly-assembled self-knowledge — is what this system exists to make possible.

It is the most intimate technology ever built, and the most private. Every byte lives locally. No cloud has it. No algorithm sells it. It belongs entirely to you, and it is studying nothing but you, in service of no one but you.

---

*Cortex Lab Applications v1.0 — The Implications of Owning Your Own Mind*  
*Built on: OpenClaw (personal assistant + device control) + Paperclip (goal-aware agent orchestration) + NemoClaw (sandboxed secure execution) + Agentic RAG v4.0 (tiered retrieval + LLM wiki) + 17-Agent Personal Intelligence Runtime*
