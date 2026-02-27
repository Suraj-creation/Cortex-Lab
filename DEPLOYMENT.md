# 🚀 Cortex Lab — Production Deployment Guide

## Complete Cloud Deployment: Google Cloud GPU + vLLM + Vercel

---

## 📋 What You Have (Current State)

| Component | Details |
|-----------|---------|
| **Model** | DeepSeek-R1-Distill-Qwen-7B, 15-stage fine-tuned (Qwen2ForCausalLM) |
| **Model Size** | 15GB (bfloat16, single `model.safetensors`) |
| **Architecture** | 28 layers, 3584 hidden, 28 heads, 4 KV heads, 152064 vocab |
| **Backend** | FastAPI + Agentic RAG + STT/TTS (1193 lines, 30+ endpoints) |
| **Frontend** | Next.js 15 + React 19 + Tailwind (15 components) |
| **RAG Stack** | BGE-large-1024d embeddings + CrossEncoder reranker + FAISS + DuckDB + NetworkX |
| **Voice** | faster-whisper (STT) + Piper-TTS + SpeechBrain speaker ID + WebRTC VAD |
| **API Endpoints** | `/api/chat`, `/api/rag/chat`, `/api/memories/*`, `/api/graph`, `/api/ambient/*`, `/api/voice/*`, `/api/tts/*`, `/ws/ambient` |

## 🏗️ Target Production Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         USERS                                │
│                    (Browser / Mobile)                         │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTPS
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              VERCEL (Frontend)                               │
│  Next.js 15 · React 19 · Tailwind                           │
│  - ChatPanel + Voice UI                                     │
│  - Knowledge Graph Visualizer                                │
│  - Memory Browser + RAG Dashboard                            │
│  - Ambient Listening Panel                                   │
│                                                              │
│  API calls → https://api.cortexlab.YOUR_DOMAIN.com           │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTPS (API)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│        GOOGLE CLOUD VM  (Backend + vLLM)                     │
│        Ubuntu 22.04 · NVIDIA L4 (24GB)                       │
│                                                              │
│  ┌─────────────────────────────────────────────┐             │
│  │  Process 1: vLLM Server (port 8001)         │             │
│  │  - DeepSeek-R1-7B fine-tuned                │             │
│  │  - OpenAI-compatible API                    │             │
│  │  - Tensor parallel, continuous batching     │             │
│  │  - ~14GB VRAM                               │             │
│  └─────────────────────────────────────────────┘             │
│                         ▲                                    │
│                         │ localhost:8001                      │
│  ┌─────────────────────────────────────────────┐             │
│  │  Process 2: FastAPI Backend (port 8000)     │             │
│  │  - Agentic RAG (Orchestrator + Agents)      │             │
│  │  - Memory Engine (Ingest + Retrieve)        │             │
│  │  - BGE-large embeddings (CPU)               │             │
│  │  - CrossEncoder reranker (CPU)              │             │
│  │  - FAISS vectors + DuckDB + NetworkX        │             │
│  │  - STT (faster-whisper) + TTS (Piper)       │             │
│  │  - WebSocket for ambient listening          │             │
│  └─────────────────────────────────────────────┘             │
│                                                              │
│  Nginx reverse proxy (port 443) → port 8000                  │
│  SSL via Let's Encrypt (certbot)                             │
└──────────────────────────────────────────────────────────────┘
```

> **Why separate vLLM from backend?**
> vLLM handles GPU inference with continuous batching, paged attention, and tensor parallelism.
> Your backend handles RAG logic, memory, voice — all CPU-bound. Mixing them in one process wastes GPU memory on non-inference tasks and prevents scaling.

---

## 💰 Google Cloud Free Tier Strategy ($300 Credits)

Google gives **$300 free credits** for 90 days on new accounts. Here's how to maximize them:

| Resource | Monthly Cost | 90-Day Total | Notes |
|----------|-------------|-------------|-------|
| **NVIDIA L4 GPU** (n1-standard-8 + 1×L4) | ~$250/mo on-demand | ~$750 | ⚠️ Use **spot/preemptible** for ~$75/mo |
| **NVIDIA T4 GPU** (n1-standard-8 + 1×T4) | ~$180/mo on-demand | ~$540 | Cheaper, 16GB VRAM (tight for FP16) |
| **Boot Disk** (100GB SSD) | ~$10/mo | ~$30 | |
| **Static IP** | ~$3/mo | ~$9 | |
| **Egress** | ~$5/mo | ~$15 | |

### ✅ Recommended: NVIDIA L4 Spot Instance
- **Cost**: ~$75-85/month (spot pricing)
- **VRAM**: 24GB (comfortable for 7B FP16)
- **90-day total**: ~$250 → **fits within $300 credits**
- **Caveat**: Spot VMs can be preempted (add auto-restart script)

### Budget Breakdown for $300 Credits:
```
L4 Spot VM (90 days):    $250
Boot disk (100GB):        $30
Static IP + egress:       $20
                         ─────
Total:                   $300  ✅ Fits perfectly
```

---

## 🔧 STEP-BY-STEP DEPLOYMENT

---

### PHASE 1: Prepare Model for Cloud Upload

#### Step 1.1 — Verify Model Files Locally

Your merged model is at `fine_tuned/stage15_spin/merged/`. Verify it has everything:

```bash
ls -la fine_tuned/stage15_spin/merged/
# Required files:
#   config.json              (1.4 KB)  ✅
#   generation_config.json   (180 B)   ✅
#   model.safetensors        (15 GB)   ✅
#   tokenizer.json           (11 MB)   ✅
#   tokenizer_config.json    (393 B)   ✅
#   chat_template.jinja      (2.2 KB)  ✅
```

#### Step 1.2 — Test with vLLM Locally (CRITICAL — Do This Before Cloud)

```bash
# Install vLLM locally
pip install vllm

# Test serving your model
python -m vllm.entrypoints.openai.api_server \
    --model ./fine_tuned/stage15_spin/merged \
    --host 127.0.0.1 \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85

# In another terminal, test it
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./fine_tuned/stage15_spin/merged",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool
```

> ⚠️ **If this fails locally, DO NOT proceed to cloud.** Fix it first.
> Common issues: missing `tokenizer.model`, wrong `config.json` architecture name, OOM.

#### Step 1.3 — Upload Model to HuggingFace (Best Option for Cloud Transfer)

HuggingFace Hub is the fastest way to get your 15GB model to any cloud VM.

```bash
# Install HF CLI
pip install huggingface_hub[cli]

# Login (create account at huggingface.co if needed)
huggingface-cli login
# Paste your HF token (get from huggingface.co/settings/tokens)

# Create a PRIVATE repo and upload
huggingface-cli repo create cortex-lab-deepseek-r1-7b --type model --private

# Upload the merged model
huggingface-cli upload YOUR_HF_USERNAME/cortex-lab-deepseek-r1-7b \
    ./fine_tuned/stage15_spin/merged/ . \
    --repo-type model

# This uploads ~15GB. On 100Mbps: ~25 minutes.
```

**Why HuggingFace?**
- `vllm serve YOUR_HF_USERNAME/cortex-lab-deepseek-r1-7b` downloads directly on VM
- No manual SCP of 15GB files
- Version controlled
- Can share with collaborators later

#### Step 1.4 — Alternative: Upload via Google Cloud Storage (GCS)

If you prefer not to use HuggingFace:

```bash
# Install Google Cloud SDK (if not installed)
curl https://sdk.cloud.google.com | bash
gcloud init

# Create a bucket
gsutil mb -l us-central1 gs://cortex-lab-models

# Upload model (~15GB)
gsutil -m cp -r fine_tuned/stage15_spin/merged/ gs://cortex-lab-models/deepseek-r1-7b/

# On the VM later, download:
gsutil -m cp -r gs://cortex-lab-models/deepseek-r1-7b/ ~/model/
```

---

### PHASE 2: Create Google Cloud GPU VM

#### Step 2.1 — Set Up Google Cloud Account

1. Go to [https://cloud.google.com](https://cloud.google.com)
2. Sign in with Google account
3. Activate the **$300 free trial** (requires credit card, but won't charge)
4. Create a new project: `cortex-lab`

#### Step 2.2 — Request GPU Quota (IMPORTANT — Do This First)

New GCP accounts have **zero GPU quota** by default. You must request it:

1. Go to: **IAM & Admin → Quotas & System Limits**
2. Filter by: `GPUs (all regions)`
3. Find: `NVIDIA L4 GPUs` (or `NVIDIA T4 GPUs`)
4. Click **Edit Quotas** → Request **1 GPU**
5. Provide justification: *"AI research project — inference server for fine-tuned 7B language model"*
6. **Wait time**: Usually approved within 1-24 hours

> ⚠️ **This is the #1 blocker for new accounts.** Start this immediately.

#### Step 2.3 — Create the VM Instance

**Via Google Cloud Console (Web UI):**

1. Go to: **Compute Engine → VM Instances → Create Instance**

2. **Configuration:**

| Setting | Value |
|---------|-------|
| **Name** | `cortex-lab-gpu` |
| **Region** | `us-central1` (cheapest for GPUs) |
| **Zone** | `us-central1-a` |
| **Machine type** | `n1-standard-8` (8 vCPU, 30GB RAM) |
| **GPU** | 1 × NVIDIA L4 |
| **VM provisioning** | **Spot** (saves 60-70%) |
| **Boot disk OS** | Ubuntu 22.04 LTS |
| **Boot disk size** | 100GB SSD |
| **Boot disk type** | Balanced persistent disk |
| **Firewall** | ✅ Allow HTTP, ✅ Allow HTTPS |

3. Click **Create**

**Or via `gcloud` CLI (copy-paste):**

```bash
# Install gcloud CLI first: https://cloud.google.com/sdk/docs/install

gcloud config set project YOUR_PROJECT_ID

gcloud compute instances create cortex-lab-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --metadata=install-nvidia-driver=True \
    --tags=http-server,https-server

# Reserve a static IP (so your backend URL doesn't change)
gcloud compute addresses create cortex-lab-ip --region=us-central1
gcloud compute instances add-access-config cortex-lab-gpu \
    --zone=us-central1-a \
    --access-config-name="External NAT" \
    --address=$(gcloud compute addresses describe cortex-lab-ip --region=us-central1 --format='value(address)')
```

#### Step 2.4 — Create Firewall Rules

```bash
# Allow backend API access (port 8000)
gcloud compute firewall-rules create allow-cortex-backend \
    --allow=tcp:8000 \
    --target-tags=http-server \
    --description="Cortex Lab backend API"

# Allow HTTPS (port 443) — for Nginx SSL
gcloud compute firewall-rules create allow-cortex-https \
    --allow=tcp:443 \
    --target-tags=https-server \
    --description="Cortex Lab HTTPS"

# DO NOT expose port 8001 (vLLM) — it should only be accessible internally
```

---

### PHASE 3: Set Up the GPU VM

#### Step 3.1 — SSH into the VM

```bash
gcloud compute ssh cortex-lab-gpu --zone=us-central1-a
```

#### Step 3.2 — Verify GPU and Install NVIDIA Drivers

```bash
# Check if NVIDIA driver installed automatically
nvidia-smi

# If NOT installed, install manually:
sudo apt update
sudo apt install -y linux-headers-$(uname -r)

# Install NVIDIA driver
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-driver-550
sudo reboot

# After reboot, verify:
nvidia-smi
# Should show: NVIDIA L4 | 24GB VRAM
```

#### Step 3.3 — Install Python Environment

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git nginx certbot python3-certbot-nginx

# Create project directory
mkdir -p ~/cortex-lab && cd ~/cortex-lab

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, GPU: NVIDIA L4
```

#### Step 3.4 — Install vLLM

```bash
pip install vllm
```

#### Step 3.5 — Download Your Model

**Option A — From HuggingFace (recommended):**

```bash
pip install huggingface_hub[cli]
huggingface-cli login  # paste your token

# Download to local disk
huggingface-cli download YOUR_HF_USERNAME/cortex-lab-deepseek-r1-7b \
    --local-dir ~/cortex-lab/model \
    --local-dir-use-symlinks False
```

**Option B — From Google Cloud Storage:**

```bash
gsutil -m cp -r gs://cortex-lab-models/deepseek-r1-7b/ ~/cortex-lab/model/
```

**Option C — Direct SCP from your local machine:**

```bash
# From your LOCAL machine (not the VM):
gcloud compute scp --recurse \
    ./fine_tuned/stage15_spin/merged/ \
    cortex-lab-gpu:~/cortex-lab/model/ \
    --zone=us-central1-a

# ⚠️ This transfers 15GB. On typical upload speeds: 1-4 hours.
# HuggingFace or GCS is MUCH faster.
```

#### Step 3.6 — Test vLLM on the VM

```bash
cd ~/cortex-lab
source venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
    --model ~/cortex-lab/model \
    --host 127.0.0.1 \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.88 \
    --enforce-eager

# Test (in another SSH session):
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/cortex-lab/model",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 200
  }' | python3 -m json.tool
```

**Expected output**: A proper JSON response with model-generated text. If this works, your model serving is ready.

---

### PHASE 4: Deploy the Backend

#### Step 4.1 — Upload Backend Code

```bash
# From your LOCAL machine:
# First, push your code to GitHub (private repo)
cd /home/btech01_06/Desktop/DeepLearning/Cortex-Lab
git add -A
git commit -m "Production deployment"
git push origin main

# On the VM:
cd ~/cortex-lab
git clone https://github.com/Suraj-creation/Cortex-Lab.git app
cd app/backend
```

#### Step 4.2 — Install Backend Dependencies

```bash
source ~/cortex-lab/venv/bin/activate
cd ~/cortex-lab/app/backend

pip install -r requirements.txt

# Install additional production dependencies
pip install gunicorn httptools uvloop
```

#### Step 4.3 — Modify Backend to Use vLLM Instead of Local Model Loading

This is the **critical architecture change**. Currently, `server.py` loads the model directly using `transformers`. In production, the backend should call vLLM's OpenAI-compatible API instead.

Create a new file `backend/src/llm/vllm_client.py`:

```python
"""
vLLM Client — Replaces direct model loading with HTTP calls to vLLM server.
This is the production inference backend.
"""

import os
import httpx
import asyncio
from typing import AsyncGenerator, List, Dict, Optional

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "/root/cortex-lab/model")


class VLLMClient:
    """Async HTTP client for vLLM's OpenAI-compatible API."""

    def __init__(self):
        self.base_url = VLLM_BASE_URL
        self.model_name = VLLM_MODEL_NAME
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> dict:
        """Non-streaming generation."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "stream": False,
            "repetition_penalty": 1.15,
        }
        resp = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
        }

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        """Streaming token generation via SSE."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "stream": True,
            "repetition_penalty": 1.15,
        }
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    import json
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token

    async def health_check(self) -> bool:
        """Check if vLLM server is responsive."""
        try:
            resp = await self.client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self.client.aclose()
```

#### Step 4.4 — Create Production Server Wrapper

Create `backend/server_production.py`:

```python
"""
Production server — uses vLLM for inference instead of loading model in-process.
Loads only the RAG pipeline (embeddings, vector store, agents) locally.
Model inference is delegated to the vLLM server running on port 8001.
"""

import os

# Tell the engine to use vLLM mode
os.environ["CORTEX_INFERENCE_MODE"] = "vllm"
os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:8001"
os.environ["VLLM_MODEL_NAME"] = os.environ.get("VLLM_MODEL_NAME", "/root/cortex-lab/model")

# Disable local model loading
os.environ["SKIP_LOCAL_MODEL"] = "true"

from server import app  # Import the FastAPI app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_production:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=120,
        workers=1,  # Single worker — shares GPU context
        log_level="info",
    )
```

#### Step 4.5 — Modify `server.py` for Production Mode

Add this block near the top of `server.py` (after imports), so it can run in **both** local and production modes:

```python
# ── Production Mode Detection ────────────────────────────────────────────────
PRODUCTION_MODE = os.environ.get("CORTEX_INFERENCE_MODE") == "vllm"
SKIP_LOCAL_MODEL = os.environ.get("SKIP_LOCAL_MODEL", "false").lower() == "true"

if PRODUCTION_MODE:
    print("  🏭 PRODUCTION MODE: Using vLLM for inference")
    from src.llm.vllm_client import VLLMClient
    vllm_client = VLLMClient()
```

Then in the model loading section, wrap it:

```python
if not SKIP_LOCAL_MODEL:
    # ... existing model loading code ...
    pass
else:
    model = None
    tokenizer = None
    print("  ⏭️  Skipping local model load (using vLLM)")
```

And in the streaming endpoints, add a vLLM code path:

```python
if PRODUCTION_MODE:
    # Use vLLM streaming
    async for token in vllm_client.stream_generate(messages, ...):
        yield f"data: {json.dumps({'id': msg_id, 'delta': token})}\n\n"
else:
    # ... existing local model streaming code ...
```

> **Note**: The full refactoring is ~50 lines of changes. The key principle is:
> local dev uses `transformers` directly, production uses vLLM HTTP calls.

#### Step 4.6 — Create Systemd Services

**vLLM Service** — `/etc/systemd/system/cortex-vllm.service`:

```ini
[Unit]
Description=Cortex Lab vLLM Inference Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/cortex-lab
Environment=PATH=/root/cortex-lab/venv/bin:/usr/local/bin:/usr/bin
ExecStart=/root/cortex-lab/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /root/cortex-lab/model \
    --host 127.0.0.1 \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.88 \
    --enforce-eager
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Backend Service** — `/etc/systemd/system/cortex-backend.service`:

```ini
[Unit]
Description=Cortex Lab FastAPI Backend (RAG + Agents + Voice)
After=cortex-vllm.service
Requires=cortex-vllm.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/cortex-lab/app/backend
Environment=PATH=/root/cortex-lab/venv/bin:/usr/local/bin:/usr/bin
Environment=CORTEX_INFERENCE_MODE=vllm
Environment=VLLM_BASE_URL=http://127.0.0.1:8001
Environment=SKIP_LOCAL_MODEL=true
ExecStart=/root/cortex-lab/venv/bin/python server_production.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Enable and start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable cortex-vllm cortex-backend
sudo systemctl start cortex-vllm

# Wait for vLLM to load model (~30-60s)
sleep 60
curl -s http://localhost:8001/health

# Then start backend
sudo systemctl start cortex-backend

# Check status
sudo systemctl status cortex-vllm
sudo systemctl status cortex-backend

# View logs
journalctl -u cortex-vllm -f
journalctl -u cortex-backend -f
```

---

### PHASE 5: Set Up SSL + Nginx Reverse Proxy

#### Step 5.1 — Get a Domain Name

You need a domain for HTTPS. Options:

| Option | Cost | Recommendation |
|--------|------|----------------|
| **Namecheap** `.me` domain | $3-5/year | Best value |
| **Google Domains** | $12/year | Easy GCP integration |
| **Freenom** `.tk` / `.ml` | Free | Unreliable, not recommended |
| **No domain — use IP directly** | Free | Works, but no SSL |

Point your domain's **A record** to your VM's static IP:
```
api.cortexlab.com → YOUR_VM_STATIC_IP
```

#### Step 5.2 — Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/cortex-lab
```

```nginx
server {
    listen 80;
    server_name api.cortexlab.YOUR_DOMAIN.com;

    # Redirect HTTP → HTTPS (after certbot runs)
    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name api.cortexlab.YOUR_DOMAIN.com;

    # SSL certs — certbot will fill these in
    # ssl_certificate /etc/letsencrypt/live/api.cortexlab.YOUR_DOMAIN.com/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/api.cortexlab.YOUR_DOMAIN.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Max upload size (for voice/audio uploads)
    client_max_body_size 50M;

    # Backend API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE streaming support
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # WebSocket proxy (ambient listening)
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
    }
}
```

```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/cortex-lab /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get SSL certificate
sudo certbot --nginx -d api.cortexlab.YOUR_DOMAIN.com

# Auto-renew
sudo certbot renew --dry-run
```

---

### PHASE 6: Deploy Frontend on Vercel

#### Step 6.1 — Update Frontend API Configuration

Modify `frontend/next.config.js` to point to your cloud backend:

```javascript
const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  outputFileTracingRoot: path.join(__dirname),

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: process.env.NEXT_PUBLIC_API_URL
          ? `${process.env.NEXT_PUBLIC_API_URL}/api/:path*`
          : "http://localhost:8000/api/:path*",  // Local dev fallback
      },
    ];
  },

  async headers() {
    return [
      {
        source: "/api/:path*",
        headers: [
          { key: "Connection", value: "keep-alive" },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
```

Also update `frontend/src/lib/api.ts` to use an environment variable:

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
```

#### Step 6.2 — Push Frontend to GitHub

```bash
cd /home/btech01_06/Desktop/DeepLearning/Cortex-Lab
git add -A
git commit -m "Production deployment config"
git push origin main
```

#### Step 6.3 — Deploy on Vercel

1. Go to [https://vercel.com](https://vercel.com) → Sign up with GitHub
2. Click **"Add New Project"**
3. Import your `Cortex-Lab` repository
4. Configure:

| Setting | Value |
|---------|-------|
| **Framework Preset** | Next.js |
| **Root Directory** | `frontend` |
| **Build Command** | `npm run build` |
| **Output Directory** | `.next` |

5. **Environment Variables** (CRITICAL):

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_URL` | `https://api.cortexlab.YOUR_DOMAIN.com` |

6. Click **Deploy**

#### Step 6.4 — Verify Deployment

```bash
# Test the deployed frontend
curl -s https://your-app.vercel.app

# Test API proxy (frontend → your GCP backend)
curl -s https://your-app.vercel.app/api/health
```

---

### PHASE 7: Connect Everything End-to-End

#### Step 7.1 — CORS Configuration

Update `server.py` CORS to allow your Vercel domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-app.vercel.app",
        "https://cortexlab.YOUR_DOMAIN.com",
        "http://localhost:3000",  # Local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### Step 7.2 — End-to-End Test Checklist

Run these in order:

```bash
# 1. vLLM health
curl -s http://YOUR_VM_IP:8001/health
# Expected: 200 OK

# 2. Backend health (via Nginx)
curl -s https://api.cortexlab.YOUR_DOMAIN.com/api/health
# Expected: {"status": "ready", "model_loaded": true, ...}

# 3. RAG chat
curl -s https://api.cortexlab.YOUR_DOMAIN.com/api/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "stream": false,
    "use_rag": true
  }' | python3 -m json.tool

# 4. Frontend loads
curl -s -o /dev/null -w "%{http_code}" https://your-app.vercel.app
# Expected: 200

# 5. Open in browser and test full chat
```

---

## ⚡ PERFORMANCE OPTIMIZATION CHECKLIST

### Model / vLLM

| Check | How to Verify | Target |
|-------|--------------|--------|
| Model loads under 60s | `journalctl -u cortex-vllm` | < 60s |
| GPU memory usage | `nvidia-smi` | < 90% of 24GB |
| Streaming enabled | Test with `stream: true` | First token < 1s |
| Continuous batching | Multiple concurrent requests | All handled |
| Max sequence length | Set `--max-model-len` | 4096 (saves VRAM) |

### Backend / RAG

| Check | How to Verify | Target |
|-------|--------------|--------|
| RAG engine init | Server logs | < 15s |
| Embedding model (CPU) | BGE-large on CPU | < 100ms/query |
| Vector search | FAISS retrieval | < 50ms |
| Full RAG pipeline | `/api/rag/chat` | < 3s total |
| DuckDB persistent | Check `data/cortex.duckdb` | Auto-saves |
| Knowledge graph | `data/graph/` | Auto-saves on shutdown |

### Voice

| Check | How to Verify | Target |
|-------|--------------|--------|
| STT (Whisper) | `/api/voice/query` | < 2s for 5s audio |
| TTS (Piper) | `/api/tts/synthesize` | < 1s for short text |
| WebSocket | `/ws/ambient` | Stable connection |

### Cost Control

| Check | Action |
|-------|--------|
| Stop VM when idle | `gcloud compute instances stop cortex-lab-gpu --zone=us-central1-a` |
| Start when needed | `gcloud compute instances start cortex-lab-gpu --zone=us-central1-a` |
| Auto-shutdown script | See below |
| Monitor credits | GCP → Billing → Budget alerts |

---

## 🛡️ Auto-Shutdown Script (Save Credits)

Create `~/cortex-lab/auto-shutdown.sh`:

```bash
#!/bin/bash
# Auto-shutdown VM if no API requests for 30 minutes
# Run via cron every 5 minutes

LOGFILE="/var/log/cortex-activity.log"
IDLE_THRESHOLD=1800  # 30 minutes in seconds

# Check last API request time from nginx logs
LAST_REQUEST=$(stat -c %Y /var/log/nginx/access.log 2>/dev/null || echo 0)
NOW=$(date +%s)
IDLE_TIME=$((NOW - LAST_REQUEST))

if [ $IDLE_TIME -gt $IDLE_THRESHOLD ]; then
    echo "$(date): Idle for ${IDLE_TIME}s — shutting down" >> $LOGFILE
    sudo shutdown -h now
fi
```

```bash
chmod +x ~/cortex-lab/auto-shutdown.sh

# Add to crontab
crontab -e
# Add this line:
# */5 * * * * /root/cortex-lab/auto-shutdown.sh
```

---

## 📁 Production File Structure on VM

```
~/cortex-lab/
├── model/                          # Your fine-tuned DeepSeek R1 7B
│   ├── config.json
│   ├── model.safetensors           # 15GB
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── app/                            # Cloned from GitHub
│   ├── backend/
│   │   ├── server.py               # Main FastAPI server
│   │   ├── server_production.py    # Production wrapper (vLLM mode)
│   │   ├── requirements.txt
│   │   ├── data/                   # Persistent storage
│   │   │   ├── cortex.duckdb       # Metadata + conversations
│   │   │   ├── vectors/            # FAISS embeddings
│   │   │   └── graph/              # Knowledge graph
│   │   └── src/
│   │       ├── engine.py           # RAG engine
│   │       ├── agents/             # Orchestrator + specialized agents
│   │       ├── retrieval/          # Hybrid retriever + query engine
│   │       ├── storage/            # Vector store + metadata + graph
│   │       ├── ingestion/          # Memory ingestion pipeline
│   │       ├── ambient/            # STT + TTS + speaker ID
│   │       └── llm/
│   │           └── vllm_client.py  # vLLM HTTP client (production)
│   └── frontend/                   # Deployed on Vercel (not needed on VM)
└── venv/                           # Python virtual environment
```

---

## 🔄 CI/CD — Auto-Deploy on Git Push

### Backend (on VM)

Create `~/cortex-lab/deploy.sh`:

```bash
#!/bin/bash
# Pull latest code and restart backend
cd ~/cortex-lab/app
git pull origin main

source ~/cortex-lab/venv/bin/activate
cd backend
pip install -r requirements.txt --quiet

sudo systemctl restart cortex-backend
echo "✅ Backend redeployed at $(date)"
```

### Frontend (Vercel)

Vercel auto-deploys on every `git push` to `main`. No manual action needed.

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| GPU quota denied | New account | Submit quota request again with more detail |
| vLLM OOM | Model too large for GPU | Use `--dtype float16` or `--quantization awq` |
| CORS errors | Frontend domain not in allow_origins | Update CORS in server.py |
| SSE streaming broken | Nginx buffering enabled | Add `proxy_buffering off;` |
| WebSocket disconnects | Nginx timeout | Set `proxy_read_timeout 3600s;` |
| Spot VM preempted | Google reclaimed capacity | VM auto-restarts via systemd |
| SSL cert expired | Certbot didn't auto-renew | `sudo certbot renew` |
| Backend can't reach vLLM | vLLM not started yet | Check `systemctl status cortex-vllm` |

### Debug Commands

```bash
# GPU status
nvidia-smi

# vLLM logs
journalctl -u cortex-vllm -f --no-pager | tail -50

# Backend logs
journalctl -u cortex-backend -f --no-pager | tail -50

# Nginx logs
tail -f /var/log/nginx/error.log

# Test vLLM directly
curl http://localhost:8001/v1/models

# Test backend directly
curl http://localhost:8000/api/health

# Check disk space
df -h

# Check memory
free -h
```

---

## 📊 Expected Performance (L4 GPU)

| Metric | Expected Value |
|--------|---------------|
| **Model load time** | 30-45s |
| **Time to first token** | 0.3-0.8s |
| **Token generation speed** | 40-60 tokens/sec |
| **RAG pipeline latency** | 1.5-3s total |
| **Concurrent users** | 5-10 (with continuous batching) |
| **GPU memory usage** | ~14GB / 24GB |
| **Monthly cost (spot)** | ~$75-85 |

---

## 🎯 Quick Reference Commands

```bash
# ── VM Management ──
gcloud compute instances start cortex-lab-gpu --zone=us-central1-a
gcloud compute instances stop cortex-lab-gpu --zone=us-central1-a
gcloud compute ssh cortex-lab-gpu --zone=us-central1-a

# ── Service Management (on VM) ──
sudo systemctl start cortex-vllm cortex-backend
sudo systemctl stop cortex-vllm cortex-backend
sudo systemctl restart cortex-backend
sudo systemctl status cortex-vllm cortex-backend

# ── Logs ──
journalctl -u cortex-vllm -f
journalctl -u cortex-backend -f

# ── Deploy Update ──
cd ~/cortex-lab/app && git pull && sudo systemctl restart cortex-backend

# ── Monitor Costs ──
# GCP Console → Billing → Reports
```

---

## ✅ DEPLOYMENT CHECKLIST

```
PREPARATION
[ ] Model tested with vLLM locally
[ ] Model uploaded to HuggingFace (private)
[ ] Code pushed to GitHub
[ ] GCP account created with $300 credits
[ ] GPU quota requested and approved

VM SETUP
[ ] VM created (n1-standard-8 + L4 GPU, spot)
[ ] Static IP reserved
[ ] NVIDIA drivers verified (nvidia-smi)
[ ] Python 3.11 + venv created
[ ] vLLM installed and tested
[ ] Model downloaded to VM

SERVICES
[ ] vLLM systemd service running
[ ] Backend systemd service running
[ ] Nginx configured with SSL
[ ] Certbot SSL certificate obtained
[ ] Firewall rules set (8000, 443)

FRONTEND
[ ] next.config.js updated for production API URL
[ ] Vercel project created
[ ] Environment variables set (NEXT_PUBLIC_API_URL)
[ ] Frontend deployed and accessible

INTEGRATION
[ ] CORS updated for Vercel domain
[ ] End-to-end chat working
[ ] RAG retrieval working
[ ] Voice query working
[ ] Knowledge graph loading
[ ] Auto-shutdown script active
[ ] Billing alert set at $250

PRODUCTION
[ ] Auto-restart on crash (systemd)
[ ] Log rotation configured
[ ] Backup script for DuckDB + vectors
[ ] CI/CD deploy script ready
```

---

> **🏆 What You've Built**: A complete AI inference stack — fine-tuned model served via vLLM on cloud GPU, agentic RAG with memory persistence, real-time voice interface, deployed as a production web application. This is the same architecture used by AI startups serving millions of users.
