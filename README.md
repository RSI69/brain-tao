THIS MODEL IS INCOMPLETE BUT IS STEPS AWAY FROM A BITTENSOR BASED MULTI LLM

# Multi-LLM Bittensor Backend

This is a fully decentralized, programmable AI backend using the Bittensor network.  
It uses wallet-based authorization, routes requests to top Bittensor miners, and streams live responses through your own API.

You can use it as a personal LLM engine or open it up for other users to call via browser, bots, or dApps.

---

## What It Does

- Wallet-based login (MetaMask)
- TAO-signed prompts and gas estimate
- Routes to Bittensor chat subnet (no OpenAI)
- Streams final LLM response
- Auto-judges and synthesizes outputs
- Built-in memory and RAG support
- Fully customizable backend

---

## Prerequisites

This guide assumes you're using **Windows**.  
It will work on macOS and Linux too, just adjust paths or terminal syntax.

You’ll install:

1. [Python 3.10](https://www.python.org/downloads/release/python-3100/)
2. pip (Python’s package manager)
3. Git
4. Your own Bittensor wallet

---

## Step-by-Step Setup

### 1. Download and Install Python 3.10

- Go to: https://www.python.org/downloads/release/python-3100/
- Download the **Windows installer**
- Check **“Add Python to PATH”** before clicking “Install Now”

Verify install:
```bash
python --version
# Expected: Python 3.10.0
```

### 2. Open Terminal

Press Win + R, type cmd, press Enter.

### 3. Clone the Code

```bash
git clone https://github.com/YOUR_USERNAME/multi-llm-backend.git
cd multi-llm-backend
```

### 4. Create Python 3.10 Environment

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

You should now see:
```bash
(venv) C:\Users\you\multi-llm-backend>
```

### 5. Install All Dependencies

```bash
pip install -r requirements.txt
```

This will install FastAPI, Bittensor, llama-cpp, Transformers, and other backend packages.

### 6. Create a Bittensor Wallet

```bash
btcli wallet new --wallet.name default --wallet.hotkey default
```

Backup your coldkey mnemonic (seed words). You’ll use this wallet to pay TAO per query and route requests through miners.

### 7. Launch the Server

```bash
uvicorn multi_llm_backend:app --host 0.0.0.0 --port 8000
```

Your backend runs at: http://localhost:8000

---

## Deploy the Frontend (HTML)

Go to https://app.fleek.co  
Click "New Site" → "Deploy with GitHub"  
Choose this repo

Build settings:

- Framework: Other  
- Build Command: (leave blank)  
- Publish Directory: ./

Result:  
Live at something like `https://multi-llm-yourname.on-fleek.app`

---

## How to Use

1. Visit your frontend  
2. Connect MetaMask  
3. Type a prompt → Sign → Streamed response shows live  
4. TAO usage tracked per session

---

## Notes

- Each prompt is authorized with a signed TAO request  
- All compute happens on decentralized miners (no OpenAI, no API keys)  
- TAO balance is required in your wallet  

---

## Optional: Persistent Memory + RAG

- Vector DB: uses DuckDB  
- Static assets: `./static`  
- Metrics: `/metrics`  

---

## Troubleshooting

**"python is not recognized"**  
→ You didn't add Python to PATH. Reinstall and check that box.

**"uvicorn: command not found"**  
→ Make sure venv is active and dependencies installed.

**MetaMask errors**  
→ Ensure you're on Ethereum Mainnet or EVM-compatible testnet with funds.

---

## Project Structure

- `multi_llm_backend.py`: Backend logic (routing, synthesis, safety, cost)  
- `index.html`: Frontend for wallet prompts and streaming responses  
- `requirements.txt`: Backend Python dependencies  
- `models/`: Store quantized GGUF models (e.g., Llama 3, Mistral, Vicuna)  
- `static/`: Host frontend assets  
- `memory.db`: DuckDB for RAG and memory storage  

---

## Included Code Files

### Frontend

Save as `index.html` in the root directory.  
View full HTML source above.

### Backend

Save as `multi_llm_backend.py`.  
View complete Python backend code above.

---

## Final Notes

You now have a wallet-authenticated, streaming, decentralized LLM backend powered by Bittensor, customizable for local or hosted usage.
