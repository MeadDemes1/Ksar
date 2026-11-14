# Ksar 0.3.0

Local assistant with persistent branching memory, controlled tools, resumable tasks, and **dynamic multi-model orchestration**. Python 3.10+, no pip packages.

## Dynamic Chip System
Ksar now supports multiple AI models ("chips") and automatically switches between them based on task complexity and your system's resources.
- **Standard Chip:** Handles everyday tasks (default: `qwen3:8b`).
- **Advanced Chip:** Activated for complex planning, debugging, or analysis (e.g. `qwen3:14b`; `qwen3:32b` needs about 21 GB of RAM and is slow without a GPU; 70B models do not fit in 32 GB).
- **Resource-Aware:** Monitors free RAM (an integrated GPU shares system RAM) and only loads advanced chips if resources permit.
- **Persistent Loading:** Chips stay loaded in memory based on your retention preferences to minimize latency.

## Setup and Model Configuration

Ksar relies on Ollama to run local AI models (referred to as "chips"). Before running Ksar, you must install Ollama and download the models you intend to use.

1. Install Ollama
Download and install Ollama from ollama.com. Ensure the Ollama background service is running before starting Ksar.

2. Pull Required Models
Open your terminal or command prompt and pull the models you want Ksar to use. 

For the standard everyday chip (recommended for most tasks, memory retrieval, and simple file operations):
ollama pull qwen3:8b

For the advanced chip (optional, recommended for complex reasoning, multi-step planning, or heavy document analysis):
ollama pull qwen3:14b

You can replace these with any Ollama-compatible model (e.g., llama3.1:8b, mistral, etc.), provided your system has enough RAM and VRAM to run them.

3. Verify Installation
To confirm the models are downloaded and ready, run:
ollama list

4. Configure Ksar
By default, Ksar uses `qwen3:8b` as the standard chip and leaves the advanced chip disabled. You can change this or enable the advanced chip using the built-in console commands once Ksar is running:

/config set chips.standard qwen3:8b
/config set chips.advanced qwen3:14b
/config set chips.auto_escalation true

Ksar will automatically manage loading and unloading these models based on your system resources and performance preferences. You do not need to manually keep them running; the Chip Manager handles this dynamically.

## Run
1. Install Ollama. Pull your models: `ollama pull qwen3:8b` (and optionally an advanced model).
2. In this folder: `python -m ksar` (Windows: double-click `run.bat`).
3. First run asks for names. Everything Ksar owns lives in `ksar_data/`.

##  Commands
```text
chat      just type; Ksar uses tools by himself, risky actions ask you [y/N]
memory    /tree  /recall <q>  /open <note>  /history <note>  /restore <note> <version>
          /forget-note <note>  /purge <note>  /doctor [fix]  /reflect
tasks     /task <objective>  /tasks  /resume <id>  /cancel <id>  /report <id>
safety    /perms  /allow read|write <folder>  /revoke read|write <folder>  /audit [n]

# NEW: Chip & Resource Management
config    /config show | get <key> | set <key> <value>
chips     /chip [status] | active | load <chip> | unload <chip>
          /chippref [speed|balanced|quality]

## Talking to Ksar (0.3)
- Just chat: answers stream live straight from the model, no tools or memory in the way.
- Memory is read only when your message asks for it ("what do you know about...", "remember...", "yesterday...").
- Tools work for their name: write `fs.read_text`, `@fs` or `@memory` in a message, or ask about files/folders.
  Or run one yourself without the model: `/tools`, `/fs.list`, `/fs.read_text notes.txt`.
- `/think on|off` for reasoning (thoughts show live in grey). `/save` writes memory now.
