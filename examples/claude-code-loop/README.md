# ACE + Claude Code Learning Loop

Run Claude Code in a loop with automatic learning. Each session improves the next via ACE's skillbook.

## Quick Start

```bash
# 1. Setup
cp .env.example .env
# Add your API key to .env

# 2. Initialize workspace
./reset_workspace.sh

# 3. Run (interactive mode)
python ace_loop.py

# Or fully automatic
AUTO_MODE=true python ace_loop.py
```

## Customize

Edit `prompt.md` to change what Claude Code does in each session.

## How It Works

```
┌─────────────────────────────────────────────┐
│  Session 1: Claude Code executes task       │
│      ↓                                      │
│  ACE learns from execution (Reflector)      │
│      ↓                                      │
│  Skillbook updated (SkillManager)           │
│      ↓                                      │
│  Session 2: Claude Code + learned skills    │
│      ↓                                      │
│  ... repeat until stalled or done ...       │
└─────────────────────────────────────────────┘
```

**Stall detection**: Stops after 4 consecutive sessions with no code commits.

## Files

- `prompt.md` - Task prompt (edit this!)
- `ace_loop.py` - Main script
- `workspace/` - Where Claude Code works
- `.data/skillbooks/` - Learned strategies (persists across runs)

## Environment Variables

- `AUTO_MODE=true` - Skip confirmations
- `ACE_MODEL=...` - Model for learning (default: claude-sonnet-4-5)
