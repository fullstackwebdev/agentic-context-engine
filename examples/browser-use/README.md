# Browser-Use + ACE Integration Examples

This folder demonstrates how to integrate the **ACE (Agentic Context Engineering)** framework with **[browser-use](https://github.com/browser-use/browser-use)** for self-improving browser automation agents.

## 🎯 What is This?

ACE enables browser automation agents to **learn from their execution feedback** and improve over time. Instead of static prompts, ACE agents:

1. **Generate** strategies for browser tasks
2. **Execute** them using browser-use
3. **Reflect** on what worked/failed
4. **Curate** lessons into a persistent playbook
5. **Improve** on subsequent tasks

## 📁 Folder Structure

```
examples/browser-use/
├── README.md              # Getting started guide (you are here!)
├── TEMPLATE.py            # Clean template for your own use cases
├── common.py              # Generic utilities (domain-agnostic)
├── utils.py               # Debug/inspection utilities
├── domain-checker/        # Domain availability examples (see README inside)
└── form-filler/           # Form filling examples (see README inside)
```

Each example folder contains:
- ACE version (WITH learning)
- Baseline version (WITHOUT learning for comparison)
- Local README with detailed documentation
- Example-specific utilities and task files

## 🚀 Quick Start

### 1. Installation

```bash
# Install ACE framework with browser-use demo dependencies
pip install ace-framework[demos]

# Or for development (using UV - recommended)
cd agentic-context-engine
uv sync --extra demos      # Install with demos dependencies
# Or: uv sync --all-extras  # Install all optional dependencies
```

### 2. Set API Key

```bash
# Set your LLM API key (ACE uses LiteLLM, supports 100+ providers)
export OPENAI_API_KEY="your-api-key"
# Or: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
```

### 3. Run an Example

```bash
# Domain checker WITH ACE (learns after each domain)
uv run python examples/browser-use/domain-checker/ace_domain_checker.py

# Domain checker WITHOUT ACE (baseline for comparison)
uv run python examples/browser-use/domain-checker/baseline_domain_checker.py

# Form filler WITH ACE
uv run python examples/browser-use/form-filler/ace_form_filler.py
```

## 📊 Expected Results

**Baseline (no learning):**
- Same performance on every task
- Static strategies
- No improvement over time

**ACE (with learning):**
- Performance improves across tasks
- Learns efficient patterns
- Adapts strategies based on feedback
- Builds reusable playbook

### Example Output

```
🚀 ACE + Browser-Use Domain Checker
🧠 Learns after each domain check!
==================================================

#   Domain                    Status      Acc  Steps    Browser-Tokens
--------------------------------------------------------------------------------
1   testdomain123456.com      AVAILABLE   ✓    12       8,234
2   myuniquedomain789.net     AVAILABLE   ✓    8        6,721    (improved!)
3   brandnewstartup2024.io    AVAILABLE   ✓    6        5,892    (improved!)
...

🎯 Learned Strategies:
  1. Use GoDaddy domain checker for fastest results
  2. Wait for search results to fully load before reading
  3. Look for "available" badge in green color
```

## 🛠️ Create Your Own Use Case

### Option 1: Start from Template

Copy `TEMPLATE.py` and customize for your task:

```python
# 1. Define your evaluation environment
class MyTaskEnvironment(TaskEnvironment):
    def evaluate(self, sample, generator_output):
        # Your task-specific evaluation logic
        pass

# 2. Create ACE components
llm = LiteLLMClient(model="gpt-4o")
adapter = OnlineAdapter(
    playbook=Playbook(),
    generator=Generator(llm),
    reflector=Reflector(llm),
    curator=Curator(llm)
)

# 3. Run and learn!
results = adapter.run(samples, environment)
```

### Option 2: Adapt an Example

Browse `domain-checker/` or `form-filler/` examples and modify them for your needs.

## 📖 Documentation

- **Main ACE Framework:** See `/README.md` and `/docs/` in root
- **Domain Checker Examples:** See `domain-checker/README.md`
- **Form Filler Examples:** See `form-filler/README.md`
- **Browser-Use Library:** https://github.com/browser-use/browser-use

## 🔬 Key Concepts

### ACE Components

1. **Generator**: Plans browser automation strategies
2. **Reflector**: Analyzes execution feedback (errors, successes, efficiency)
3. **Curator**: Updates playbook with learned lessons
4. **Playbook**: Persistent knowledge base (bullets with helpful/harmful scores)

### Adaptation Modes

- **OnlineAdapter**: Learn after each task (used in these examples)
- **OfflineAdapter**: Train on batch of examples first, then deploy

### Environment Integration

Your `TaskEnvironment` bridges ACE with browser-use:
- Receives strategy from Generator
- Executes browser automation
- Returns feedback to Reflector

## 💡 Tips

1. **Start Simple**: Begin with baseline demo, then compare with ACE version
2. **Headless Mode**: Set `headless=True` for faster execution (no GUI)
3. **Debug Mode**: Use `utils.print_history_details()` to inspect browser actions
4. **Cost Tracking**: Enable Opik observability to monitor token usage
5. **Prompt Versions**: Use v2.1 prompts for best performance (see CLAUDE.md)

## 📝 Common Utilities

### `common.py` - Generic Utilities

```python
from common import (
    calculate_timeout_steps,  # Convert timeout to step count
    format_result_output,      # Pretty-print results
    save_results_to_file,      # Save to JSON
    get_browser_config,        # Browser settings
    get_llm_config,            # LLM settings
    MAX_RETRIES,               # Retry constants
    DEFAULT_TIMEOUT_SECONDS
)
```

### `utils.py` - Debug Utilities

```python
from utils import print_history_details

# Print comprehensive browser execution details
history = await agent.run()
print_history_details(history)
# Shows: actions, results, URLs, errors, thoughts, timing, etc.
```

## 🤝 Contributing

Have a cool browser automation use case? Add a new example folder!

1. Create `your-use-case/` folder
2. Add `ace_*.py` and `baseline_*.py` files
3. Create local `README.md` and `*_utils.py`
4. Keep `common.py` generic (no use-case-specific code)

## 🐛 Troubleshooting

**Import errors after restructuring?**
- Files in subfolders use `sys.path.insert()` to import from parent
- Check that `common.py` and `utils.py` are in `browser-use/` root

**Browser not starting?**
- Install playwright: `playwright install chromium`
- Check browser-use installation: `pip install browser-use`

**LLM API errors?**
- Verify API key is set: `echo $OPENAI_API_KEY`
- Check LiteLLM supported models: https://docs.litellm.ai/docs/

## 📚 Next Steps

1. ✅ Run domain-checker example
2. ✅ Compare baseline vs ACE performance
3. ✅ Copy `TEMPLATE.py` and customize
4. ✅ Read `domain-checker/README.md` for detailed walkthrough
5. ✅ Explore advanced features (Opik, v2.1 prompts, checkpoints)

Happy automating! 🤖✨
