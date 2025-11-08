# Domain Checker Examples

Browser automation that checks domain availability and learns to get faster and more accurate over time.

## Files

- **ace_domain_checker.py** - WITH ACE learning (improves after each domain)
- **baseline_domain_checker.py** - WITHOUT ACE (static performance)
- **domain_utils.py** - Domain-specific utilities (test domains, parsing)

## Quick Run

```bash
# WITH ACE (learns and improves)
uv run python examples/browser-use/domain-checker/ace_domain_checker.py

# WITHOUT ACE (baseline for comparison)
uv run python examples/browser-use/domain-checker/baseline_domain_checker.py
```

## How It Works

### Baseline (No Learning)
1. Navigate to domain checker website
2. Search for domain
3. Read availability status
4. Repeat for each domain (same strategy every time)

### ACE (With Learning)
1. **Generator** creates strategy for checking domains
2. **Browser-use agent** executes the strategy
3. **Reflector** analyzes what worked/failed
4. **Curator** updates playbook with lessons
5. **Next domain** uses improved strategy!

## Expected Results

**First domain:** May take 10-15 steps, tries different approaches

**After learning:** Typically 5-7 steps, uses best strategy

**Learned strategies include:**
- Which domain checker websites work best
- How to wait for results to load
- Where to look for availability status
- Error handling patterns

## Configuration

Edit the main() function to customize:

```python
# Change LLM model
llm = LiteLLMClient(model="claude-sonnet-4-5-20250929")  # or gpt-4o, etc.

# Change browser model
environment = DomainCheckEnvironment(
    headless=False,  # Set True to hide browser
    model="claude-sonnet-4-5-20250929"
)

# Test different domains
domains = get_test_domains()  # Or define your own list
```

## Output Explained

```
#   Domain                    Status      Acc  Steps    Browser-Tokens    Details
--------------------------------------------------------------------------------
1   testdomain123456.com      AVAILABLE   ✓    12       8,234            (1 attempt)
2   myuniquedomain789.net     AVAILABLE   ✓    8        6,721            (1 attempt)
```

- **Status**: AVAILABLE, TAKEN, or ERROR
- **Acc**: ✓ if correctly identified (test domains are AVAILABLE)
- **Steps**: Browser actions taken
- **Browser-Tokens**: LLM tokens used by browser-use agent
- **Details**: Number of retry attempts

## Utilities

### domain_utils.py

```python
from domain_utils import (
    get_test_domains,              # List of test domains
    parse_domain_checker_output,   # Parse "AVAILABLE: domain.com" format
    DOMAIN_CHECKER_TEMPLATE        # Prompt template for browser agent
)
```

## Customization Ideas

1. **Different domains**: Edit `get_test_domains()` in domain_utils.py
2. **Different checker sites**: Let ACE discover which ones work best
3. **Bulk checking**: Increase number of domains to see more learning
4. **Cost optimization**: Use cheaper models (gpt-4o-mini) and compare results

## Troubleshooting

**"No module named 'domain_utils'"**
- Make sure you're running from the correct directory
- The script uses `sys.path.insert()` to find parent imports

**Browser doesn't start**
```bash
playwright install chromium
```

**Domains always show ERROR**
- Some domain checkers block automated access
- ACE will learn to switch to working sites
- Try running with `headless=False` to see what's happening

## Next Steps

1. Run both baseline and ACE versions
2. Compare the playbooks generated
3. Try adding your own domains
4. Customize the evaluation criteria in DomainCheckEnvironment
5. Adapt this pattern for your own browser automation tasks!
