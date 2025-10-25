#!/usr/bin/env python3
"""
Baseline Form Filler (WITHOUT ACE)

Simple form filler using browser automation without any learning.
Compare this with ace_form_filler.py to see ACE's value.
"""

import asyncio
from typing import List, Dict
from dotenv import load_dotenv

from browser_use import Agent, Browser, ChatOpenAI

load_dotenv()


def get_test_forms() -> List[Dict]:
    """Get list of test forms to fill."""
    return [
        {
            "name": "Contact Form",
            "data": {
                "name": "John Doe",
                "email": "john@example.com",
                "message": "Hello, this is a test message."
            }
        },
        {
            "name": "Newsletter Signup",
            "data": {
                "email": "jane@example.com",
                "name": "Jane Smith"
            }
        },
        {
            "name": "User Registration",
            "data": {
                "username": "testuser123",
                "email": "test@example.com",
                "password": "SecurePass123"
            }
        }
    ]


async def fill_form(form_data: Dict, model: str = "gpt-4o-mini", headless: bool = True):
    """Fill form without any learning."""
    browser = None
    try:
        # Start browser
        browser = Browser(headless=headless)
        await browser.start()

        # Create agent with basic task (no learning, no strategy optimization)
        llm = ChatOpenAI(model=model, temperature=0.0)

        # Format form data for task
        form_text = ""
        for field, value in form_data.items():
            form_text += f"- {field}: {value}\n"

        task = f"""Fill out a web form with this data:
{form_text}

Navigate to a form (like a contact form, signup form, etc.) and fill it out accurately.
You can use Google Forms, demo forms, or create a simple HTML form.

Output when done:
SUCCESS: Form filled successfully
ERROR: <reason>"""

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            max_actions_per_step=5,
            max_steps=15,
        )

        # Run with timeout
        history = await asyncio.wait_for(agent.run(), timeout=120.0)

        # Parse result
        output = history.final_result() if hasattr(history, "final_result") else ""
        steps = len(history.action_names()) if hasattr(history, "action_names") and history.action_names() else 0

        # Determine status
        status = "ERROR"
        if "SUCCESS:" in output.upper():
            status = "SUCCESS"

        return {
            "status": status,
            "steps": steps,
            "output": output,
            "success": status == "SUCCESS"
        }

    except asyncio.TimeoutError:
        return {"status": "ERROR", "steps": 999, "error": "Timeout", "success": False}
    except Exception as e:
        return {"status": "ERROR", "steps": 999, "error": str(e), "success": False}
    finally:
        if browser:
            try:
                await browser.stop()
            except:
                pass


def main():
    """Main function - basic form filling without learning."""

    print("\n🤖 Baseline Form Filler (WITHOUT ACE)")
    print("🚫 No learning - same approach every time")
    print("=" * 40)

    # Get test forms
    forms = get_test_forms()
    print(f"📋 Testing {len(forms)} forms:")
    for i, form in enumerate(forms, 1):
        print(f"  {i}. {form['name']}")

    print("\n🔄 Starting form filling (no learning)...\n")

    results = []

    # Fill each form without any learning
    for i, form in enumerate(forms, 1):
        print(f"📝 [{i}/{len(forms)}] Filling: {form['name']}")

        # Run form fill
        result = asyncio.run(fill_form(form['data'], headless=False))
        results.append(result)

        # Show what happened
        status = result['status']
        steps = result['steps']
        success = result['success']

        print(f"   📊 Result: {status} ({'✓' if success else '✗'}) in {steps} steps")
        print()

    # Show final results
    print("=" * 40)
    print("📊 Results:")

    for i, (form, result) in enumerate(zip(forms, results), 1):
        status = result['status']
        steps = result['steps']
        success = result['success']
        print(f"[{i}] {form['name']}: {status} ({'✓' if success else '✗'}) - {steps} steps")

    # Summary
    successful = sum(1 for r in results if r['success'])
    total_steps = sum(r['steps'] for r in results)
    avg_steps = total_steps / len(results) if results else 0

    print(f"\n✅ Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"⚡ Average steps: {avg_steps:.1f}")
    print(f"🚫 No learning - same performance every time")

    print(f"\n💡 Compare with: python examples/browser-use/ace_form_filler.py")
    print(f"   ACE learns and improves after each form!")


if __name__ == "__main__":
    main()