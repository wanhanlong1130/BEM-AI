
import asyncio

# ---------------------------------------------------------------------
# Natural slow-streaming effect
# ---------------------------------------------------------------------
async def natural_delay(text_fragment: str):
    """Add a natural delay based on fragment length."""
    # You can tune these values
    base_delay = 0.02  # 20 ms
    per_char_delay = 0.002  # +2 ms per character

    delay = base_delay + len(text_fragment) * per_char_delay
    await asyncio.sleep(delay)