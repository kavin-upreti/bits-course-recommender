"""`python manage.py llm_smoke_test`: one real call ("Say hello") per configured provider, to check keys and model
names. Not a unit test."""
from unittest.mock import patch

from django.core.management.base import BaseCommand, CommandError

from recommender import llm


class Command(BaseCommand):
    help = "Send 'Say hello' to each provider in LLM_PROVIDERS and print the reply and token usage."

    def handle(self, *args, **options) -> None:
        failed = False
        for name, model in llm.providers():
            # why patch: chat() falls back to the next provider; here each one is tried on its own
            with patch.object(llm, "providers", lambda: [(name, model)]), patch.object(llm.config, "LLM_MAX_RETRIES_ON_RATE_LIMIT", 0):
                try:
                    response = llm.chat("You are a friendly assistant.", [{"role": "user", "text": "Say hello"}], None)
                    self.stdout.write(f"{name}:{model}  OK   {response.text!r}  {response.usage}")
                except (llm.LLMError, llm.LLMUnavailable) as error:
                    failed = True
                    self.stdout.write(f"{name}:{model}  FAIL {type(error).__name__}: {str(error)[:200]}")
        if failed:
            raise CommandError("At least one provider failed (the others still work as fallbacks).")
