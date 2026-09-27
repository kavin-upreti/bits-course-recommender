"""{% versioned_static "students/planner.css" %}: the static URL plus the file's modified time, so browsers refetch a
changed file instead of reusing a cached copy (which renders new markup unstyled)."""
from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static
from pathlib import Path

register = template.Library()


@register.simple_tag
def versioned_static(path: str) -> str:
    found = finders.find(path)
    # ponytail: stat on every render; switch to ManifestStaticFilesStorage when this is deployed with collectstatic
    version = int(Path(found).stat().st_mtime) if found else 0
    return f"{static(path)}?v={version}"
