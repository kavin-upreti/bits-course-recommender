from django.apps import apps
from django.contrib import admin

# ponytail: default ModelAdmin for everything, enough to browse by hand; add list_display/search where it gets tedious
admin.site.register(apps.get_app_config("students").get_models())
