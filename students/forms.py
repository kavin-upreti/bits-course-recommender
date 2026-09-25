"""Register, profile and past-electives forms. Every input with a fixed set of answers is a dropdown."""
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from catalog.models import Minor

from .bits_id import BitsIdError, ParsedId, parse_bits_id, resolve_programme
from .models import COMFORT, GRADES, Student

YES_NO = [("", "Not decided"), ("True", "Yes"), ("False", "No")]
MINOR_FROM_YEAR = 3  # minor_rules: declared at the end of the 2nd year


class RegisterForm(UserCreationForm):
    """Name + BITS ID + password. The ID must resolve to a Bulletin programme, or registration is refused."""

    name = forms.CharField(max_length=150)
    username = forms.CharField(label="BITS ID", max_length=20, help_text="e.g. 2025A7PS0832P or 2025B3A7PS0832P")

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("name", "username")

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("label_suffix", "")
        super().__init__(*args, **kwargs)

    def clean_username(self) -> str:
        bits_id = self.cleaned_data["username"].strip().upper()
        try:
            resolve_programme(parse_bits_id(bits_id))
        except BitsIdError as error:
            raise forms.ValidationError(str(error)) from error
        if User.objects.filter(username=bits_id).exists():
            raise forms.ValidationError("This ID is already registered; log in instead.")
        return bits_id

    def save(self, commit: bool = True) -> User:
        user = super().save(commit=False)
        user.first_name = self.cleaned_data["name"].strip()
        if commit:
            user.save()
        return user


def semester_choices(max_year: int) -> list[tuple[str, str]]:
    """Every year-sem of the programme. The pick only decides what's done: all semesters before it."""
    return [(f"{year}-{sem}", f"{year}-{sem}") for year in range(1, max_year + 1) for sem in (1, 2)]


class ProfileForm(forms.ModelForm):
    """Everything about the student except courses. Programme, campus and batch are not asked: they come from the ID."""

    planning = forms.ChoiceField(label="Semester you're going into")
    minor = forms.ModelChoiceField(Minor.objects.order_by("name"), required=False, empty_label="No minor")
    interests = forms.CharField(required=False, help_text="Comma-separated, e.g. machine learning, finance")
    sop_plan = forms.TypedChoiceField(
        label="Planning an SOP (study-oriented project under a professor)?", choices=YES_NO, required=False,
        coerce=lambda value: value == "True", empty_value=None,
    )

    class Meta:
        model = Student
        fields = ("minor", "interests", "goal", "cgpa", "strengths", "weaknesses", "sop_plan")
        labels = {"cgpa": "CGPA (optional)"}
        widgets = {"strengths": forms.Textarea(attrs={"rows": 2}), "weaknesses": forms.Textarea(attrs={"rows": 2})}

    def __init__(self, *args, max_year: int, **kwargs) -> None:
        kwargs.setdefault("label_suffix", "")
        super().__init__(*args, **kwargs)
        self.fields["planning"].choices = semester_choices(max_year)
        if self.instance.pk:
            self.initial["planning"] = f"{self.instance.current_year}-{self.instance.current_semester}"
            self.initial["interests"] = ", ".join(self.instance.interests)
            self.initial["sop_plan"] = "" if self.instance.sop_plan is None else str(self.instance.sop_plan)

    def clean_interests(self) -> list[str]:
        return [item.strip() for item in self.cleaned_data["interests"].split(",") if item.strip()]

    def clean_cgpa(self):
        cgpa = self.cleaned_data["cgpa"]
        if cgpa is not None and not 0 <= cgpa <= 10:
            raise forms.ValidationError("CGPA is out of 10.")
        return cgpa

    def clean(self) -> dict:
        cleaned = super().clean()
        if cleaned.get("planning") and cleaned.get("minor") and int(cleaned["planning"].split("-")[0]) < MINOR_FROM_YEAR:
            self.add_error("minor", "A minor is declared at the end of the 2nd year, so it applies from 3-1 onwards.")
        return cleaned

    def save_for(self, user: User, parsed: ParsedId) -> Student:
        """Fill the ID-derived fields, then save. Called instead of save() so a new Student gets them too."""
        student = super().save(commit=False)
        student.user = user
        student.programme = resolve_programme(parsed)
        student.campus = parsed.campus
        student.admission_year = parsed.admission_year
        student.current_year, student.current_semester = map(int, self.cleaned_data["planning"].split("-"))
        student.minor_registered = student.minor is not None
        student.save()
        return student


class ElectiveForm(forms.Form):
    """One elective taken in a past semester. The category only narrows the course list; the category that
    counts is worked out from the programme (recommender/categories.py), never stored from here."""

    category = forms.ChoiceField(choices=[("", "—"), ("DEL", "DEL"), ("HUEL", "HUEL"), ("OPEL", "OPEL")], required=False)
    course = forms.CharField(required=False, widget=forms.TextInput(attrs={"placeholder": "Start typing a code"}))
    grade = forms.ChoiceField(choices=[("", "—")] + GRADES, required=False)
    comfort = forms.ChoiceField(choices=[("", "—")] + COMFORT, required=False)

    def __init__(self, *args, codes_by_category: dict[str, set[str]], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.codes_by_category = codes_by_category

    def clean(self) -> dict:
        cleaned = super().clean()
        code = " ".join(cleaned.get("course", "").upper().split())  # "cs  f211" -> "CS F211"
        cleaned["course"] = code
        if not code:
            return cleaned  # empty row: ignored
        category = cleaned.get("category")
        if not category:
            self.add_error("category", "Pick DEL, HUEL or OPEL.")
        elif code not in self.codes_by_category[category]:
            actual = next((cat for cat, codes in self.codes_by_category.items() if code in codes), None)
            hint = f" For your programme it's a {actual}." if actual else " It's compulsory, an audit course, or not in the Bulletin."
            self.add_error("course", f"{code} isn't a {category} option.{hint}")
        if not cleaned.get("comfort"):
            self.add_error("comfort", "Say how comfortable you were with it.")
        return cleaned


ElectiveFormSet = forms.formset_factory(ElectiveForm, extra=3)
