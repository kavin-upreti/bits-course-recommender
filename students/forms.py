"""Register, profile and past-electives forms. Every input with a fixed set of answers is a dropdown."""
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from catalog.models import Minor

from dataclasses import replace

from .bits_id import BitsIdError, ParsedId, parse_bits_id, resolve_programme
from recommender.config import DAY_CODES, DAY_NAMES

from .models import EVAL_STYLES, MAX_PICKED_COURSES, Student

YES_NO = [("", "Not decided"), ("True", "Yes"), ("False", "No")]
MINOR_AIM_FROM_YEAR = 2  # 2nd-years can name the minor they're aiming for
MINOR_FROM_YEAR = 3  # minor_rules: declared at the end of the 2nd year, so pursued from 3-1
SECOND_DEGREE_ASKED_FROM = (2, 1)  # dual degrees are allotted after the 1st year
SECOND_DEGREE_NEEDED_FROM = (2, 2)  # the planner needs the B.E. chart from here on (user's rule, 2026-09-26)


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


class ProfileForm(forms.ModelForm):
    """Everything about the student except courses. Programme, campus, batch and the semester being planned are not
    asked: they come from the ID (the semester from its batch and the loaded timetable, recommender.timetable)."""

    second_degree_code = forms.ChoiceField(label="Your B.E. (second) degree", required=False)
    minor = forms.ModelChoiceField(Minor.objects.order_by("name"), required=False, empty_label="No minor")
    interests = forms.CharField(required=False, help_text="Comma-separated, e.g. machine learning, finance")
    sop_plan = forms.TypedChoiceField(
        label="Planning an SOP (study-oriented project under a professor)?", choices=YES_NO, required=False,
        coerce=lambda value: value == "True", empty_value=None,
    )
    default_avoid_8am = forms.BooleanField(label="Avoid 8 AM classes by default", required=False)
    default_avoid_day = forms.TypedChoiceField(
        label="Keep this day free by default", required=False, empty_value=None,
        choices=[("", "No preference")] + [(code, DAY_NAMES[code]) for code in DAY_CODES],
    )
    avoid_eval_styles = forms.MultipleChoiceField(
        label="Evaluation styles you'd rather avoid", choices=EVAL_STYLES, required=False,
        widget=forms.CheckboxSelectMultiple, help_text="Courses with these are ranked a little lower, not removed.",
    )
    grade_oriented = forms.BooleanField(label="Grades matter a lot to me", required=False)
    did_well = forms.MultipleChoiceField(label="Courses you did well in", required=False,
                                         widget=forms.CheckboxSelectMultiple, help_text=f"Up to {MAX_PICKED_COURSES}.")
    struggled = forms.MultipleChoiceField(label="Courses you struggled with", required=False,
                                          widget=forms.CheckboxSelectMultiple, help_text=f"Up to {MAX_PICKED_COURSES}.")

    class Meta:
        model = Student
        fields = ("second_degree_code", "minor", "interests", "strengths", "sop_plan", "did_well", "struggled",
                  "grade_oriented", "default_avoid_8am", "default_avoid_day", "avoid_eval_styles")
        help_texts = {"strengths": "Subjects you're good at, comma-separated and in full words, e.g. programming, "
                                   "algorithms, statistics. Courses on them rank a little higher."}
        widgets = {"strengths": forms.Textarea(attrs={"rows": 2})}

    def __init__(self, *args, planning: tuple[int, int], second_degrees: list[tuple[str, str]],
                 completed: list[tuple[str, str]] | None = None, **kwargs) -> None:
        """planning: the semester being planned. second_degrees: the B.E. options for an M.Sc.-only ID (empty = not asked).
        completed: (code, label) of the student's completed courses, for the did well / struggled pickers (none = not asked).
        Questions that don't apply to that semester are removed, not hidden."""
        kwargs.setdefault("label_suffix", "")
        super().__init__(*args, **kwargs)
        self.planning = planning
        if second_degrees and planning >= SECOND_DEGREE_ASKED_FROM:
            field = self.fields["second_degree_code"]
            field.choices = [("", "Not decided yet")] + second_degrees
            if planning >= SECOND_DEGREE_NEEDED_FROM:
                field.help_text = "Needed from 2-2: your plan includes its courses from here on."
            else:
                field.label = "B.E. degree you expect to get (optional)"
                field.help_text = "Not needed for 2-1, but picking the one you expect keeps its courses out of your recommendations."
        else:
            del self.fields["second_degree_code"]
        if planning[0] < MINOR_AIM_FROM_YEAR:
            del self.fields["minor"]
        else:
            self.fields["minor"].label = "Minor you're pursuing" if planning[0] >= MINOR_FROM_YEAR else "Minor you're aiming for"
            self.fields["minor"].help_text = "Minors are declared at the end of 2nd year; before that, pick the one you're aiming for."
        if completed:
            self.fields["did_well"].choices = self.fields["struggled"].choices = completed
        else:  # why: a new profile has no courses until it's saved (they come from the programme chart)
            del self.fields["did_well"], self.fields["struggled"], self.fields["grade_oriented"]
        if self.instance.pk:
            self.initial["interests"] = ", ".join(self.instance.interests)
            self.initial["sop_plan"] = "" if self.instance.sop_plan is None else str(self.instance.sop_plan)

    def clean_interests(self) -> list[str]:
        return [item.strip() for item in self.cleaned_data["interests"].split(",") if item.strip()]

    def clean_did_well(self) -> list[str]:
        return self.picked("did_well")

    def clean_struggled(self) -> list[str]:
        return self.picked("struggled")

    def picked(self, name: str) -> list[str]:
        codes = self.cleaned_data[name]
        if len(codes) > MAX_PICKED_COURSES:
            raise forms.ValidationError(f"Pick at most {MAX_PICKED_COURSES}.")
        return codes

    def clean(self) -> dict:
        cleaned = super().clean()
        if both := sorted(set(cleaned.get("did_well") or []) & set(cleaned.get("struggled") or [])):
            self.add_error("struggled", f"Picked as both did well and struggled: {', '.join(both)}.")
        if "second_degree_code" in self.fields:
            if self.planning >= SECOND_DEGREE_NEEDED_FROM and not cleaned.get("second_degree_code"):
                self.add_error("second_degree_code", "From 2-2 on your plan includes your B.E. courses, so pick your B.E. degree.")
        return cleaned

    def save_for(self, user: User, parsed: ParsedId) -> Student:
        """Fill the ID-derived fields, then save. Called instead of save() so a new Student gets them too."""
        student = super().save(commit=False)
        student.user = user
        second = self.cleaned_data.get("second_degree_code") or ""
        student.second_degree_code = second
        student.programme = resolve_programme(replace(parsed, second_code=second) if second else parsed)
        student.campus = parsed.campus
        student.admission_year = parsed.admission_year
        student.current_year, student.current_semester = self.planning
        if "minor" not in self.fields:  # 1st year: no minor yet
            student.minor = None
        # 2nd-years only aim for a minor; it's registered (pursued) from 3-1 on
        student.minor_registered = student.minor is not None and student.current_year >= MINOR_FROM_YEAR
        student.save()
        return student


class ElectiveForm(forms.Form):
    """One elective taken in a past semester. The category only narrows the course list; the category that
    counts is worked out from the programme (recommender/categories.py), never stored from here."""

    category = forms.ChoiceField(choices=[("", "—"), ("DEL", "DEL"), ("HUEL", "HUEL"), ("OPEL", "OPEL")], required=False)
    course = forms.CharField(required=False, widget=forms.TextInput(attrs={"placeholder": "Start typing a code"}))

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
        return cleaned


ElectiveFormSet = forms.formset_factory(ElectiveForm, extra=3)
