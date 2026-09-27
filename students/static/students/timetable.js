// Timetable grids (students/_timetables.html): pager, switching between same-time sections, and the filter form.
// Delegated from `document`, so it also works for grids loaded into a popup later.
document.addEventListener("click", (event) => {
  const step = event.target.closest(".pager button[data-step]");
  if (step) {
    const wrap = step.closest(".tt-wrap");
    const tables = [...wrap.querySelectorAll(".tt")];
    const shown = tables.findIndex((table) => !table.hidden);
    const next = (shown + Number(step.dataset.step) + tables.length) % tables.length;
    tables[shown].hidden = true;
    tables[next].hidden = false;
    wrap.querySelector(".tt-index").textContent = next + 1;
    return;
  }
  // another section at exactly the same time: show it in the list and in the grid (display only, nothing saved)
  const choice = event.target.closest(".alt-choice");
  if (choice) {
    const row = choice.closest("tr[data-pick]"), tt = choice.closest(".tt");
    row.querySelector(".sec-id").textContent = choice.dataset.id;
    row.querySelector(".sec-profs").textContent = choice.dataset.profs;
    row.querySelectorAll(".alt-choice").forEach((button) => button.classList.toggle("chosen", button === choice));
    tt.querySelectorAll(`.slot[data-pick="${CSS.escape(row.dataset.pick)}"]`).forEach((slot) => {
      slot.querySelector(".sid").textContent = choice.dataset.id;
      slot.querySelector(".room").textContent = choice.dataset.room;
      slot.title = [slot.dataset.pick.replace("|", " ") + " " + choice.dataset.id, choice.dataset.room && "room " + choice.dataset.room, choice.dataset.profs].filter(Boolean).join(", ");
    });
    choice.closest("details").open = false;
  }
});

// In the popup the filter form reloads just the grids instead of the page.
document.addEventListener("submit", async (event) => {
  const form = event.target.closest(".tt-filters[data-fetch]");
  if (!form) return;
  event.preventDefault();
  const response = await fetch(form.action + "?" + new URLSearchParams(new FormData(form)));
  form.closest(".tt-wrap").outerHTML = await response.text();
});

// Teacher filter: pick a course, then one of its teachers.
document.addEventListener("change", (event) => {
  const course = event.target.closest("[data-teacher-course]");
  if (!course) return;
  const names = course.selectedOptions[0].dataset.names;
  const teacher = course.parentElement.querySelector("[data-teacher-name]");
  teacher.replaceChildren(new Option("Pick a teacher", ""), ...(names ? names.split("||") : []).map((name) => new Option(name, course.value + "|" + name)));
  teacher.hidden = teacher.disabled = !names;
});
