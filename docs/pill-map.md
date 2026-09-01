# BUS 390A "Getting Started with Python" — Module/Topic Map for Virtual TA Pills (v2)

Sources: Canvas course 167896 (all 9 modules, 33 pages, read 2026-08-31) + full New Quizzes extract supplied by Wen 2026-08-31 (56 module-quiz items M1–M6, 50 final-quiz items, 2 pulse-check surveys; 98 MC + 8 matching; matching answer-pairs not captured, stems only).

Structure and topic lists [Certain]. Depth notes on video-only lessons remain [Likely] — simulation content still unread.

---

## Design principle for the pills

Quiz coverage is deliberately narrower than instruction: quizzes test syntax and vocabulary; the applications (bill_per_person, dog_years, Madlibs, QR-with-logo, the whole M7 toolchain) are assessed only through projects. So each module pill needs two layers — **teach at instruction scope, practice at quiz scope** — or students will pass quizzes and stall on projects, or vice versa.

## Module-by-module: taught vs. quizzed

### M0: Welcome — survey only
Orientation, Peyton intro (Streamlit: `bus390-virtualta-beta.streamlit.app`), Semester Start pulse check (syllabus/honor-code attestation, background). Pill value: logistics only.

### Module 1: Introduction — quiz 7 MC, due Aug 31
- Taught: high-level language, why Python / Python vs. Excel, Colab intro, `print('Hello World')`, **Rule of Harmony** (matched parens/quotes → SyntaxError)
- Quizzed: `print()` usage, valid syntax, parens inside strings, unclosed quotes, `print(42)` vs `print("42")`, meaning of "high-level"
- Taught-not-quizzed: Colab workflow; "Rule of Harmony" as a named idea (its substance IS quizzed — the pill should connect the name to the questions)

### Module 2: Python as Calculator — quiz 1 matching + 9 MC, due Sep 7
- Taught: arithmetic + PEMDAS, `#` comments, variables, bill_per_person application
- Quizzed: operator symbols (matching), `*`, comments (symbol, purpose, ignored at runtime), assignment, reassignment `x = x + 2`, informative names (`revenue_quarter`), `print(2 + )` SyntaxError
- Taught-not-quizzed: bill_per_person (project territory); PEMDAS itself is thin on the quiz

### Module 3: Data Types — quiz 6 MC + 2 matching, due Sep 14
- Taught: `input()`, Madlibs, dog-age (`age * 7 + 1`), `str`/`int`/`float`, casting, TypeError fix
- Quizzed: `input()` purpose and str return, `int()`/`float()`/`type()`, values↔types matching, `"Hello" + 5` TypeError, casting input before arithmetic
- Taught-not-quizzed: Madlibs and dog-age as stories (Q8 is the dog-age lesson stripped of the story)

### Module 4: Functions — quiz 10 MC, due Sep 21
- Taught: Reeborg, `def`/call, parameters, **`return`**, add/subtract, bill_split v1→v2, dog_years; gotchas: indentation, name consistency
- Quizzed: `def` keyword and header, indentation (2 items), parameter in header, call with argument, "parameter" vocab, missing-argument behavior, local vs global scope (2 items)
- Taught-not-quizzed: **`return` — the largest conceptual gap on any quiz.** Also Reeborg, bill_split, dog_years. But M5's entire quiz presumes functions that return values, and the projects use them — the M4 pill must carry `return` at full weight despite zero quiz items.

### Module 5: Create Your Own Module — quiz 10 MC, due Sep 21 (same night as M4)
- Taught: module vs function, `simple_calculator.py`, 4-step Colab session-storage workflow, dog_years module
- Quizzed (all 10): what a module is, `.py`, `import`, alias (`import x as sc`), dotted calls (`simple_calculator.add()`, `sc.divide()`, `mo.square(5)`)
- Taught-not-quizzed: the Colab session-storage steps — yet they're where students actually get stuck (files vanish between sessions). Pill must cover them.
- Scheduling note: M4+M5 due the same night → expect a combined help spike the week of Sep 21; pills 4, 5, and the error pill should cross-link.

### Module 6: Use Other's Modules — quiz 10 MC + 1 matching, due Sep 28
- Taught: libraries, `%pip install` + import, basic `qrcode.make()`, advanced `QRCode(version, box_size, border)` + colors, logo embed via PIL (full snippets on the "6 Module Summary" page — lift verbatim)
- Quizzed: library concept (2), `pip install`, three import styles (whole / alias / `from qrcode import make`), QR for a URL, advanced-code matching, business uses, `version=1` size, `fill_color`/`back_color`
- Taught-not-quizzed: logo embedding (project/differentiation content)

### Module 7: Python on Your Computer — NO quiz; project due Oct 5
Install Python (PATH!), VS Code, Python + Jupyter extensions, ipykernel; recreate simple_calculator and my_module locally. Assessed only by the Use Python Locally project → the M7 pill is pure how-to/troubleshooting (PATH, extension install, ipykernel prompt, "0 KB file = not saved").

### Final Course Assessment — final quiz 50 pts due Nov 2, final project
Final quiz = shuffle of M1–M6, no new topics, no M7. M5 copied in full; M6 most thinned (drops `pip install`, whole-library import, `version=1`); M4 Q8 key corrected to TypeError on the final. Final Project Hints FAQ: submit `.ipynb`; `%pip` in cells vs `pip` in terminal; 0 KB = unsaved.

---

## Recommended pill set (12) — revised emphasis

1–7. One pill per module, two-layer (teach instruction scope / practice quiz scope). Specific weight changes from the quiz data:
   - **M4 pill**: `return` stays central despite zero quiz items — it's the load-bearing concept for M5 and both projects.
   - **M5 pill**: lead with the session-storage workflow (unquizzed, high-friction), then the import/alias/dotted-call material the quiz drills.
   - **M7 pill**: troubleshooting only; no quiz to align to.
2. **Error doctor** — now confirmed as quiz-aligned, not just project-aligned: SyntaxError (M1 quotes/parens, M2 `print(2 + )`), TypeError (M3 `"Hello" + 5`, casting input; M4 missing argument — see key conflict below), IndentationError (M4, 2 items), NameError/scope (M4, 2 items).
3. **The three running examples** (bill_per_person, dog_years, simple_calculator) — reframed: zero quiz items reference them; this pill exists for **project** support and for showing how concepts chain M2→M4→M5→M7.
4. **Colab & session storage** — unquizzed but the top predicted friction point for M5's project.
5. **%pip vs pip** — quizzed in M6 (as `pip install`) and a stated final-project FAQ.
6. **Final prep** — can now say precisely: "the final re-asks your module quizzes; redo M1–M6 quizzes, M5 verbatim; nothing from M7."

## Course defects the pills must route around (fix in Canvas)

1. **M4 Q8 answer-key conflict** [Certain]: same stem ("call a parameterized function with no argument") keyed "uses a default value" on the module quiz but "TypeError" (correct) on the final. Until the module key is fixed, Peyton will either contradict the module quiz or teach something false. Fix the module key; the pill teaches TypeError.
2. M3 Q4 is named "Syntax Error" but tests a TypeError — rename to avoid reinforcing the exact confusion M3 tries to resolve.
3. M4 indent item keys "Tab" while Python convention (and Colab's default) is 4 spaces — key accepts tab; consider rewording so the pill doesn't have to hedge.
4. End-of-semester survey's "hardest module" options use older labels (M1 Hello World, M2 Sequential Instructions, …) — those labels are good short pill titles, but update the survey or the Fall responses won't map cleanly to current module names.

## Remaining gaps

- [Certain] 8 matching questions have stems only — answer pairs not captured. Minor: pairs are inferable from lesson content.
- [Likely] Simulation-video content still unread; M1–M4 and M7 lack summary pages (M5/M6 have them). Writing those summaries remains the highest-leverage content task — they double as pill source text.
- [Guessing] Course homepage/syllabus page still unread; needed only if pills answer logistics questions.
