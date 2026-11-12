# ISE 5406 (Nonlinear Programming / Optimization II): Spring 2026 materials

Robert Hildebrand's Spring 2026 offering, with the lectures organized into one sequence.

## What is here

| Folder | Contents |
|---|---|
| `Lectures/` | 26 lectures, one folder each: slide deck, plus instructor proof notes and handouts where they exist. |
| `Final Exam/` | Take-home final (PDF), student starter notebook, and datasets with the script that generates them. No solutions. |
| `../Quizzes/` | Four self-contained interactive HTML quizzes. Open in a browser. |
| `../visualizations/` | Source for the interactive demos (React + Vite). `npm install && npm run dev` runs them locally; `../.github/workflows/static.yml` deploys them to GitHub Pages. |

These are compiled PDFs only. LaTeX sources, homework and solutions, the syllabus, the course
notes and the Jupyter notebook suite are not in this repository.

## Lecture sequence

Files are named `LNN-Slides.pdf`, `LNN-Proofs-Notes.pdf`, `LNN-Handout-*.pdf`.

| # | Topic | Spring 2026 label | Main files | Also in folder |
|---|---|---|---|---|
| 01 | Introduction | Lecture 1 | `L01-Slides.pdf` (54 pp) |  |
| 02 | Linear Algebra, Convex Sets and Convex Functions | Lecture 2 | `L02-Slides.pdf` (50 pp)<br>`L02-Proofs-Notes.pdf` (8 pp) | `older-versions/` (1) |
| 03 | Real Analysis and Separation Theorems | Lecture 3 | `L03-Slides.pdf` (41 pp)<br>`L03-Proofs-Notes.pdf` (12 pp) | `older-versions/` (1) |
| 04 | Convex Function Characterizations | Lecture 4 | `L04-Slides.pdf` (94 pp)<br>`L04-Proofs-Notes.pdf` (12 pp) | `older-versions/` (1) |
| 05 | Subgradients and Convex Optimization | Lecture 5 | `L05-Slides.pdf` (47 pp)<br>`L05-Proofs-Notes.pdf` (9 pp) | `older-versions/` (1) |
| 06 | KKT Conditions and Constrained Optimization | Lecture 6 | `L06-Slides.pdf` (86 pp)<br>`L06-Proofs-Notes.pdf` (14 pp) |  |
| 07 | Duality Theory | Lecture 7 | `L07-Slides.pdf` (59 pp)<br>`L07-Proofs-Notes.pdf` (12 pp) |  |
| 08 | Gradient Monotonicity, Optimal Solutions and Quasiconvexity | Lecture 8 | `L08-Slides.pdf` (44 pp)<br>`L08-Proofs-Notes.pdf` (9 pp) |  |
| 09 | Lab - From Fritz John to KKT to Duality | Lecture 8 Lab | `L09-Slides.pdf` (30 pp) |  |
| 10 | Pseudoconvex Functions, Strong Convexity and Generalized Convexity | Lecture 9 | `L10-Slides.pdf` (66 pp)<br>`L10-Proofs-Notes.pdf` (20 pp)<br>`L10-Handout-ConvexityClasses.pdf` (12 pp) |  |
| 11 | Optimality Conditions and Line Search Methods | Lecture 10 | `L11-Slides.pdf` (44 pp)<br>`L11-Proofs-Notes.pdf` (14 pp)<br>`L11-Handout-LineSearch.pdf` (14 pp)<br>`L11-Handout-ScipyOptimization.pdf` (7 pp) |  |
| 12 | Steepest Descent and Line Search | Lecture 11 | `L12-Slides.pdf` (32 pp)<br>`L12-Proofs-Notes.pdf` (17 pp) | `alternate-decks/` (1) |
| 13 | Convergence of Gradient Descent | Lecture 11b (deck in the Lecture 11 folder) | `L13-Slides.pdf` (33 pp) |  |
| 14 | Newton's Method and Modifications | Lecture 12; notes labeled Lecture 13 | `L14-Slides.pdf` (44 pp)<br>`L14-Proofs-Notes.pdf` (9 pp) |  |
| 15 | Accelerated Methods and Convergence Rate Comparisons | Lecture 11c | `L15-Slides.pdf` (29 pp) |  |
| 16 | Conjugate Gradient | Lecture 18 (Beamer v9), Lecture 16 (tutorial); notes labeled Lectures 12, 14, 15 | `L16-Slides.pdf` (88 pp)<br>`L16-Tutorial-Conjugate-Gradient.pdf` (11 pp)<br>`L16-Proofs-Notes-1-Conjugate-Direction-Method.pdf` (9 pp)<br>`L16-Proofs-Notes-2-Conjugate-Direction-and-CG.pdf` (8 pp)<br>`L16-Proofs-Notes-3-Conjugate-Gradient.pdf` (9 pp) | `alternate-decks/` (2) |
| 17 | Trust Region Methods | Lecture 17 | `L17-Slides.pdf` (114 pp) |  |
| 18 | Quasi-Newton Methods - BFGS and L-BFGS | Lecture 19; alternates from Lecture 11b - BFGS and Lecture 14 | `L18a-Slides-BFGS.pdf` (30 pp)<br>`L18b-Slides-L-BFGS.pdf` (22 pp) | `alternate-decks/` (3), `older-versions/` (1) |
| 19 | Acceleration, Momentum, SGD | Lecture 20 | `L19-Slides.pdf` (28 pp) |  |
| 20 | Code Walkthrough | Lecture 21 | `L20-Slides.pdf` (85 pp) |  |
| 21 | Self-Concordance and CVXPY | Lecture 22 | `L21-Slides.pdf` (34 pp) |  |
| 22 | Log Barrier and IPOPT | Lecture 23 | `L22-Slides.pdf` (103 pp) | `older-versions/` (1) |
| 23 | SGD, Saddle Points, Splitting | Lecture 24 | `L23-Slides.pdf` (26 pp) |  |
| 24 | Acceleration, FISTA, Projection | Lecture 25 | `L24-Slides.pdf` (51 pp) |  |
| 25 | GD, CG, Momentum, Nesterov | Lecture 26 | `L25-Slides.pdf` (48 pp) |  |
| 26 | Deep Learning Optimization | Lecture 27 | `L26-Slides.pdf` (53 pp) |  |

## Read this before using the numbers

- **The PDFs still carry their Spring 2026 numbers.** Title pages, "last time" recaps and
  cross-references inside the decks use the labels in the third column, so from Lecture 09 on
  they differ from the folder number. Fixing that needs the LaTeX sources.
- **The quizzes use the Spring 2026 numbers too.** `../Quizzes/ISE5406_Quiz_Lectures8-9.html` and
  `...Lectures10-11.html` refer to the labels in the third column.
- **Lectures 12-18 were regrouped by topic.** In Spring 2026 the unconstrained-methods block was
  taught once in March (Lectures 11-15) and partly rebuilt in April (Lectures 11b, 16-19). Here
  the April decks are the main deck for conjugate gradient and quasi-Newton, and the March decks
  on the same topic are in `alternate-decks/`. That choice was made from file dates and content,
  so swap them if you prefer the shorter March versions.
- Proof notes for Lectures 14 and 16 were filed by content: their title pages carry a different
  lecture number from the deck they accompany.

## Subfolders inside a lecture

- `alternate-decks/`: a different deck on the same topic (earlier, longer or combined).
- `older-versions/`: an earlier build of the main deck, superseded by it.
- `figures/`: standalone figure PDFs used by the deck.
- `Lectures/_Reference - Rohit Kannan slides (earlier offering)/`: Rohit Kannan's slides for his
  Lectures 2, 5, 6, 7 and 8, which were kept alongside the 2026 decks.
