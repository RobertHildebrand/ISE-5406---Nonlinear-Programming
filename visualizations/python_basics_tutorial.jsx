import React, { useState } from "react";
import { Terminal, Package } from "lucide-react";
import { CopyCodeButton, DownloadNotebookButton } from "./code_panel_utils.jsx";

/* ============================================================
   PYTHON BASICS — TUTORIAL FOR ISE 5406
   ISE 5406

   For students who are new (or rusty) at Python. Covers:
     • how to install Python
     • environments and package managers
     • the standard scientific stack (numpy, pandas, matplotlib,
       scipy, geopandas, shapely)
     • core syntax: variables, lists, loops, functions, printing
     • plotting basics

   Each topic is self-contained — pick a tab, read, copy the code,
   run it in your own notebook.
   ============================================================ */

// ============================================================
// Topic registry
// ============================================================
const TOPICS = [
  {
    key: "install",
    name: "1. Installing Python",
    body: () => <InstallSection />,
  },
  {
    key: "envs",
    name: "2. Environments & packages",
    body: () => <EnvsSection />,
  },
  {
    key: "ecosystem",
    name: "3. Key packages — what each one does",
    body: () => <EcosystemSection />,
  },
  {
    key: "syntax",
    name: "4. Core syntax",
    body: () => <SyntaxSection />,
  },
  {
    key: "loops",
    name: "5. Loops & conditionals",
    body: () => <LoopsSection />,
  },
  {
    key: "functions",
    name: "6. Functions",
    body: () => <FunctionsSection />,
  },
  {
    key: "numpy",
    name: "7. NumPy arrays",
    body: () => <NumpySection />,
  },
  {
    key: "pandas",
    name: "8. Pandas dataframes",
    body: () => <PandasSection />,
  },
  {
    key: "matplotlib",
    name: "9. Plotting with Matplotlib",
    body: () => <MatplotlibSection />,
  },
  {
    key: "next",
    name: "10. Where to go next",
    body: () => <NextSection />,
  },
];

// ============================================================
// Main component
// ============================================================
export default function PythonBasicsTutorial() {
  const [tab, setTab] = useState(TOPICS[0].key);
  const topic = TOPICS.find((t) => t.key === tab);
  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Python Basics for ISE 5406
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        New to Python or want a refresher? Walk through the topics below in
        order, or jump to whatever you need. Every code block has a copy button
        — paste straight into a Jupyter notebook to run it. The aim is the 20%
        of Python that covers 80% of what you'll do in this course.
      </p>

      <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
        {TOPICS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            style={{ ...tabBtn, ...(tab === t.key ? tabBtnActive : {}) }}
          >
            {t.name}
          </button>
        ))}
      </div>

      {topic.body()}
    </div>
  );
}

// ============================================================
// Section 1 — Install
// ============================================================
function InstallSection() {
  return (
    <div>
      <H2>How to install Python</H2>
      <P>
        Python comes pre-installed on macOS and most Linux distributions, but
        the version is usually old. For ISE 5406 you want{" "}
        <Code>Python 3.10 or newer</Code>. Pick ONE of the routes below.
      </P>

      <H3>Option A — Anaconda (easiest for ISE)</H3>
      <P>
        Anaconda is a Python distribution that bundles Python plus the entire
        scientific stack (NumPy, SciPy, pandas, matplotlib, Jupyter,
        scikit-learn) in a single installer. Recommended if you don't already
        have Python set up.
      </P>
      <Steps items={[
        <>Go to <ExtLink href="https://www.anaconda.com/download">anaconda.com/download</ExtLink>, get the installer for your OS.</>,
        <>Run the installer. Accept the defaults.</>,
        <>Open a NEW terminal window (so the installer's PATH changes take effect). Type <Code>python --version</Code> — should report 3.x.</>,
        <>Type <Code>jupyter lab</Code> to launch the notebook editor in your browser.</>,
      ]} />

      <H3>Option B — Miniconda (lighter)</H3>
      <P>
        Same package manager (conda) but skips the bundled GUIs. Good if you
        prefer minimal installs.
      </P>
      <CodeBlock code={`# macOS / Linux
brew install --cask miniconda      # macOS via Homebrew
# OR
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Windows: download the .exe from
# https://docs.conda.io/projects/miniconda/en/latest/`} title="install miniconda" />

      <H3>Option C — Plain Python via pyenv</H3>
      <P>
        For developers who don't want conda. <Code>pyenv</Code> manages
        multiple Python versions in parallel.
      </P>
      <CodeBlock code={`# macOS
brew install pyenv

# Then install + activate a Python version
pyenv install 3.11.9
pyenv global 3.11.9
python --version`} title="pyenv route" />

      <H3>Option D — Use Google Colab (no install at all)</H3>
      <P>
        Colab is a free, hosted Jupyter environment with Python and most
        packages pre-installed. Great for quick experiments.
        Open <ExtLink href="https://colab.research.google.com/">colab.research.google.com</ExtLink>{" "}
        in your browser and start a new notebook — that's it.
      </P>

      <Tip>
        If you only do this course on one laptop, install Anaconda. If you'll
        switch laptops often or use cloud machines, just live in Colab.
      </Tip>
    </div>
  );
}

// ============================================================
// Section 2 — Environments & packages
// ============================================================
function EnvsSection() {
  return (
    <div>
      <H2>Virtual environments and package managers</H2>
      <P>
        A <i>virtual environment</i> is an isolated Python install with its own
        set of packages. It prevents project A's dependencies from breaking
        project B's. <b>Use one environment per project</b>; treat it as
        disposable.
      </P>

      <H3>With conda</H3>
      <CodeBlock code={`# Create a new env named "ise5406" with Python 3.11
conda create -n ise5406 python=3.11

# Activate it (do this every time you open a new terminal)
conda activate ise5406

# Install packages into the active env
conda install numpy pandas matplotlib scipy jupyter

# Some packages prefer the conda-forge channel
conda install -c conda-forge geopandas pyomo ipopt

# Anything not on conda? Use pip inside the active env:
pip install amplpy

# When you're done
conda deactivate

# List your envs / remove one
conda env list
conda env remove -n ise5406`} title="conda environment workflow" />

      <H3>With venv + pip (built-in to Python)</H3>
      <CodeBlock code={`# Create a venv (just a folder named .venv)
python -m venv .venv

# Activate
source .venv/bin/activate          # macOS/Linux
.venv\\Scripts\\activate             # Windows

# Install packages
pip install numpy pandas matplotlib scipy jupyter

# Save the exact list to share with classmates / TA / instructor
pip freeze > requirements.txt

# Reproduce the env elsewhere
pip install -r requirements.txt`} title="venv + pip workflow" />

      <H3>pip vs conda — which?</H3>
      <table style={tbl}>
        <thead>
          <tr><th style={th}>aspect</th><th style={th}>pip</th><th style={th}>conda</th></tr>
        </thead>
        <tbody>
          <tr><td style={td}>installs from</td><td style={td}>PyPI (pypi.org)</td><td style={td}>Anaconda + conda-forge</td></tr>
          <tr><td style={td}>Python-only</td><td style={td}>yes</td><td style={td}>no — also handles C/Fortran libs (e.g. IPOPT, GDAL)</td></tr>
          <tr><td style={td}>speed (pre-built)</td><td style={td}>fast</td><td style={td}>slower (resolves harder)</td></tr>
          <tr><td style={td}>system deps</td><td style={td}>you handle (compiler, headers)</td><td style={td}>conda installs them</td></tr>
        </tbody>
      </table>
      <Tip>
        For ISE 5406: use conda (or amplpy.modules) for binaries with native
        code (IPOPT, HiGHS, geopandas, scipy), and pip for pure-Python packages
        not on conda (amplpy, gurobipy).
      </Tip>
    </div>
  );
}

// ============================================================
// Section 3 — Ecosystem (what each package does)
// ============================================================
function EcosystemSection() {
  const PACKAGES = [
    { name: "numpy", inst: "conda install numpy", what: "N-dimensional arrays + fast linear algebra. The bedrock of scientific Python — every other package on this list either uses it or is compatible with it.", use: "Vectors, matrices, dot products, broadcasting, FFT, random sampling." },
    { name: "scipy", inst: "conda install scipy", what: "Scientific algorithms layered on top of NumPy. Optimization, integration, sparse linear algebra, statistics, signal processing.", use: "scipy.optimize.minimize for NLP, scipy.sparse for big matrices, scipy.stats for distributions." },
    { name: "pandas", inst: "conda install pandas", what: "DataFrames — labeled, tabular data. Like a spreadsheet you control with code. Reads CSV, Excel, SQL, parquet.", use: "Cleaning, merging, grouping, summarizing data before you feed it into a model." },
    { name: "matplotlib", inst: "conda install matplotlib", what: "Standard 2D plotting. Slightly verbose but extremely flexible.", use: "Line plots, scatter plots, contour, bar charts, axes, subplots." },
    { name: "seaborn", inst: "conda install seaborn", what: "Higher-level statistical plots, built on matplotlib. Pretty defaults, less code.", use: "boxplots, pair plots, heatmaps from a dataframe." },
    { name: "jupyter", inst: "conda install jupyter", what: "Notebook interface — run Python in cells with prose / equations / plots inline.", use: "Course assignments, demos, interactive exploration. Run with `jupyter lab` (modern) or `jupyter notebook` (classic)." },
    { name: "scikit-learn", inst: "conda install scikit-learn", what: "Classical machine learning — regression, classification, clustering, dimensionality reduction. Consistent API across every model.", use: "from sklearn.linear_model import LogisticRegression — fit / predict / score." },
    { name: "pyomo", inst: "conda install -c conda-forge pyomo ipopt", what: "Algebraic modeling for mathematical programming (LP / MIP / NLP). Builds models you ship to a solver like IPOPT, Gurobi, CBC.", use: "Course tutorials use this for every NLP example." },
    { name: "amplpy", inst: "pip install amplpy && python -m amplpy.modules install ipopt highs", what: "Python interface to AMPL — a stand-alone modeling language and solver framework. Cleaner syntax than Pyomo for certain problem styles.", use: "ISE 5406 has a dedicated amplpy tutorial." },
    { name: "cvxpy", inst: "pip install cvxpy", what: "DSL for CONVEX optimization. Knows whether your problem is convex; if so, automatically picks an appropriate solver.", use: "LPs / QPs / SOCPs / SDPs with very natural syntax." },
    { name: "gurobipy", inst: "pip install gurobipy", what: "Gurobi solver's Python API. Industrial-grade MIP solver. Free academic license required.", use: "Production-scale LP / IP / MIP. Course tutorial walks through warm-starts, callbacks, sensitivity." },
    { name: "geopandas", inst: "conda install -c conda-forge geopandas", what: "Pandas DataFrames with geographic geometry — points, lines, polygons. Built on shapely + fiona + pyproj.", use: "Reading shapefiles, GIS data, spatial joins, plotting maps. Common for facility-location and routing problems." },
    { name: "shapely", inst: "conda install -c conda-forge shapely", what: "Pure geometric operations — buffer, intersect, union, contains. Used internally by geopandas.", use: "Geometric computations without the GIS overhead." },
    { name: "networkx", inst: "conda install networkx", what: "Graphs and network algorithms — shortest paths, max flow, MST, traversal.", use: "Course network-flow demo uses this. Also for graph generation, layout, drawing." },
    { name: "torch", inst: "pip install torch", what: "PyTorch — tensors and autograd, plus neural-network primitives. The 'NumPy on GPU + automatic differentiation'.", use: "Deep learning, but also any optimization where you want gradients for free." },
    { name: "tqdm", inst: "pip install tqdm", what: "Progress bars for any Python loop. One import, one wrap.", use: "Long-running optimization loops, training, data downloads." },
  ];
  return (
    <div>
      <H2>Key packages — what each one does</H2>
      <P>
        These are the workhorses. You won't use all of them, but it helps to
        know which one to reach for. Click the install command to copy.
      </P>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {PACKAGES.map((p) => (
          <PkgCard key={p.name} pkg={p} />
        ))}
      </div>
    </div>
  );
}

function PkgCard({ pkg }) {
  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 12, background: "#fff" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
        <span style={{ fontFamily: "monospace", fontWeight: 700, fontSize: 16, color: "#0b3da0" }}>{pkg.name}</span>
        <code style={{ fontFamily: "monospace", fontSize: 12, background: "#f0eee9", padding: "2px 8px", borderRadius: 4, color: "#444" }}>
          {pkg.inst}
        </code>
        <CopyCodeButton code={pkg.inst} label="copy install" />
      </div>
      <div style={{ marginTop: 6, fontSize: 13.5, lineHeight: 1.55, color: "#222" }}>{pkg.what}</div>
      <div style={{ marginTop: 4, fontSize: 12.5, color: "#555", fontStyle: "italic" }}>Used for: {pkg.use}</div>
    </div>
  );
}

// ============================================================
// Section 4 — Core syntax
// ============================================================
function SyntaxSection() {
  return (
    <div>
      <H2>Core syntax — variables, types, operators, printing</H2>
      <P>
        Python has dynamic typing: you don't declare a variable's type, you
        just assign. Everything is an object.
      </P>
      <CodeBlock title="variables and types" code={`# Numbers
x = 3                # int
y = 2.5              # float
z = 1 + 2j           # complex (rare)

# Strings — single or double quotes (no difference)
name = "Alice"
greeting = f"Hello, {name}!"           # f-string — embed expressions

# Booleans
ok = True
done = False

# None — the "no value" sentinel
result = None

# Containers
nums = [1, 2, 3, 4]                     # list (mutable, ordered)
coords = (1.5, 2.5)                     # tuple (immutable, ordered)
lookup = {"alice": 30, "bob": 41}       # dict (key-value)
unique = {1, 2, 3}                      # set (no duplicates)

# Type-checking when you need it
type(x)              # <class 'int'>
isinstance(x, int)   # True

# Printing
print("z =", x + y)               # space-separated args
print(f"name = {name}, age = 30") # f-string is the modern way
print(f"pi ≈ {3.14159:.2f}")      # format specifier inside f-string
`} />

      <H3>Operators</H3>
      <CodeBlock title="arithmetic & comparison" code={`# Arithmetic
a + b, a - b, a * b, a / b   # / is FLOAT division (3 / 2 = 1.5)
a // b                        # integer (floor) division: 7 // 2 = 3
a % b                         # modulo: 7 % 2 = 1
a ** b                        # power: 2 ** 10 = 1024

# Comparison
a == b, a != b
a < b, a >= b
a < b < c                     # chained — equivalent to a<b AND b<c

# Logical
True and False                # short-circuit
not True
x is None                     # 'is' for identity, '==' for equality

# Membership
3 in [1, 2, 3]                # True
"a" in "alphabet"             # True
`} />

      <Tip>
        Use <Code>f"…"</Code> for all string formatting — it's faster, clearer,
        and more recent than <Code>"%s"</Code> or <Code>"{ }".format(...)</Code>.
      </Tip>
    </div>
  );
}

// ============================================================
// Section 5 — Loops & conditionals
// ============================================================
function LoopsSection() {
  return (
    <div>
      <H2>Loops, conditionals, and comprehensions</H2>
      <P>
        Indentation defines blocks (no curly braces). Use 4 spaces. Mixing tabs
        and spaces will cause errors — let your editor handle this.
      </P>
      <CodeBlock title="if / elif / else" code={`x = 10

if x > 0:
    print("positive")
elif x == 0:
    print("zero")
else:
    print("negative")

# Ternary expression
sign = "positive" if x > 0 else "non-positive"
`} />

      <CodeBlock title="for loops" code={`# Iterate over a list
for n in [1, 2, 3, 4]:
    print(n ** 2)

# range(start, stop, step) — gives a sequence of ints
for i in range(5):              # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 10, 2):       # 2, 4, 6, 8
    print(i)

# Iterate with index AND value
fruits = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits):
    print(i, fruit)

# Iterate two lists in parallel
prices = [1.0, 0.5, 2.0]
for fruit, price in zip(fruits, prices):
    print(f"{fruit}: {price:.2f}")

# Iterate dict items
ages = {"alice": 30, "bob": 41}
for name, age in ages.items():
    print(f"{name} is {age}")
`} />

      <CodeBlock title="while + break + continue" code={`# while loop
n = 100
while n > 1:
    n = n // 2
    print(n)

# break: exit early; continue: skip to next iteration
for i in range(20):
    if i % 7 == 0:
        continue              # skip multiples of 7
    if i > 15:
        break                 # stop entirely
    print(i)
`} />

      <H3>List comprehensions — the Pythonic shortcut</H3>
      <CodeBlock title="comprehensions" code={`# Squares of 0..9
squares = [n ** 2 for n in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With a filter
even_squares = [n ** 2 for n in range(10) if n % 2 == 0]
# [0, 4, 16, 36, 64]

# Dict comprehension
inverse = {v: k for k, v in {"a": 1, "b": 2}.items()}
# {1: 'a', 2: 'b'}

# Set comprehension
unique_lengths = {len(w) for w in ["hi", "hello", "hi", "world"]}
# {2, 5}

# Generator expression — same syntax with () instead of []
# Good for big ranges (no list materialized in memory)
total = sum(n ** 2 for n in range(10_000_000))
`} />

      <Tip>
        Comprehensions read FAST once you're used to them. Use them for any
        "build a list / dict / set from another list / dict / set" pattern.
      </Tip>
    </div>
  );
}

// ============================================================
// Section 6 — Functions
// ============================================================
function FunctionsSection() {
  return (
    <div>
      <H2>Functions</H2>
      <CodeBlock title="defining and calling" code={`def square(x):
    """Return x squared. (This is a docstring — accessible via help(square).)"""
    return x ** 2

print(square(7))             # 49

# Default arguments
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Alice")                       # "Hello, Alice!"
greet("Bob", greeting="Hola")        # "Hola, Bob!"  (keyword arg)

# Multiple return values (actually a tuple)
def divmod_pair(a, b):
    return a // b, a % b

q, r = divmod_pair(17, 5)            # q=3, r=2

# *args and **kwargs — variable-length arguments
def total(*args):
    return sum(args)

total(1, 2, 3, 4)                    # 10

def render(name, **kwargs):
    print(name, kwargs)

render("Alice", age=30, city="NYC")  # Alice {'age': 30, 'city': 'NYC'}

# Lambdas — single-expression functions
square2 = lambda x: x ** 2
square2(7)                           # 49

# Sorting with a key function
words = ["pear", "fig", "apple", "blueberry"]
sorted(words, key=len)               # ['fig', 'pear', 'apple', 'blueberry']
`} />

      <H3>Type hints (optional, recommended)</H3>
      <CodeBlock title="type hints" code={`def sum_squares(xs: list[float]) -> float:
    return sum(x * x for x in xs)

# Type hints are PURELY documentation — Python does not enforce them.
# But editors / type-checkers (mypy, pyright) will warn on misuse.
`} />
    </div>
  );
}

// ============================================================
// Section 7 — NumPy
// ============================================================
function NumpySection() {
  return (
    <div>
      <H2>NumPy — arrays and vectorized math</H2>
      <P>
        NumPy gives you fast arrays. Operations apply element-wise — no
        explicit loops. This is THE foundation of scientific Python.
      </P>
      <CodeBlock title="numpy basics" code={`import numpy as np

# Creating arrays
a = np.array([1, 2, 3, 4])              # 1D
b = np.array([[1, 2], [3, 4]])          # 2D
np.zeros(5)                              # [0, 0, 0, 0, 0]
np.ones((3, 4))                          # 3x4 of ones
np.arange(0, 10, 2)                      # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)                     # [0, 0.25, 0.5, 0.75, 1]
np.eye(3)                                # 3x3 identity
np.random.rand(2, 3)                     # 2x3 uniform [0, 1) random

# Shape and indexing
b.shape                                  # (2, 2)
b[0, 1]                                  # 2  (row 0, col 1)
b[:, 0]                                  # column 0 → array([1, 3])
b[0, :]                                  # row 0    → array([1, 2])

# Slicing — same as Python lists, just multi-dimensional
arr = np.arange(20).reshape(4, 5)
arr[1:3, ::2]                            # rows 1-2, every 2nd column

# Element-wise math (NO loops needed!)
x = np.array([1, 2, 3])
x * 2                                    # array([2, 4, 6])
x + 10                                   # array([11, 12, 13])
np.sin(x)                                # element-wise sin
x ** 2                                   # array([1, 4, 9])

# Linear algebra
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
np.linalg.solve(A, b)                    # solve Ax = b
A @ b                                    # matrix-vector product (also np.dot(A, b))
np.linalg.inv(A)                         # inverse
np.linalg.eig(A)                         # eigenvalues / eigenvectors

# Reductions
arr.sum(), arr.mean(), arr.std()
arr.sum(axis=0)                          # sum each column
arr.sum(axis=1)                          # sum each row
`} />
      <Tip>
        Rule of thumb: if you're writing a Python <Code>for</Code> loop over a
        NumPy array, you're probably doing it wrong. Express the operation as
        an array-level expression — usually 10–100× faster.
      </Tip>
    </div>
  );
}

// ============================================================
// Section 8 — Pandas
// ============================================================
function PandasSection() {
  return (
    <div>
      <H2>Pandas — labeled tables</H2>
      <P>
        A pandas <Code>DataFrame</Code> is a table — rows have an index,
        columns have names, and each column has a type. Like Excel, but
        scriptable.
      </P>
      <CodeBlock title="dataframes" code={`import pandas as pd
import numpy as np

# Create a DataFrame from a dict
df = pd.DataFrame({
    "name": ["alice", "bob", "carol", "dan"],
    "age":  [30, 41, 25, 38],
    "city": ["NYC", "Chicago", "NYC", "LA"],
})
print(df)

#       name  age     city
#  0   alice   30      NYC
#  1     bob   41  Chicago
#  2   carol   25      NYC
#  3     dan   38       LA

# Reading from a file
# df = pd.read_csv("data.csv")
# df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# Selection
df["age"]                    # one column → Series
df[["name", "age"]]          # multiple columns → DataFrame
df.iloc[0]                    # first row by position
df.loc[df.age > 30]           # filter — rows where age > 30

# Adding / modifying columns
df["age_in_5"] = df["age"] + 5
df["adult"] = df["age"] >= 18

# Group + aggregate
df.groupby("city")["age"].mean()
#  city
#  Chicago    41.0
#  LA         38.0
#  NYC        27.5

# Common stats
df.describe()                 # count, mean, std, min, 25%, 50%, 75%, max
df["age"].mean()
df["city"].value_counts()

# Joining two dataframes
prices = pd.DataFrame({"city": ["NYC", "LA", "Chicago"], "rent": [3500, 2800, 1900]})
df.merge(prices, on="city")    # like SQL JOIN

# Save
df.to_csv("people.csv", index=False)
df.to_excel("people.xlsx", index=False)
`} />
    </div>
  );
}

// ============================================================
// Section 9 — Matplotlib
// ============================================================
function MatplotlibSection() {
  return (
    <div>
      <H2>Matplotlib — plotting</H2>
      <P>
        Two APIs: a quick, MATLAB-like one (<Code>plt.plot(…)</Code>), and an
        object-oriented one (<Code>fig, ax = plt.subplots()</Code>). The
        OO version is preferred for anything beyond a single plot.
      </P>
      <CodeBlock title="line plot — quick API" code={`import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y, label="sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Simple sine wave")
plt.legend()
plt.grid(True)
plt.show()
`} />

      <CodeBlock title="multiple subplots — OO API" code={`import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))    # 1 row, 2 cols

axs[0].plot(x, np.sin(x), color="blue")
axs[0].set_title("sin")
axs[0].grid(True)

axs[1].plot(x, np.cos(x), color="red")
axs[1].set_title("cos")
axs[1].grid(True)

fig.suptitle("Trig functions")
plt.tight_layout()
plt.savefig("trig.png", dpi=150)        # save to file
plt.show()
`} />

      <CodeBlock title="scatter + contour — types you'll use a lot" code={`import matplotlib.pyplot as plt
import numpy as np

# Scatter
n = 200
x, y = np.random.randn(n), np.random.randn(n)
labels = (x + y > 0).astype(int)

plt.figure(figsize=(5, 5))
plt.scatter(x, y, c=labels, cmap="coolwarm", edgecolors="k")
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.title("2-class scatter")
plt.show()

# Contour of a function f(x, y)
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = X**2 + Y**2

fig, ax = plt.subplots(figsize=(5, 5))
ax.contourf(X, Y, Z, levels=20, cmap="viridis")
ax.contour(X, Y, Z, levels=10, colors="white", linewidths=0.5)
ax.set_title("f(x, y) = x² + y²")
plt.show()
`} />
      <Tip>
        Forget to call <Code>plt.show()</Code> in a script and the window won't
        appear. In a Jupyter notebook, plots render automatically — no
        <Code>show()</Code> needed.
      </Tip>
    </div>
  );
}

// ============================================================
// Section 10 — Where to go next
// ============================================================
function NextSection() {
  return (
    <div>
      <H2>Where to go next</H2>
      <ul style={{ lineHeight: 1.7, fontSize: 14, color: "#222", paddingLeft: 22 }}>
        <li>
          <b>Optimization (course core).</b> Walk through the Pyomo + IPOPT
          tutorial in this collection. Then try CVXPY for the convex side and
          AMPL for the textbook-style modeling language.
        </li>
        <li>
          <b>Data wrangling.</b> Find a CSV from your domain. Practice
          <Code>read_csv → groupby → describe → matplotlib</Code> until it's
          automatic.
        </li>
        <li>
          <b>NumPy fluency.</b> Project Euler problems, particularly the early
          ones, are great for vectorized-thinking practice.
        </li>
        <li>
          <b>Effective Python (book).</b> Brett Slatkin. Reads in a weekend; the
          59 short items lift you from "writes Python" to "writes idiomatic
          Python".
        </li>
        <li>
          <b>Real-Python.com</b> for tutorials, and the official{" "}
          <ExtLink href="https://docs.python.org/3/tutorial/">Python tutorial</ExtLink>{" "}
          if you want to fill gaps.
        </li>
        <li>
          <b>scikit-learn user guide.</b> Best ML reference. Read the API for
          the model you're using before tuning anything.
        </li>
      </ul>
      <Tip>
        Best way to learn: pick a small project (one of the homework problems
        in this course is plenty) and rewrite it three times — once
        scrappy-quick, once carefully, once in a Jupyter notebook with a
        narrative. By round three you'll know which patterns matter.
      </Tip>
    </div>
  );
}

// ============================================================
// Reusable bits
// ============================================================
function H2({ children }) {
  return <h2 style={{ fontSize: 22, fontWeight: 800, marginTop: 6, marginBottom: 10, color: "#1f4e3d" }}>{children}</h2>;
}
function H3({ children }) {
  return <h3 style={{ fontSize: 16, fontWeight: 700, marginTop: 18, marginBottom: 6, color: "#222" }}>{children}</h3>;
}
function P({ children }) {
  return <p style={{ fontSize: 14, lineHeight: 1.6, color: "#222", marginTop: 0, marginBottom: 12 }}>{children}</p>;
}
function Code({ children }) {
  return (
    <code style={{
      fontFamily: "monospace", fontSize: 13.5,
      background: "#f0eee9", padding: "1px 6px", borderRadius: 4, color: "#0b3da0",
    }}>
      {children}
    </code>
  );
}
function ExtLink({ href, children }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: "#0b3da0", textDecoration: "underline" }}>
      {children}
    </a>
  );
}
function Steps({ items }) {
  return (
    <ol style={{ paddingLeft: 22, lineHeight: 1.6, fontSize: 14, color: "#222" }}>
      {items.map((it, i) => <li key={i} style={{ marginBottom: 4 }}>{it}</li>)}
    </ol>
  );
}
function Tip({ children }) {
  return (
    <div style={{
      marginTop: 12, marginBottom: 12,
      padding: "10px 14px",
      background: "#fff8e1", border: "1px solid #f5d68d", borderRadius: 8,
      fontSize: 13.5, lineHeight: 1.55,
    }}>
      <Terminal size={14} style={{ verticalAlign: "middle", marginRight: 6 }} />
      <b>Tip.</b> {children}
    </div>
  );
}
function CodeBlock({ code, title }) {
  return (
    <div style={{ marginBottom: 14 }}>
      {title && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span style={{ fontFamily: "monospace", fontSize: 11, color: "#888", letterSpacing: "0.12em", textTransform: "uppercase" }}>
            # {title}
          </span>
          <CopyCodeButton code={code} />
          <DownloadNotebookButton
            code={code}
            filename={`python_basics_${(title || "snippet").replace(/[^a-z0-9]+/gi, "_").toLowerCase()}.ipynb`}
            title={title || "Python snippet"}
            description=""
            label=".ipynb"
          />
        </div>
      )}
      <pre style={{
        background: "#1f1d1a",
        color: "#e8e2d4",
        padding: "12px 14px",
        borderRadius: 6,
        fontFamily: "'JetBrains Mono', Menlo, ui-monospace, monospace",
        fontSize: 12.5,
        lineHeight: 1.55,
        margin: 0,
        whiteSpace: "pre",
        overflowX: "auto",
      }}>
        {code}
      </pre>
    </div>
  );
}

// ============================================================
// Style atoms
// ============================================================
const tabBtn = {
  padding: "8px 14px",
  border: "1px solid #ccc",
  borderRadius: 6,
  background: "#fff",
  cursor: "pointer",
  fontWeight: 500,
  fontSize: 13,
};
const tabBtnActive = {
  background: "#1f4e3d",
  color: "#fff",
  border: "1px solid #1f4e3d",
};
const tbl = {
  width: "100%",
  borderCollapse: "collapse",
  marginBottom: 14,
  fontSize: 13.5,
};
const th = {
  padding: "6px 10px",
  borderBottom: "2px solid #888",
  textAlign: "left",
  fontFamily: "monospace",
  fontSize: 12,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  color: "#555",
};
const td = {
  padding: "6px 10px",
  borderBottom: "1px solid #eee",
  verticalAlign: "top",
};
