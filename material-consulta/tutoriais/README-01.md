# Tutorial 1 — Setting Up Your Working Environment

**MM845 — Tópicos de Geometria III: AI for Geometry**
Paired with **Lecture 1: Coding Foundations — Python, Git & AI Agents**

---

## What this tutorial is for

By the end of this document you will have a working, *reproducible* computational
environment: a Python installation you control, an isolated environment for this
course, Jupyter for interactive experiments, VS Code as an editor, Git/GitHub for
version control, and at least one AI coding assistant.

**No prior programming experience is assumed.** Every command is given explicitly.
Where the three operating systems differ, all three are shown — read only your own.

> **Conventions used below**
> - Lines starting with `$` are typed into a *terminal* (do not type the `$`).
> - `<angle brackets>` mark something you must replace with your own value.
> - Anything marked **(optional)** can be skipped on a first pass.

---

## Contents

1. [Opening a terminal](#1-opening-a-terminal)
2. [Installing Python](#2-installing-python)
3. [Creating the course environment](#3-creating-the-course-environment)
4. [Jupyter notebooks](#4-jupyter-notebooks)
5. [Visual Studio Code](#5-visual-studio-code)
6. [Git and GitHub](#6-git-and-github)
7. [AI coding assistants](#7-ai-coding-assistants)
8. [Repository structure and reproducibility](#8-repository-structure-and-reproducibility)
9. [Troubleshooting](#9-troubleshooting)
10. [Final checklist](#10-final-checklist)
11. [(Optional) SageMath](#11-optional-sagemath)

---

## 1. Opening a terminal

The *terminal* (or *shell*, or *command line*) is a text interface to your computer.
You type a command, press <kbd>Enter</kbd>, and the computer executes it. Almost
everything below happens here.

| OS | How to open |
|---|---|
| **Linux** | <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>T</kbd>, or search "Terminal" in your applications menu. |
| **macOS** | <kbd>Cmd</kbd>+<kbd>Space</kbd>, type `Terminal`, press <kbd>Enter</kbd>. |
| **Windows** | Press <kbd>Win</kbd>, type `PowerShell`, press <kbd>Enter</kbd>. (After step 2 you will instead use the **Miniforge Prompt**.) |

Four commands are enough to navigate:

```bash
pwd                 # "print working directory" — where am I?
ls                  # list the files here   (Windows PowerShell: also `ls`)
cd <folder>         # enter a folder
cd ..               # go up one level
```

Try them now:

```bash
$ cd ~                # go to your home directory
$ pwd                 # confirm where you are
$ ls                  # see what is there
```

(`~` means your home directory on all three systems.) In §6.8 you will download
the course repository here, which creates `~/MM845` — so there is no folder to
make by hand.

---

## 2. Installing Python

Python 3 may already exist on your machine, but you should **not** use the system
Python: modifying it can break your operating system's own tools. Instead we install
a self-contained distribution that also gives us the environment manager discussed
in Lecture 1.

We use **Miniforge**, a minimal installer for `conda` configured to use the
community `conda-forge` package repository. It is free, open-source, works
identically on all three operating systems, and — unlike full Anaconda — has no
commercial licensing restrictions for institutional use.

Download page (all installers): <https://github.com/conda-forge/miniforge#download>

### Linux

```bash
$ curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
$ bash Miniforge3-$(uname)-$(uname -m).sh
```

Accept the licence, accept the default install location (`~/miniforge3`), and answer
**yes** when asked whether to initialise conda in your shell. Then close and reopen
the terminal.

### macOS

Identical to Linux (the `uname` calls pick the right installer for both Apple
Silicon and Intel machines):

```bash
$ curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
$ bash Miniforge3-$(uname)-$(uname -m).sh
```

Close and reopen the terminal afterwards.

<details>
<summary>macOS alternative: Homebrew</summary>

If you already use [Homebrew](https://brew.sh):

```bash
$ brew install miniforge
$ conda init zsh
```
</details>

### Windows

1. Download `Miniforge3-Windows-x86_64.exe` from the
   [releases page](https://github.com/conda-forge/miniforge/releases/latest).
2. Run it. Choose **"Just Me"** and accept the default location.
3. When installation finishes, open **Miniforge Prompt** from the Start menu
   (search for "Miniforge"). **Use this prompt, not plain PowerShell,** for every
   `conda` command in this document.

### Verify

In a *newly opened* terminal:

```bash
$ conda --version
conda 24.x.x
```

If the command is not found, see [Troubleshooting](#9-troubleshooting).

---

## 3. Creating the course environment

An **environment** is an isolated Python installation with its own set of libraries
at pinned versions. Lecture 1's analogy: it is the computational counterpart of
fixing your conventions and axioms before writing a proof. Two projects needing
incompatible versions of a library can then coexist peacefully.

Create the environment for this course:

```bash
$ conda create -n aigeo python=3.12
$ conda activate aigeo
```

Your prompt should now be prefixed with `(aigeo)`. **You must run `conda activate
aigeo` in every new terminal** before working on the course.

### Install the scientific stack

```bash
$ conda install -c conda-forge numpy scipy matplotlib pandas scikit-learn sympy networkx jupyterlab ipykernel
```

Then install PyTorch (the deep-learning library used from Lecture 3 onwards). It is
distributed separately because it must match your hardware:

```bash
# CPU-only — correct for laptops, and sufficient for this entire course
$ pip install torch torchvision
```

<details>
<summary>If you have an NVIDIA GPU (Linux/Windows)</summary>

Use the selector at <https://pytorch.org/get-started/locally/> to obtain the exact
command for your CUDA version, e.g.

```bash
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

On Apple Silicon, the ordinary `pip install torch` already supports GPU
acceleration through the `mps` backend — nothing extra is needed.
</details>

### Verify the installation

```bash
$ python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
```

You should see two version numbers and no error.

### Record the environment

Reproducibility requires that a colleague can recreate exactly what you ran. Export
the specification into a file that lives in your repository:

```bash
$ conda env export --from-history > environment.yml
```

Anyone can then rebuild it with `conda env create -f environment.yml`.

> **Useful conda commands**
> | Command | Effect |
> |---|---|
> | `conda env list` | list all environments |
> | `conda activate <name>` | switch into an environment |
> | `conda deactivate` | leave the current environment |
> | `conda list` | list packages installed in the active environment |
> | `conda env remove -n <name>` | delete an environment |

---

## 4. Jupyter notebooks

A **notebook** is a document mixing executable code cells, their outputs (numbers,
plots, tables), and prose in Markdown/LaTeX. It is the natural medium for
experimental mathematics: you compute, look, adjust, and the whole record of what
you did is preserved. All tutorials in this course are distributed as notebooks.

### Launching

With `(aigeo)` active, from your course folder:

```bash
$ jupyter lab
```

A browser tab opens at `http://localhost:8888`. The terminal must stay open while
you work — it is running the server. Stop it with <kbd>Ctrl</kbd>+<kbd>C</kbd>.

Create a new notebook: **File → New → Notebook**, and choose the **Python 3
(ipykernel)** kernel.

### The essentials

A notebook is a sequence of **cells**. Each cell is either *Code* or *Markdown*.

| Action | Shortcut |
|---|---|
| Run cell, stay | <kbd>Ctrl</kbd>+<kbd>Enter</kbd> |
| Run cell, move to next | <kbd>Shift</kbd>+<kbd>Enter</kbd> |
| New cell below | <kbd>Esc</kbd> then <kbd>B</kbd> |
| New cell above | <kbd>Esc</kbd> then <kbd>A</kbd> |
| Delete cell | <kbd>Esc</kbd> then <kbd>D</kbd> <kbd>D</kbd> |
| Convert to Markdown | <kbd>Esc</kbd> then <kbd>M</kbd> |
| Convert to Code | <kbd>Esc</kbd> then <kbd>Y</kbd> |

Test it. Paste this into a code cell and run it — it samples a point cloud on
$S^2$, the running example of Tutorial 1:

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)          # fixed seed => reproducible

def sample_sphere(n, rng):
    """Sample n points uniformly on the unit sphere S^2 in R^3."""
    x = rng.normal(size=(n, 3))              # isotropic Gaussian
    return x / np.linalg.norm(x, axis=1, keepdims=True)

X = sample_sphere(500, rng)
assert np.allclose(np.linalg.norm(X, axis=1), 1.0)   # sanity check

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=6)
ax.set_box_aspect((1, 1, 1))
plt.show()
```

Note the two habits from Lecture 1 already present: an explicit **seed**, and an
**assertion** verifying a property you know must hold.

### The one trap to know about

A notebook's *cell order on screen* need not be its *execution order*. The numbers
in `In [7]:` tell you the true order. If results stop making sense, use
**Kernel → Restart Kernel and Run All Cells** to guarantee a clean linear run.
Always do this before committing a notebook or showing results to anyone.

> **Classic Jupyter Notebook.** `jupyter lab` is the modern interface and is
> recommended. If you prefer the older, simpler one, `pip install notebook` and run
> `jupyter notebook`. The file format (`.ipynb`) is identical.

---

## 5. Visual Studio Code

VS Code is a free editor from Microsoft. It gives you file browsing, an integrated
terminal, a debugger, native notebook support, Git integration, and — importantly
for this course — the AI assistants of §7, all in one window.

### Install

| OS | Method |
|---|---|
| **Linux** | Download the `.deb`/`.rpm` from <https://code.visualstudio.com/download>, or `sudo snap install code --classic`. |
| **macOS** | Download the `.zip` from the same page, unzip, drag **Visual Studio Code** into `/Applications`. Or `brew install --cask visual-studio-code`. |
| **Windows** | Download and run the **User Installer** from the same page. Tick *"Add to PATH"* when offered. |

### Extensions

Open the Extensions panel (<kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>X</kbd>)
and install:

- **Python** (`ms-python.python`) — language support, environment selection, debugging.
- **Jupyter** (`ms-toolsai.jupyter`) — run `.ipynb` notebooks inside the editor.

**(optional but recommended)**

- **Ruff** (`charliermarsh.ruff`) — fast linter/formatter, catches mistakes as you type.
- **GitLens** (`eamodio.gitlens`) — richer view of Git history.

### Point VS Code at the `aigeo` environment

This is the step most often missed, and the cause of most "but I installed it!"
errors.

1. Open your course folder: **File → Open Folder…**
2. Press <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> to open the
   Command Palette.
3. Type `Python: Select Interpreter`, press <kbd>Enter</kbd>.
4. Choose the entry containing **`aigeo`**.

For notebooks, the kernel is chosen separately: open a `.ipynb` file and click the
kernel name at the **top right**, then select `aigeo`.

### Worth knowing

| Feature | How |
|---|---|
| Integrated terminal | <kbd>Ctrl</kbd>+<kbd>`</kbd> |
| Command Palette (everything lives here) | <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> |
| Search across all files | <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>F</kbd> |
| Source Control (Git) panel | <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>G</kbd> |
| Run a Python file | <kbd>Ctrl</kbd>+<kbd>F5</kbd> |

---

## 6. Git and GitHub

**Git** tracks the history of a project as a directed acyclic graph of snapshots
(*commits*) — Lecture 1, slide 7. **GitHub** hosts those repositories online and
adds collaboration: pull requests, issues, code review.

### 6.1 Install Git

| OS | Command |
|---|---|
| **Linux (Debian/Ubuntu)** | `sudo apt update && sudo apt install git` |
| **Linux (Fedora)** | `sudo dnf install git` |
| **macOS** | `xcode-select --install` (installs Git with the developer tools), or `brew install git` |
| **Windows** | Download **Git for Windows** from <https://git-scm.com/download/win>. Accept the defaults; this also installs *Git Bash*. |

Verify:

```bash
$ git --version
git version 2.4x.x
```

### 6.2 Create a GitHub account

Do this before configuring Git, because the next step needs the email address you
register here.

Sign up at <https://github.com>. Use your **institutional address**
(`@unicamp.br`, `@ime.unicamp.br`, …) or add it later under
*Settings → Emails* — it is what makes you eligible for the free
[GitHub Student Developer Pack](https://education.github.com/pack), which in turn
gives free **GitHub Copilot** (§7.1).

Enable two-factor authentication when prompted; GitHub requires it.

> Git and GitHub are independent: Git is the version-control program on your
> machine, GitHub is a company hosting Git repositories. You can use Git with no
> account at all. The account matters from §6.4 onwards, when you start pushing
> your work somewhere other people (and your future self, on another machine) can
> reach it.

### 6.3 One-time configuration

Git stamps every commit with a name and email. Use the ones on the account you
just created — otherwise GitHub cannot match your commits to your profile, and
your contribution history will silently point at nobody:

```bash
$ git config --global user.name  "<Your Name>"
$ git config --global user.email "<you@example.com>"
$ git config --global init.defaultBranch main
$ git config --global pull.rebase false
```

Check what Git believes at any time with `git config --global --list`.

> **Privacy note.** Commit emails are public in a public repository. If you would
> rather not expose yours, GitHub can supply a no-reply address: *Settings →
> Emails → Keep my email addresses private* shows one of the form
> `12345678+username@users.noreply.github.com`. Use that as `user.email` instead.

### 6.4 Authenticate your machine

GitHub stopped accepting account passwords over Git in 2021, so `git push` needs
credentials of its own. The standard answer is an **SSH key**: a key pair whose
public half you hand to GitHub and whose private half never leaves your machine.
Set it up once and Git stops asking you for anything.

**1. Generate a key.** Press <kbd>Enter</kbd> at every prompt to accept the
defaults (a passphrase is optional; leaving it empty is fine on a personal
machine):

```bash
$ ssh-keygen -t ed25519 -C "<you@example.com>"
```

**2. Copy the public half.** Note the `.pub` — this is the half that is safe to
share. Never copy the other file.

```bash
$ cat ~/.ssh/id_ed25519.pub
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... you@example.com
```

Copy that entire line.

**3. Give it to GitHub.** Go to *Settings → SSH and GPG keys → New SSH key*, paste
the line into the **Key** box, give it a title such as `laptop`, and save.

**4. Test it.**

```bash
$ ssh -T git@github.com
Hi <your-username>! You've successfully authenticated...
```

Say `yes` if asked about authenticity the first time. From now on, use the
`git@github.com:` form of repository URLs and Git will never ask for a password.

<details>
<summary>On Windows</summary>

Run these in **Git Bash** (installed with Git for Windows), not in PowerShell —
`ssh-keygen` and the `~/.ssh/` path both behave as they do on Linux there.
</details>

<details>
<summary>Alternative: HTTPS with a personal access token</summary>

If SSH is blocked on your network, use HTTPS instead. Create a token at
*Settings → Developer settings → Personal access tokens → Tokens (classic)* with
the **repo** scope, then tell Git to remember it:

```bash
$ git config --global credential.helper store   # macOS: osxkeychain
```

The first `git push` asks for a username and password — give your username and
paste the **token** as the password. Note that tokens expire and must be renewed.
</details>

### 6.5 The everyday workflow

```bash
$ git clone <repository-url>       # download a repository (once)
$ cd <repository-name>

# ... edit files ...

$ git status                       # what has changed?
$ git diff                         # show me exactly what changed
$ git add <file>                   # stage a file for the next snapshot
$ git add .                        # stage everything changed
$ git commit -m "Add S^2 sampling and norm check"
$ git push                         # upload commits to GitHub
$ git pull                         # download collaborators' commits
```

Inspect history at any time:

```bash
$ git log --oneline --graph --all   # the DAG of snapshots, drawn
$ git show <commit-hash>            # what did that commit change?
```

### 6.6 Branches

A **branch** is an independent line of development — the co-authored-paper analogy
from Lecture 1. Do your work on a branch, keep `main` clean:

```bash
$ git switch -c tutorial-1          # create a branch and move onto it
# ... work, add, commit ...
$ git push -u origin tutorial-1     # publish the branch (first time only)

$ git switch main                   # go back
$ git merge tutorial-1              # bring the work in
```

`git branch` lists branches; `git switch <name>` moves between them.

### 6.7 Pull requests

On GitHub, a **pull request** (PR) proposes merging one branch into another and
displays the *diff* for review — refereeing, for code.

Like forking, this happens on GitHub's side; Git itself has no notion of a pull
request. Push your branch, then open the repository page — GitHub shows a
*"Compare & pull request"* button for any branch you have recently pushed. Press
it, write a sentence saying what the change does, and submit.

### 6.8 Fork the course repository

Every tutorial in this course lives in one repository, and you work in your own
copy of it. That copy is a **fork**: your own repository on GitHub, which remembers
where it came from.

The memory is the point. **Each tutorial is published shortly before its session**,
so the repository grows week by week, and you will collect each new tutorial with a
single command (§6.9) — while your own work from previous weeks stays exactly where
you left it.

> You will also need a repository for the assessed mini-project, but **not yet**.
> In week 7 you will be asked to create a public one and share it with us.

The course repository is

<https://github.com/TomasSilva/MM845>

**Do this once.**

**1. Fork it on GitHub.** Forking happens on GitHub's side, not on your machine,
so there is no Git command for it. Open the page above while logged in and press
**Fork** at the top right, then **Create fork**. You now own a copy at
`https://github.com/<your-username>/MM845`.

**2. Clone your fork.**

```bash
$ cd ~
$ git clone git@github.com:<your-username>/MM845.git
$ cd MM845
```

The clone creates a folder named after the repository, so you are now in
`~/MM845`. **That folder is where you work for the rest of the course** — every
command from here on assumes you are inside it.

**3. Add the course repository as a second remote**, so you can collect new
tutorials from it:

```bash
$ git remote add upstream git@github.com:TomasSilva/MM845.git
```

**4. Check.**

```bash
$ git remote -v
origin      git@github.com:<your-username>/MM845.git  (fetch)
origin      git@github.com:<your-username>/MM845.git  (push)
upstream    git@github.com:TomasSilva/MM845.git       (fetch)
upstream    git@github.com:TomasSilva/MM845.git       (push)
```

Two remotes, and they mean different things:

- **`origin`** is *your* fork. You push here. You have write access.
- **`upstream`** is *ours*. You pull from here. You have no write access, which is
  a feature — you cannot break the course materials by accident.

**5. Teach Git one new command.** Collecting each week's tutorial is three
commands (§6.9), so we bundle them into one. Paste this line exactly — it is long,
and the code block scrolls sideways rather than wrapping:

```bash
$ git config --global alias.get-tutorial '!f() { git fetch upstream && git checkout upstream/main -- ":/$1" && (git diff --cached --quiet || git commit -m "Collect $1") && echo "OK: $1 is ready"; }; f'
```

That defines `git get-tutorial`, which you will use once a week. There is nothing
magic about it: a Git **alias** is just a nickname for a longer command, stored in
the same `--global` configuration as your name and email (§6.3). Read it back with

```bash
$ git config --global --get alias.get-tutorial
```

<details>
<summary>If you set up HTTPS rather than SSH in §6.4</summary>

Use the `https://` forms throughout:

```bash
$ cd ~
$ git clone https://github.com/<your-username>/MM845.git
$ cd MM845
$ git remote add upstream https://github.com/TomasSilva/MM845.git
```
</details>

> **Forks are public.** GitHub does not allow a private fork of a public
> repository. If you would rather your working notebooks were not visible, say so
> in the session and we will arrange a private repository with `upstream` added by
> hand instead.

### 6.9 Before every class: collect the new tutorial

The repository grows as the course runs: each tutorial is published as a new
`tutorial_n/` folder in the days before its session. So every week you collect one
folder — and **nothing you have already done is touched**, whatever state it is in.

#### The weekly routine

One command, run before each session. Replace `tutorial_03` with whichever folder
is new that week — we will tell you in class.

```bash
$ cd ~/MM845
$ git get-tutorial tutorial_03
OK: tutorial_03 is ready
```

That is the whole thing. Then back up your fork whenever you like:

```bash
$ git push
```

Running `git get-tutorial` twice is harmless: it simply reports that the folder is
ready again. If you mistype the folder name, it says
`error: pathspec ... did not match` and does nothing at all.

<details>
<summary>What the alias actually runs</summary>

Nothing you could not type yourself:

```bash
$ git fetch upstream                            # 1. download our history
$ git checkout upstream/main -- tutorial_03     # 2. take just the new folder
$ git commit -m "Collect tutorial_03"           # 3. keep it
```

1. **`git fetch upstream`** downloads our new commits into your repository and
   **changes none of your files**. Fetching is always safe; it is the half of
   `git pull` that cannot hurt you.
2. **`git checkout upstream/main -- tutorial_03`** reaches into that downloaded
   history and takes out *exactly one folder*, placing it in your working tree
   already staged. Because you name the folder, Git will not write anywhere else.
3. **`git commit`** records it. No `git add` is needed — step 2 staged the files
   for you.

Use these directly if the alias ever misbehaves, or if you are on a machine where
you have not set it up.
</details>

> **The guarantee.** It writes inside `tutorial_03/` and nowhere else. Your
> `tutorial_02/` survives exactly as you left it — committed or not, finished or
> not, and even if you scribbled all over the notebook we gave you. There is no
> merge, so there is nothing to conflict.

> ### ⚠ Do not press GitHub's "Sync fork" button
>
> Your fork's page on GitHub offers a **Sync fork** button, and it looks like
> exactly what you want. It is not. Once your fork contains your own commits, that
> button can offer to **discard** them so as to match our repository again — and
> that means your work, gone, in one click.
>
> `git get-tutorial` never does this. Use it and ignore the button.

Not sure which folder is new? Ask Git what we have been doing:

```bash
$ git log --oneline HEAD..upstream/main     # commits you do not have yet
```

<details>
<summary>Why not just <code>git pull upstream main</code>?</summary>

Because `pull` merges *everything*, and a merge can reach into files you have
edited. Notebooks are JSON, and Git merges them line by line with no idea what a
cell is — so if you had edited `tutorial_02/spheres.ipynb` and we later fixed a
cell in it, `git pull` would leave a **merge conflict in the middle of that JSON**.
Resolvable in principle; thoroughly unpleasant in practice.

Naming a single folder sidesteps the whole question. You collect what is new and
keep what is yours.
</details>

#### Working on a tutorial

Once the folder has arrived, **work in a copy**:

```bash
$ cd tutorial_03
$ cp <the-distributed-notebook>.ipynb  work_<yourname>.ipynb
```

This is a recommendation rather than a rule — the routine above protects you
either way — but it is worth doing. It keeps your answers clearly distinguishable
from the material, and it means the original stays pristine for reference.

It also buys you an undo button. If you ever wreck a distributed file, take a
fresh copy straight from us:

```bash
$ git checkout upstream/main -- tutorial_03/<the-distributed-notebook>.ipynb
```

> **Branches are optional here.** §6.6 is worth knowing and you should practise it,
> but nobody else pushes to your fork, so working directly on `main` is perfectly
> safe for the tutorials.

> **Found a bug in a tutorial?** Fix it on a branch, push it, and open a pull
> request (§6.7). It arrives as a reviewable diff — §6.7's refereeing analogy, for
> real. Corrections from students are welcome and will be merged.

### 6.10 Do this now

1. Fork and clone the course repository as in §6.8, including the
   `git get-tutorial` alias in step 5.
2. Confirm `git remote -v` lists **both** `origin` (yours) and `upstream` (ours).
3. Run `git get-tutorial tutorial_01`. You already have that folder, so it should
   simply reply `OK: tutorial_01 is ready` — which confirms the alias, the
   `upstream` remote and your network all work, before you need them next week.
4. Make a trivial commit to check your write access to `origin`: create a file
   `students/<yourname>.md` saying who you are and what you want from the course,
   then
   ```bash
   $ git add students/<yourname>.md
   $ git commit -m "Add <yourname>"
   $ git push
   ```
5. Refresh your fork's page on GitHub and confirm the file is there.

> **The mini-project repository comes later.** In week 7 you will be asked to
> create a *separate*, public repository for it and share the link with us. Nothing
> to do now.

### 6.11 What not to commit

Git is for *source*, not for generated output. Large binaries and data files bloat
the history permanently, and secrets committed once stay in the history forever.
Create a file named `.gitignore` at the root of your repository:

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
env/

# Jupyter
.ipynb_checkpoints/

# Data and results — regenerate these, do not version them
data/raw/
results/
*.npy
*.npz
*.pt
*.ckpt

# OS / editor
.DS_Store
.vscode/
.idea/

# Secrets — never commit these
.env
*.key
```

> Notebooks store their outputs inside the `.ipynb` file, which makes diffs noisy.
> Clearing outputs before committing (**Kernel → Restart Kernel and Clear All
> Outputs**) keeps history readable.

---

## 7. AI coding assistants

Lecture 1 makes the case: syntax is no longer the barrier to entry, so your
mathematical precision in *specifying* a problem becomes the valuable skill. Three
levels of autonomy were distinguished — autocomplete, chat, and agents.

**The course uses one assistant: GitHub Copilot.** It is free for verified students,
it lives inside the editor you set up in §5, and it provides all three levels of
autonomy. Set it up before Tutorial 2. Other agents exist and some will be
demonstrated in class (§7.2), but nothing in this course requires you to pay for
anything.

### 7.1 GitHub Copilot

Copilot lives inside VS Code and provides all three levels: inline completion, a
chat panel, and an agent mode that edits multiple files and runs commands.

1. **Get access.** Apply for the
   [Student Developer Pack](https://education.github.com/pack) with your Unicamp
   address. Verified students get the free **Copilot Student** plan — unlimited
   code completions, plus a monthly allowance of GitHub AI Credits for chat, agent
   mode, code review and the Copilot CLI. See the
   [current plans](https://docs.github.com/en/copilot/get-started/plans).

   **Apply early.** Verification takes days, occasionally longer, and GitHub has
   paused new Copilot sign-ups before (April–June 2026). Until it comes through
   you land on **Copilot Free**, which works with any GitHub account and is
   perfectly usable for this course — just with tighter limits.
2. **Install the extension.** In VS Code, search the Extensions panel for
   **GitHub Copilot** (`GitHub.copilot`) and install it. **GitHub Copilot Chat**
   comes with it.
3. **Sign in.** A prompt appears at the bottom right; otherwise click the account
   icon in the Activity Bar → *Sign in to GitHub*.
4. **Use it.**
   - *Inline*: start typing; grey ghost text is a suggestion — <kbd>Tab</kbd>
     accepts, <kbd>Esc</kbd> dismisses.
   - *Chat*: <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Alt</kbd>+<kbd>I</kbd>. Ask
     questions about the open file; `#file` and `#selection` add context.
   - *Inline edit*: <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>I</kbd> on a selection,
     describe the change in words.
   - *Agent mode*: select **Agent** in the Chat panel's mode dropdown, then give a
     goal rather than an instruction.

### 7.2 Other agents — demonstrated, not required

[Claude Code](https://claude.com/claude-code) and
[OpenAI Codex](https://developers.openai.com/codex) are terminal agents in the same
family, and both will be shown in the tutorial sessions so that you can see what the
current state of the art looks like. **Both require a paid subscription**, neither
has a student free tier, and nothing in this course needs them. Do not buy anything
on the course's account.

### 7.3 Using it properly

This part matters more than the installation. From Lecture 1, slide 10:

**Do**
- Give precise context: definitions, conventions, expected input/output shapes.
- Work in small, verifiable steps, and **commit after each working step** — Git is
  your undo button when an agent goes astray.
- Ask for tests and sanity checks against cases where you know the answer.
- Ask it to *explain* its code, and read the explanation.

**Avoid**
- Accepting long generated code you have not read.
- Vague prompts (*"make it work"*).
- Letting an agent run without version control.
- Assuming numerical output is correct because the program did not crash.

**Prompt pattern** — *role + task + context + constraints + verification*:

> You are an expert mathematician assisting with differential geometry
> computations. Write a NumPy function sampling `N` points uniformly on
> $S^2 \subset \mathbb{R}^3$. Input: `N`, `seed`. Output: array of shape `(N, 3)`.
> Include a test verifying that all point norms equal 1.

Compare with *"give me points on a sphere"*, and note which one you could referee.

---

## 8. Repository structure and reproducibility

Use one repository per project, laid out predictably:

```
mm845-<yourname>/
├── README.md            # what this is, how to run it
├── environment.yml      # the pinned environment (§3)
├── .gitignore           # what Git should ignore (§6.11)
├── data/                # inputs; raw data ignored by Git, generators committed
├── src/                 # reusable functions and modules
├── notebooks/           # exploratory work, one notebook per tutorial
└── results/             # figures, tables, trained models (ignored by Git)
```

The standard set in Lecture 1: **a colleague should be able to clone your repository
and reproduce your results in two actions** — create the environment, then run the
script.

Three things must be controlled for that to hold:

| Ingredient | How |
|---|---|
| **Code** | Git — commit early, commit often, with messages that say *why*. |
| **Environment** | `environment.yml`, kept up to date in the repository. |
| **Randomness** | Fix seeds explicitly; report behaviour averaged over several. |

Seeding, concretely:

```python
import numpy as np, torch, random

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)   # preferred NumPy style: an explicit generator
```

---

## 9. Troubleshooting

**`conda: command not found`**
The installer did not modify your shell configuration, or you did not reopen the
terminal. Close and reopen it. If it persists:

```bash
$ ~/miniforge3/bin/conda init "$(basename "$SHELL")"
```

then reopen the terminal. On Windows, use the **Miniforge Prompt** rather than
PowerShell.

**`ModuleNotFoundError: No module named 'numpy'` (or similar)**
Almost always the wrong environment. Check:

```bash
$ conda env list          # is (aigeo) the active one?
$ which python            # Windows: where python — is it inside miniforge3/envs/aigeo?
```

In VS Code, re-run `Python: Select Interpreter`. In a notebook, check the kernel
name at the top right.

**A notebook cannot see the `aigeo` environment**
Register the kernel explicitly:

```bash
$ conda activate aigeo
$ python -m ipykernel install --user --name aigeo --display-name "Python (aigeo)"
```

**`git push` asks for a password and rejects it**
GitHub removed password authentication for Git in 2021. Set up an SSH key (§6.4),
and check that your remote uses the `git@github.com:` form — `git remote -v` will
tell you. To switch an existing clone over:

```bash
$ git remote set-url origin git@github.com:<your-username>/MM845.git
```

**PowerShell refuses to run a script** (`running scripts is disabled`)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows, generally**
If a tool works only under a Unix-like shell, install **WSL2** (Windows Subsystem
for Linux) — `wsl --install` in an *administrator* PowerShell, then reboot — and
follow the Linux instructions inside it.

**Merge conflict** (`CONFLICT (content): Merge conflict in ...`)
Git marks the disputed region in the file with `<<<<<<<`, `=======`, `>>>>>>>`.
Edit the file so it reads as you want, delete the markers, then:

```bash
$ git add <file>
$ git commit
```

Nothing is lost: `git merge --abort` returns you to the state before the merge.

---

## 10. Final checklist

Before Tutorial 2, confirm each of the following:

- [ ] A terminal opens and `conda --version` prints a version.
- [ ] `conda activate aigeo` works, and the prompt shows `(aigeo)`.
- [ ] `python -c "import numpy, scipy, matplotlib, sklearn, torch; print('ok')"` prints `ok`.
- [ ] `jupyter lab` opens in a browser and a code cell runs.
- [ ] The $S^2$ sampling snippet in §4 runs and produces a picture.
- [ ] VS Code opens your course folder and its interpreter is set to `aigeo`.
- [ ] `git --version` works and `git config --global user.name` returns your name.
- [ ] You have a GitHub account, and `ssh -T git@github.com` greets you by name.
- [ ] You forked and cloned the course repository, and `git remote -v` shows both
      `origin` (yours) and `upstream` (ours).
- [ ] `git get-tutorial tutorial_01` replies `OK: tutorial_01 is ready`.
- [ ] You committed and pushed `students/<yourname>.md` to your fork, and can see
      it on GitHub.
- [ ] You know the weekly routine (§6.9): **`git get-tutorial tutorial_n`** before
      each class — and that GitHub's "Sync fork" button is not a substitute.
- [ ] GitHub Copilot is installed in VS Code and responds (§7.1) — on the free
      student plan, or on Copilot Free while verification is pending.

If any box is unticked, bring it to the tutorial session — that is what it is for.

---

## 11. (Optional) SageMath

SageMath is a computer algebra system built on Python, with a great deal of
geometry and number theory available directly. Tutorial 1 uses it for constructing
charts and differential objects on manifolds. It is a large install; if it gives you
trouble, use the browser-based option and move on.

| Route | How |
|---|---|
| **conda** (recommended, Linux/macOS) | `conda create -n sage -c conda-forge sage python=3.12` — a *separate* environment from `aigeo`, since Sage pins many of its own dependencies. |
| **Linux** | `sudo apt install sagemath` (Debian/Ubuntu) — often an older version. |
| **macOS** | The conda route above, or the `.dmg` from <https://www.sagemath.org/download-mac.html>. |
| **Windows** | Install via WSL2 and follow the Linux instructions. |
| **No install at all** | <https://cocalc.com> runs Sage in the browser; free accounts suffice for the tutorials. |

Once installed, `sage -n jupyter` launches a Jupyter server with the Sage kernel
available.

---

## Further reading

- [Python tutorial (official)](https://docs.python.org/3/tutorial/) — the language itself.
- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Pro Git](https://git-scm.com/book/en/v2) — chapters 1–3 cover everything used here, free online.
- [Matplotlib quick-start guide](https://matplotlib.org/stable/users/explain/quick_start.html)
- [PyTorch: Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [The Turing Way](https://the-turing-way.netlify.app/) — reproducible research practice, in depth.

---

*Questions and problems: raise them in the tutorial session, or open an issue on the
course repository.*
