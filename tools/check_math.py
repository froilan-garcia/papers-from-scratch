"""Check that the LaTeX in this repository survives GitHub's renderer.

GitHub rewrites Markdown before any mathematics is parsed, and two of its rules
destroy formulas silently:

  * a backslash before an ASCII punctuation character is a Markdown escape and
    is deleted, so `\\,` `\\;` `\\#` `\\{` reach the renderer as bare
    punctuation.  A backslash before a LETTER is left alone, which is why
    `\\frac` and `\\hat` always look fine and hide the problem;
  * asterisks pair up as emphasis and vanish, taking `X^*` with them.

Two spellings are immune and are the ones this repository uses: a fenced block
tagged `math` for displays, and `$`...`$` for inline mathematics carrying
anything vulnerable.  Plain `$...$` is fine, and more readable in the source,
when every escape is before a letter.

    python tools/check_math.py             offline, instant, run it always
    python tools/check_math.py --github    ask GitHub itself, needs the network

The offline pass applies the rules above.  The --github pass is the ground
truth: it posts each file to api.github.com/markdown, which needs no
authentication and works on a private repository, and reports any mathematics
that came back different from what went in.  That is how the rules above were
established in the first place, and it is the way to settle a new doubt rather
than reasoning about the specification.

Exits non-zero if anything is wrong, so it can be wired into a hook.
"""

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

# A backslash before anything that is not a letter, or an asterisk.
VULNERABLE = re.compile(r"\\[^A-Za-z]|\*")
INLINE = re.compile(r"\$([^$\n]+?)\$")
PROTECTED = re.compile(r"\$`[^`]+`\$")
# A backtick run opens a code span and only a run of the same length closes it,
# which is what lets ` ```math ` appear inside single backticks.
CODE_SPAN = re.compile(r"(`+)(.+?)\1")


def strip_fences(text):
    """Drop fenced blocks, whose contents no Markdown rule may touch."""
    out, fence = [], None
    for line in text.split("\n"):
        marker = re.match(r"\s*(`{3,})", line)
        if fence is None and marker:
            fence = len(marker.group(1))
            continue
        if fence is not None:
            if marker and len(marker.group(1)) >= fence:
                fence = None
            continue
        out.append(line)
    return "\n".join(out)


def bare_maths(text):
    """Every $...$ span that Markdown is free to rewrite."""
    text = strip_fences(text)
    text = PROTECTED.sub("", text)
    text = CODE_SPAN.sub("", text)
    return text


def tracked_markdown():
    files = subprocess.run(["git", "ls-files", "*.md"],
                           capture_output=True, text=True, check=True)
    return files.stdout.split()


def check_offline(paths):
    problems = 0
    for path in paths:
        text = open(path, encoding="utf-8").read()
        exposed = bare_maths(text)
        for m in INLINE.finditer(exposed):
            if VULNERABLE.search(m.group(1)):
                print(f"  {path}: ${m.group(1)}$")
                print("      -> wrap it as $`...`$")
                problems += 1
        if "$$" in exposed:
            print(f"  {path}: a $$ display outside a fenced block")
            print("      -> use a fenced block tagged math")
            problems += 1
    return problems


def render(markdown):
    request = urllib.request.Request(
        "https://api.github.com/markdown",
        data=json.dumps({"text": markdown, "mode": "markdown"}).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "papers-from-scratch"})
    html = urllib.request.urlopen(request, timeout=60).read().decode()
    text = re.sub(r"<[^>]+>", "", html)
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&quot;", '"'), ("&#39;", "'")]:
        text = text.replace(entity, char)
    return text


def check_github(paths):
    problems = 0
    for path in paths:
        source = open(path, encoding="utf-8").read()
        try:
            rendered = render(source)
        except urllib.error.HTTPError as err:
            if err.code == 403:
                print("  rate limited by GitHub (60 requests an hour "
                      "unauthenticated); try again later")
                return -1
            raise
        exposed = bare_maths(source)
        spans = [m.group(1) for m in INLINE.finditer(exposed)]
        spans += PROTECTED.findall(strip_fences(source))
        for span in spans:
            if span not in rendered:
                print(f"  {path}: ${span}$")
                print("      -> GitHub rewrote this one")
                problems += 1
        time.sleep(1)
    return problems


if __name__ == "__main__":
    use_github = "--github" in sys.argv
    paths = tracked_markdown()
    print(f"Checking {len(paths)} markdown files"
          f"{' against GitHub' if use_github else ''}...\n")

    problems = check_github(paths) if use_github else check_offline(paths)

    if problems < 0:
        sys.exit(2)
    if problems:
        print(f"\n{problems} problem(s). See the LaTeX section of CLAUDE.md.")
        sys.exit(1)
    print("No mathematics at risk.")
