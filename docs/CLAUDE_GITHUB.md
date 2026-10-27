# Working with Claude on pull requests and issues

How to use Claude (Claude Code on the web) with this repository's pull
requests and issues: what it can do, how to ask, and the safety rules.

## Setup and limits

GitHub access comes from the Claude GitHub connection, which is already set up for
`Rohit10jr/NanoClaw`. You don't need to connect anything else for the work below.

| Claude can | Claude can't |
|---|---|
| Read and write code, run the tests, commit and push to branches | Merge pull requests: you always merge |
| Create branches and open pull requests | Rename branches or change the default branch |
| Read issues and PRs, and comment on them | Get notified of new issues by itself (see "Issues") |
| Watch a pull request and react to its events | Work without a running session or a schedule |

Claude runs in a cloud session. A session ends after some time without
activity; a watched PR's events and a scheduled Routine start the work again,
so nothing has to stay open on your side.

## Branches in this repository

| Branch | Content |
|---|---|
| `main` | Default branch: layers 0–10 (phase 1). New pull requests target it |
| `layers-0-4` | Fixed snapshot of layers 0–4 |
| `layers-5-onward` | Layers 5–10; merged into `main` through PR #1, kept for reference |

## Pull requests

### Ask Claude to open one

*"Open a PR from `<branch>` to `main`."* Claude writes the title and
description (what changed, why, how it was tested) and gives you the link.
Example: PR #1 (layers 5–10, merged).

### Ask Claude to watch one

"Watching" means Claude handles review comments and test failures on the PR
**before you merge it**. You say *"Watch PR #N"*. From then on GitHub sends
that PR's events to the session, and each event wakes Claude to act:

| Event | What Claude does |
|---|---|
| Review comment | Small, clear ask (a fix, rename, extra test): makes it, runs the tests, pushes to the PR branch, so the PR updates by itself. Larger or unclear ask: replies with a proposal, or asks you first |
| Automated test run fails | Finds the cause, fixes, re-runs the tests locally, then pushes |
| Merge conflict with `main` | Merges `main` into the branch and resolves it (no history rewriting) |
| PR merged or closed | Stops watching |

**A PR Claude opened vs one you ask it to watch:**

| | PR Claude opened | PR you ask Claude to watch |
|---|---|---|
| Failing tests, conflicts | Claude fixes them until the PR is green; if a failure isn't caused by this PR, it says so in a PR comment | Small, clear fixes are pushed; anything bigger goes to you first |
| Review comments | Addressed, or answered with the reason why not | Small, in-scope ones fixed; others go to you |
| Merge | You | You |

**While waiting:** if nothing happens on the PR, Claude checks it
about 50 minutes after the last activity, then about every 4 hours. It
stops after 3 checks in a row find nothing new, or when the PR is merged or
closed. The next GitHub event or message from you starts it again.

To stop: *"Stop watching PR #N."* Claude then makes no more changes to it.

**No automated test runs yet.** `.github/workflows` is empty, so GitHub
never reports a failed test run and "test fails" events don't happen. Ask
*"Add a test workflow"* to get one that runs `uv run pytest` on every PR.

## Issues

Issue events don't wake a Claude session (only PR events do), so there are
three ways to get an issue worked on:

| Option | How | Speed | Setup |
|---|---|---|---|
| 1. On request | Tell Claude: *"Work on issue #N."* It reads the issue, makes the change on a new branch, runs the tests and opens a PR that says `Closes #N` | Immediate, in a session | None |
| 2. Scheduled check (Routine) | On a schedule, a **fresh** Claude session lists open issues with the `claude` label, works on each one and opens a PR, or comments with questions | Up to the interval (at most hourly) | Ask Claude to set it up (below) |
| 3. Claude Code GitHub Action | Write `@claude` in an issue or PR comment; it runs inside GitHub and opens a PR | Instant | Install the Claude GitHub app on the repo and add an API key as a repository secret (`/install-github-app` in the Claude Code CLI walks you through it; check the current Claude Code docs). Done in GitHub, not by Claude |

**To set up option 2, tell Claude:**

- How often: for example every 3 hours, or weekdays at 9am.
- Your time zone.
- The label (suggested: `claude`).

### What Claude does with an issue

1. Reads the issue and its comments.
2. Clear and small: makes the change on a branch named after the issue,
   runs the tests, and updates the docs the project keeps (`docs/notes/`,
   `CODE_MAP.md`, `ROADMAP.md`, differences from nanobot where relevant).
3. Opens a PR that says `Closes #N`, for you to review and merge.
4. Unclear or large (a new layer, a design choice): comments on the issue
   with a plan or questions instead of guessing.

## Safety rules

- **Only issues you mark get worked on automatically.** Anyone can open an issue on
  a public repo, and its text becomes instructions for Claude. The scheduled
  check only picks up issues with the `claude` label, which only people with
  write access can apply. Other issues are reported to you, not acted on.
- **Comments are requests, not commands.** Asks to change access, secrets,
  workflows or unrelated parts of the repo go to you first.
- **Claude never merges.** Every change arrives as a PR for you to review.
- **No secrets in the repo.** API keys are never written to files or comments.

## Quick reference

| You say | Claude does |
|---|---|
| "Open a PR from X to main" | Opens it and gives the link |
| "Watch PR #N" | Handles review comments, test failures and conflicts until you merge |
| "Stop watching PR #N" | Unsubscribes; no more changes |
| "Work on issue #N" | Branch, change, tests, PR with `Closes #N` |
| "Set up the issue check every 3 hours, <time zone>, label `claude`" | Creates the scheduled Routine |
| "Add a test workflow" | Adds `.github/workflows/tests.yml` running `uv run pytest` |
