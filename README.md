# FOR ALL

This project has been set up with uv. You should be able to use 

```bash
uv run workshop-{n}/[file].py
```

to see the code in action.

## Workshop 1 overview:

### Post generator

```bash
uv run workshop-1/post-generator.py --post
```

Takes the company docs and generates a post using a free model from OpenRouter and posts it on Mastodon. Remove the --post flag if you do not want to publish the post. 

### Keyword Responder

```bash
uv run workshop-1/keyword-responder.py --post
```

Searches 3 keywords based on Emanon (the company described in the business-docs folder) and uses structured outputs to generate responses to the top 5 posts.

Same as above, you can remove the --post flag if you do not want to publish the responses (highly recommended). The one part of the project that is untested because it is spamming other people's posts.

## Workshop 2 overview (Human-in-the-Loop):

See `workshop-2-hitloop/` folder for Telegram-based human approval workflow.

## Workshop 3 overview:

Workshop 3 focuses on **backend and cloud deployment** using Google Cloud Platform (GCP) and Claude Code. It builds on Workshop 1 by adding SQLite database tracking and a FastAPI server.

### Prerequisites

1. GCloud Account (cloud.google.com)
2. Billing Setup - $300 Free Credits
3. GCloud CLI installed (`brew install google-cloud-sdk` on Mac)

### FastAPI Server

```bash
uv run workshop-3/api.py
```

Runs a FastAPI server that provides REST endpoints to view posts, responses, and statistics stored in SQLite.

### Post Generator (with database)

```bash
uv run workshop-3/post_generator_db.py --post
```

Extends workshop-1's post generator - now saves posts to SQLite database.

### Keyword Responder (with database)

```bash
uv run workshop-3/keyword_responder_db.py --post
```

Extends workshop-1's keyword responder - now saves responses to SQLite database.

### Key Deployment Prompts (for Claude Code)

See `workshop-3/prompts.md` for full details. Key prompts:

1. `which gcloud account am I logged into and which projects does it have?`
2. `can we deploy a e2 vm to that project? Please turn on any necessary apis`
3. `we have this virtual machine in gcloud. you have the gcloud cli to ssh into that machine. We need to install sqlite`
4. `let's also deploy a fastapi server that uses that database and make sure the fast api is setup properly as a service on linux`

# WORKSHOP DESIGNERS

Please put the content for your workshop in a folder called workshop-[your workshop number]

Please build upon the previous workshop's code, so we have one cohesive project by the last workshop.

I have included reference docs for Emanon (renamed version of Vector Lab) in this repo as well in the business-docs folder that you can use as well.

Also please READ and TEST your code so that you know what it is doing and actually works, this will be used as ground truth for the other TAs, so any issues could derail things very quickly.

I also have kept a file highlighting the ALL prompts I used to get to my code for reference as well, I encourage you to do the same. 

# FOR TAs

The expected code for each day's workshop can be found in its respective folder. 

Each folder only contains the code needed for that workshop and does not include the previous day's code, since the students code will most likely look very different. 

They are there so that you can check to see what working function calls and env setup for the different services looks like so that you can diagnose when there is an issue with the student's implementation.
