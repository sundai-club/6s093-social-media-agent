# Workshop 3: Backend and Cloud Deployment

This workshop teaches students how to deploy the social media agent to Google Cloud Platform (GCP) using Claude Code and the gcloud CLI.

Workshop 3 builds on **Workshop 1** (post generation + keyword responses) and adds:
- SQLite database for tracking posts and responses
- FastAPI server for viewing data via REST API
- Cloud deployment to GCP VM

---

## Workshop 1 Prompts (Reference)

### Initial Setup

```
we want to make a script that does teh following: reads the docs from the business docs folder and passed them into a llm and generates a social media post for it and then posts it on the users mastodon account. we want to use openrouter and the openai api library for running teh llm. creds for the mastodon account will be in an .env file. we then also want to have a 2nd script that goes and searches for different keywords related to our business on mastadon and crafts responses that may or may not be related to out compnay. the reponses should be returned in teh form of a structured output using pydantic. dont actually post the reponses right now, expose a flag we can turn on to actually post them, but for now just print the outputs instead.
The scripts should be put into a fodler called workshop1 and use uv to run
```

### Refinements

```
can you add a flag to confirm if we want to make a post to @workshop-1/post_generator.py
```

```
in @workshop-1/keyword_responder.py can t we use reponse format={pydantic_class}
```

```
for @workshop-1/keyword_responder.py only have the top 5 posts get passed to teh llm
```

```
we dont need the models generating the post content for us, we can just add that back in after. same with author and original post id
```

---

## Workshop 2 Prompts (Reference - Human-in-the-Loop)

Workshop 2 adds Telegram approval workflow. See `workshop-2-hitloop/` folder.

---

## Workshop 3: Cloud Deployment

### Prerequisites

Before starting, students need:
1. GCloud Account (cloud.google.com)
2. Billing Setup - $300 Free Credits
3. GCloud CLI installed on your machine (`brew install google-cloud-sdk` on Mac)

### Learning Outcomes

1. How to use a Cloud Provider (GCP)
2. How to deploy from localhost to your VM
3. AI Enabled Deployment (using Claude Code)
4. SQLite and FastAPI

### Steps Overview

1. Setup your Google Cloud account
   a. Make the account and claim free credits (1 point)
   b. Create Project - Billing Project (1 point)
   c. Make VM (1 point)
2. Deploy a SQLite db on the VM (1 point)
3. Deploy FastAPI as a service (2 points)

---

## Workshop 3 Prompts - Cheat Sheet

### 1. Login and Setup GCloud Project

```
which gcloud account am I logged into and which projects does it have? Set the default project to the one with iap in the name
```

### 2. Create a VM

```
can we deploy a e2 vm to that project? Please turn on any necessary apis
```

If billing is not setup:
```
the billing is setup. and let's use e2-standard size
```

### 3. Install SQLite on the VM

```
we have this virtual machine in gcloud. you have the gcloud cli to ssh into that machine. We need to install sqlite
```

### 4. Deploy FastAPI as a Service

```
let's also deploy a fastapi server that uses that database and make sure the fast api is setup properly as a service on linux so it persists when we restart the instance
```

### 5. Cleanup (End of Workshop)

```
let's kill that vm so we save costs
```

```
delete that too and find any other items to clean up
```

---

## Manual Steps Reference

### Creating a GCP Project (via Console)

1. Go to: https://console.cloud.google.com/projectselector2/
2. Click "CREATE PROJECT"
3. Enter a project name (e.g., "sundai-iap")
4. Note the Project ID (may differ from name if not globally unique)
5. Click "CREATE"

### Recording Your Project ID

Find the ID value and record it:
```
PROJECT_ID=your-project-id-here
```

---

## Files in This Workshop

- `database.py` - SQLite database module with schema for posts and responses
- `api.py` - FastAPI server with REST endpoints
- `post_generator_db.py` - Extends workshop-1's post generator (saves to database)
- `keyword_responder_db.py` - Extends workshop-1's keyword responder (saves to database)

## Running Locally

```bash
# Run the FastAPI server locally
uv run workshop-3/api.py

# Generate a post (saves to database)
uv run workshop-3/post_generator_db.py

# Generate keyword responses (saves to database)
uv run workshop-3/keyword_responder_db.py
```

## API Endpoints

Once deployed, the FastAPI server provides:

- `GET /` - API information
- `GET /health` - Health check
- `GET /stats` - Statistics about posts and responses
- `GET /posts` - List all generated posts
- `GET /posts/{id}` - Get specific post
- `GET /responses` - List all generated responses
- `GET /responses/{id}` - Get specific response

## Systemd Service File (for VM deployment)

When deploying to the VM, Claude Code will create a systemd service file similar to:

```ini
[Unit]
Description=Social Media Agent API
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/social-media-agent
ExecStart=/home/your-username/.local/bin/uv run workshop-3/api.py
Restart=always
Environment=PORT=8000

[Install]
WantedBy=multi-user.target
```

## Bonus Tasks

1. **Chrome Remote Desktop** - Setup easy UI login to your VM
2. **Systemd Services** - Running persistent services on Linux with unit files
3. **Cloudflare Tunnel** - Expose API on your domain (bonus)

---

## Architecture Diagram

```
                    GCP Project
                    ┌─────────────────────────────────────────┐
                    │                                         │
┌─────────┐         │  ┌──────────┐      ┌─────────────────┐ │
│  Users  │─Access──│──│ VM       │      │   SQLite +      │ │
└─────────┘         │  │ Ubuntu   │──────│   Vectors       │ │
                    │  └──────────┘      └─────────────────┘ │
                    │       │                                 │
                    │  ┌────┴─────────────────────────┐      │
                    │  │         Services             │      │
                    │  │  ┌─────────┐ ┌──────────┐   │      │
                    │  │  │Scrapers │ │AI Agents │   │      │
                    │  │  └─────────┘ └──────────┘   │      │
                    │  │  ┌──────────────────────┐   │      │
                    │  │  │     Publishing       │   │      │
                    │  │  └──────────────────────┘   │      │
                    │  └──────────────────────────────┘      │
                    │                                         │
                    └─────────────────────────────────────────┘
                              │
                              │ APIs
                              ▼
                    ┌─────────────────────┐
                    │   External APIs     │
                    │  Twitter, LinkedIn  │
                    │  Claude, OpenAI     │
                    └─────────────────────┘
```
