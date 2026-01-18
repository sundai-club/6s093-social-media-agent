# Workshop 1

1.
we want to make a script that does teh following: reads the docs from the business docs folder and passed them into a llm and generates a social media post for it and then posts it on the users mastodon account. we want to use openrouter and the openai api library for running teh llm. creds for the mastodon account will be in an .env file. we then also want to have a 2nd script that goes and searches for different keywords related to our business on mastadon and crafts responses that may or may not be related to out compnay. the reponses should be returned in teh form of a structured output using pydantic. dont actually post the reponses right now, expose a flag we can turn on to actually post them, but for now just print the outputs instead.
The scripts should be put into a fodler called workshop1 and use uv to run

2.
can you add a flag to confirm if we want to make a post to @workshop-1/post_generator.py

3.
in @workshop-1/keyword_responder.py can t we use reponse format={pydantic_class}

4.
for @workshop-1/keyword_responder.py only have the top 5 posts get passed to teh llm

5.
we dont need the models generating the post content for us, we can just add that back in after. same with author and original post id


# Workshop 2

1.
Please study the content of workshop-2 folder carefuly and integrate image generation into our agent. In particular, we need to at first generate a prompt for the diffusion model and then call the image generation api to create an image based on that prompt. Finally, we need to have the agent post the generated image along with the social media post on the user's Mastodon account. Make sure to handle any necessary authentication and API calls properly.

2.
add telegram bot integration so we can approve posts before they go live. send the generated post to telegram and wait for approve/reject button click


# Workshop 3

Prerequisites: A GCloud account with billing enabled ($300 free credits) and the gcloud CLI installed (brew install google-cloud-sdk on Mac).
First, figure out which GCloud account I am currently logged into and list all projects associated with that account. Set the default project to the one that contains “iap” in its name. Enable any required APIs, then deploy an e2-micro VM in that project (billing is already set up). Once the VM is created, use the gcloud CLI to SSH into the instance, install SQLite, and prepare the machine for application deployment.


Next, deploy a FastAPI server on the VM that uses SQLite as its database. Make sure the FastAPI app is configured as a proper Linux service (for example, using systemd) so that it continues running after the VM is restarted. Set up a cron job to run the post generator every day at 9am with Telegram approval still required before any post goes live. 
