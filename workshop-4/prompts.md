# Workshop 1

1.
we want to make a script that does the following: reads the docs from the business docs folder and passed them into a llm and generates a social media post for it and then posts it on the users mastodon account. we want to use openrouter and the openai api library for running the llm. creds for the mastodon account will be in an .env file. we then also want to have a 2nd script that goes and searches for different keywords related to our business on mastadon and crafts responses that may or may not be related to out compnay. the reponses should be returned in the form of a structured output using pydantic. dont actually post the reponses right now, expose a flag we can turn on to actually post them, but for now just print the outputs instead.
The scripts should be put into a fodler called workshop1 and use uv to run

2.
can you add a flag to confirm if we want to make a post to @workshop-1/post_generator.py

3.
in @workshop-1/keyword_responder.py can t we use reponse format={pydantic_class}

4.
for @workshop-1/keyword_responder.py only have the top 5 posts get passed to the llm

5.
we dont need the models generating the post content for us, we can just add that back in after. same with author and original post id


# Workshop 2

1.
Please study the content of workshop-2 folder carefuly and integrate image generation into our agent. In particular, we need to at first generate a prompt for the diffusion model and then call the image generation api to create an image based on that prompt. Finally, we need to have the agent post the generated image along with the social media post on the user's Mastodon account. Make sure to handle any necessary authentication and API calls properly.

2.
add telegram bot integration so we can approve posts before they go live. send the generated post to telegram and wait for approve/reject button click


# Workshop 3

Prerequisites: GCloud account with billing setup ($300 free credits), gcloud CLI installed (brew install google-cloud-sdk on Mac)

1.
which gcloud account am I logged into and which projects does it have? Set the default project to the one with iap in the name

2.
can we deploy a e2 vm to that project? Please turn on any necessary apis

3.
the billing is setup. and let's use e2-micro size

4.
we have this virtual machine in gcloud. you have the gcloud cli to ssh into that machine. We need to install sqlite

5.
let's also deploy a fastapi server that uses that database and make sure the fast api is setup properly as a service on linux so it persists when we restart the instance

(Optional) 6.
can you set up a cron job to automatically run the post generator every day at 9am? skip the telegram approval for automated posts

7.
let's kill that vm so we save costs

8.
delete that too and find any other items to clean up

# Workshop 4

9.
We want to create a RAG system on the embeddings of the documentation and the posts; store the vector embeddings using sqlite-vec with the sqllite database. Chunk the documentation. 

10.
For retrieval use a hybrid search strategy with both keyword and cosine similarity (50-50). Use bm25 with FTS5 sqllite plugin. Include comments about memory/context management, use the top 10 results are used in context.

11.
Create a comment listener that can look for new comments to our posts and respond to the comments.  System prompt should include the style and retrieved context and the comment. 

12.
Create an autopost feature that uses a listener that looks for when there are significant diffs in the business-doc folder and add them to embeddings and make a post about the new content

13.
Create api.py endpoints for toggling listeners