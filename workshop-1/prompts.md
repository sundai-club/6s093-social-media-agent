Workshop 1:

1. 
we want to make a script that does the following: reads the docs from the business docs folder and passed them into a llm and generates a social media post for it and then posts it on the users mastodon account. we want to use openrouter and the openai api library for running teh llm. creds for the mastodon account will be in an .env file. we then also want to have a 2nd script that goes and searches for different keywords related to our business on mastadon and crafts responses that may or may not be related to out compnay. the reponses should be returned in teh form of a structured output using pydantic. dont actually post the reponses right now, expose a flag we can turn on to actually post them, but for now just print the outputs instead.
The scripts should be put into a fodler called workshop1 and use uv to run

2. 
can you add a flag to confirm if we want to make a post to @workshop-1/post_generator.py

3. 
in @workshop-1/keyword_responder.py can t we use reponse format={pydantic_class}

4.
for @workshop-1/keyword_responder.py only have the top 5 posts get passed to the llm

5.
we dont need the models generating the post content for us, we can just add that back in after. same with author and original post id

Workshop 2:


