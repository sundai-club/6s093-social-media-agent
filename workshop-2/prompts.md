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

1. Please study the content of workshop-2 folder carefuly and integrate image generation into our agent. In particular, we need to at first generate a prompt for the diffusion model and then call the image generation api to create an image based on that prompt. Finally, we need to have the agent post the generated image along with the social media post on the user's Mastodon account. Make sure to handle any necessary authentication and API calls properly. The diffusion model is fine-tuned to generate my dog. To invoke the model to generate my dog exactly, include the phrase "a cartoon noir style image of the djeny dog dressed as a detective doing ..". Your job is to come up with what it should be doing to illustrate the post you are making.

Here are the instructions how to integrate the image generative model into the python code:

```
import replicate
output = replicate.run(
    "sundai-club/artems_dog_model:7103c7f706fe1429cf4bdb282ee81dfc218d643788b56f28dc6549c7dfb70967",
    input={
        "prompt": "A cartoon noir style image of the djeny dog dressed as a detective scrolling twitter",
        "num_inference_steps": 28, # typically need ~30 for "dev" model. Less steps == faster generation, but the quality is worse
        "guidance_scale": 7.5,     # how much attention the model pays to the prompt. Try different values between 1 and 50 to see
        "model": "dev",        # after fine-tuning you can use "schnell" model to generate images faster. In that case put num_inference_steps=4
    }
)
```

2. <You might need to fix a few errors after that>