# FOR ALL

This project has been set up with uv. You should be able to use 

```bash
uv run workshop-{n}/[file].py
```

to see the code in action.

Workshop 1 overview:

## Post generator

```bash
uv run workshop-1/post-generator.py --post
```

Takes the company docs and generates a post using a free model from OpenRouter and posts it on Mastodon. Remove the --post flag if you do not want to publish the post. 

## Keyword Responder

```bash
uv run workshop-1/keyword-responder.py --post
```

Searches 3 keywords based on Emanon (the company described in the business-docs folder) and uses structured outputs to generate responses to the top 5 posts.

Same as above, you can remove the --post flag if you do not want to publish the responses (highly recommended). The one part of the project that is untested because it is spamming other people's posts.

# WORKSHOP DESIGNERS

Please put the content for your workshop in a folder called workshop-[your workshop number]

Please build upon the previous workshop's code, so we have one cohesive project by the last workshop.

I have included reference docs for Emanon in this repo as well in the business-docs folder that you can use as well.

Also please READ and TEST your code so that you know what it is doing and actually works, this will be used as ground truth for the other TAs, so any issues could derail things very quickly.

I also have kept a file highlighting the ALL prompts I used to get to my code for reference as well, I encourage you to do the same. 

# FOR TAs

The expected code for each day's workshop can be found in its respective folder. 

Each folder only contains the code needed for that workshop and does not include the previous day's code, since the students code will most likely look very different. 

They are there so that you can check to see what working function calls and env setup for the different services looks like so that you can diagnose when there is an issue with the student's implementation.
