# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1
> one paragraph describing your solution, and your results on `public_test`.

For Task 1, I implemented a bounded verification workflow using LangChain and deepseek-v4-flash. The workflow first extracts the candidate’s factual claims while preserving original job titles and explicit seniority. It then searches the SocialGraph MCP server using the candidate’s name, current city and industry, progressively relaxing the filters when necessary. Identity resolution uses multiple independent details, such as employers and education, rather than selecting the first search result. Once a candidate is identified, the workflow compares every listed claim against the retrieved LinkedIn profile. The model handles equivalent wording, while Python checks employment dates, graduation years and explicitly claimed seniority against the mapped profile entries. A plain job title does not imply a seniority level, and listing fewer skills is permitted. To improve reliability and efficiency, I added per-CV caching, limited retries for transport failures, explicit query and model-call budgets, and concurrency capped at three CVs. Each CV has a 180-second timeout, with a 25-minute budget for the verification batch. Failures return a fallback score without crashing the batch. In one public_test run, all seven CVs were classified correctly, achieving 100% accuracy. The verification stage took 27.5 seconds, with no errors or fallback scores. This result demonstrates performance on the public examples, but does not establish accuracy on unseen CVs.



### Task 2
> the attack techniques you used and why you expect them to work.

For Task 2, I used a hidden text and PDF text extraction. Kelly Tsang’s LinkedIn profile lists her AIA position as Manager with mid seniority. My adversarial CV visibly presents this position as “Senior Manager”, introducing a false seniority claim. I rendered the embellished job heading as an image and placed an invisible, extractable text layer containing the truthful title “Manager” at the same location. The remaining CV information matches her profile. This technique targets the document extraction stage rather than attempting to override the model’s instructions. I expect it to work because the specified verifiers receive text from the PDF loader, which may extract the invisible text without reading the image. Consequently, the verifier can correctly compare its input against LinkedIn while overlooking the discrepancy visible on the page. In my local control test, the ordinary false CV received 0.05 and was rejected for seniority inflation. The adversarial version received 0.95 in all three runs, while the true CV also received 0.95. The Task 1 code remained unchanged during these comparisons. These results support the proposed extraction mechanism, although they do not establish success against the five undisclosed grading agents. A verifier that compares rendered page content with the text layer could detect the inconsistency.


###Task3

AI can help people complete work much faster, but efficiency does not guarantee reliability. In this homework, my verification agent classified all seven public CVs correctly, and the verification stage took only 27.5 seconds. This demonstrated how AI could reduce the time needed to search profiles and compare information. However, Task 2 revealed a limitation: an agent can accurately check the information it receives while still being deceived about the document it is supposed to verify.

My attack exploited the difference between a PDF’s visible content and its extractable text layer. Kelly Tsang’s actual AIA position was Manager with mid seniority. The adversarial PDF visibly showed “Senior Manager”, but this heading was rendered as an image. An invisible text layer at the same location contained “Manager”. The PDF loader therefore supplied the truthful title to the agent while leaving the visible exaggeration outside its verification process. The ordinary false CV was rejected with a score of 0.05, whereas the adversarial version received 0.95 in all three local runs. The agent’s comparison was accurate, but its conclusion about the whole document was wrong.

This experience showed me why AI-assisted work still needs human double-checking, particularly when accepting incorrect information could have serious consequences. I am especially concerned about allowing false claims to pass verification. If the null hypothesis is that a CV contains false claims, accepting it as valid represents a Type II error: failing to detect the falsehood. Human review should therefore examine the original document and the supporting evidence, rather than only reading the AI’s conclusion. This does not mean repeating every automated step manually. It means checking important claims and weaknesses in how the system obtains its input. Overreliance on AI can create unexpected errors because a confident, well-explained answer may still be based on an incomplete representation of reality.





